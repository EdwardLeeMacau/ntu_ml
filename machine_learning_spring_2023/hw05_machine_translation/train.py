import logging
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import fairseq
import numpy as np
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
import utils
from fairseq.data import EpochBatchIterator, iterators
from fairseq.models.transformer import base_architecture
from inference import inference_step
from model import LabelSmoothedCrossEntropyCriterion, NoamOpt, build_model
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import same_seeds

from dataset import get_data_iterator_kwargs, setup_task

with open('params.yaml', 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

seed = hparams['seed']
same_seeds(seed)

data_dir = hparams['env']['binarized']
binarized = hparams['env']['binarized']
checkpoint = hparams['env']['checkpoint']
dataset_name = 'ted2020'
# prefix = Path(data_dir).absolute()

source_lang = 'en'
target_lang = 'zh'

"""# Configuration for experiments"""

config = Namespace(
    datadir = binarized,
    savedir = "./checkpoints/rnn",
    source_lang = source_lang,
    target_lang = target_lang,

    # cpu threads when fetching & processing data.
    num_workers=2,

    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=1,

    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=hparams['scheduler']['kwargs']['lr_factor'],
    lr_warmup=hparams['scheduler']['kwargs']['lr_warmup'],

    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=hparams['clip_norm'],

    # maximum epochs for training
    max_epoch=hparams['epochs'],

    # decoding
    **hparams['decoding'],

    # checkpoints
    keep_last_epochs=5,

    # logging
    use_wandb=True,
)

### Logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO", # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)

proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb:
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

### CUDA Environments

env = fairseq.utils.CudaEnvironment()
fairseq.utils.CudaEnvironment.pretty_print_cuda_env_list([env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Dataloader: setup translation task in Fairseq.TranslationTask schema

# TODO: combine if you have back-translation data.
task = setup_task(data_dir=data_dir, source_lang=source_lang, target_lang=target_lang)
task.load_dataset(split="train", epoch=1, combine=True)
task.load_dataset(split="valid", epoch=1)

### Architecture Related Configuration

# For strong baseline, please refer to the hyperparameters for *transformer-base* in
# Table 3 in [Attention is all you need](#vaswani2017)
# https://arxiv.org/pdf/1706.03762.pdf

arch_args = hparams['model']
arch_args.update(utils.flatten(arch_args['encoder'], prefix='encoder', delim='_'))
arch_args.update(utils.flatten(arch_args['decoder'], prefix='decoder', delim='_'))
arch_args.pop('encoder')
arch_args.pop('decoder')
arch_args = Namespace(**arch_args)

def add_transformer_args(args):
    args.encoder_attention_heads=8
    args.encoder_normalize_before=True

    args.decoder_attention_heads=8
    args.decoder_normalize_before=True

    args.activation_fn="relu"
    args.max_source_positions=1024
    args.max_target_positions=1024

    # patches on default parameters for Transformer (those not set above)
    base_architecture(args)

add_transformer_args(arch_args)

if config.use_wandb:
    wandb.config.update(vars(arch_args))

### Instance construction

model = build_model(arch_args, task)
generator = task.build_generator([model], config)

# generally, 0.1 is good enough
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)

optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim,
    factor=config.lr_factor,
    warmup=config.lr_warmup,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001)
)

def train_one_epoch(epoch_itr: EpochBatchIterator, model, criterion, optimizer, accum_steps=1):
    # Gradient accumulation: update every accum_steps samples
    #
    # API References:
    # https://fairseq.readthedocs.io/en/latest/data.html#fairseq.data.GroupedIterator
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)

    stats = {"loss": []}
    scaler = GradScaler() # automatic mixed precision (amp)

    model.train()
    progress = tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", ncols=0, leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0

        # gradient accumulation: update every accum_steps samples
        for i, sample in enumerate(samples):
            # emptying the CUDA cache after the first step can reduce the chance of OOM
            if i == 1:
                torch.cuda.empty_cache()

            sample = fairseq.utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            # mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()

        # (sample_size or 1.0) handles the case of a zero gradient
        # grad norm clipping prevents gradient exploding
        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0))
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)

        scaler.step(optimizer)
        scaler.update()

        # logging
        loss_print = accum_loss/sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })

    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats


@torch.no_grad()
def validate(model, task, criterion, log_to_wandb=True):
    itr = task.get_batch_iterator(
        dataset=task.dataset('valid'), epoch=1,
        **get_data_iterator_kwargs(task, max_tokens=config.max_tokens)
    ).next_epoch_itr(shuffle=False)

    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []

    model.eval()
    progress = tqdm(itr, desc=f"validation", ncols=0, leave=False)
    for i, sample in enumerate(progress):
        sample = fairseq.utils.move_to_cuda(sample, device=device)
        net_output = model.forward(**sample["net_input"])

        # validation loss
        lprobs = F.log_softmax(net_output[0], -1)
        target = sample["target"]
        sample_size = sample["ntokens"]
        loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
        progress.set_postfix(valid_loss=loss.item())
        stats["loss"].append(loss)

        # do inference
        s, h, r = inference_step(generator, sample, model)
        srcs.extend(s)
        hyps.extend(h)
        refs.extend(r)

    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # calculate BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs

    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)

    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])

    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats


if __name__ == "__main__":
    # Prepare checkpointing directory
    savedir = Path(config.savedir).absolute()
    savedir.mkdir(parents=True, exist_ok=True)

    # Send model to CUDA
    model = model.to(device=device)
    criterion = criterion.to(device=device)

    # Prepare dataloader
    itr_kwarg = get_data_iterator_kwargs(task, max_tokens=config.max_tokens)
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset('train'), epoch=1, **itr_kwarg
    )

    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # Iteratively update model
        train_one_epoch(epoch_itr, model, criterion, optimizer, config.accum_steps)

        # validate model performance
        stats = validate(model, task, criterion)
        bleu, loss = stats['bleu'], stats['loss']

        # save epoch checkpoints
        epoch = epoch_itr.epoch
        ckpt = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }

        torch.save(ckpt, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

        # save epoch samples
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # get best valid bleu
        if getattr(validate, "best_bleu", 0) < bleu.score:
            validate.best_bleu = bleu.score
            torch.save(ckpt, savedir/f"checkpoint_best.pt")

        # remove too-old checkpoints
        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()

        # move to next epoch
        epoch_itr = task.get_batch_iterator(
            dataset=task.dataset('train'), epoch=epoch_itr.next_epoch_idx, **itr_kwarg
        )

raise

"""# Back-translation

## Train a backward translation model

1. Switch the source_lang and target_lang in **config**
2. Change the savedir in **config** (eg. "./checkpoints/transformer-back")
3. Train model

## Generate synthetic data with backward model

### Download monolingual data
"""

mono_dataset_name = 'mono'

mono_prefix = Path(data_dir).absolute() / mono_dataset_name
mono_prefix.mkdir(parents=True, exist_ok=True)

urls = (
    "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ted_zh_corpus.deduped.gz",
)
file_names = (
    'ted_zh_corpus.deduped.gz',
)

for u, f in zip(urls, file_names):
    path = mono_prefix/f
    if not path.exists():
        # !wget {u} -O {path}
        raise
    else:
        print(f'{f} is exist, skip downloading')
    if path.suffix == ".tgz":
        # !tar -xvf {path} -C {prefix}
        raise
    elif path.suffix == ".zip":
        # !unzip -o {path} -d {prefix}
        raise
    elif path.suffix == ".gz":
        # !gzip -fkd {path}
        raise

"""### TODO: clean corpus

1. remove sentences that are too long or too short
2. unify punctuation

hint: you can use clean_s() defined above to do this
"""



"""### TODO: Subword Units

Use the spm model of the backward model to tokenize the data into subword units

hint: spm model is located at DATA/raw-data/\[dataset\]/spm\[vocab_num\].model
"""



"""### Binarize

use fairseq to binarize data
"""

binpath = Path('./DATA/data-bin', mono_dataset_name)
src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
tgt_dict_file = src_dict_file
monopref = str(mono_prefix/"mono.tok") # whatever filepath you get after applying subword tokenization
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    # !python -m fairseq_cli.preprocess\
    #     --source-lang 'zh'\
    #     --target-lang 'en'\
    #     --trainpref {monopref}\
    #     --destdir {binpath}\
    #     --srcdict {src_dict_file}\
    #     --tgtdict {tgt_dict_file}\
    #     --workers 2
    raise

"""### TODO: Generate synthetic data with backward model

Add binarized monolingual data to the original data directory, and name it with "split_name"

ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]

then you can use 'generate_prediction(model, task, split="split_name")' to generate translation prediction
"""

# Add binarized monolingual data to the original data directory, and name it with "split_name"
# ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
# !cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin
# !cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx
# !cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin
# !cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx
raise

# hint: do prediction on split='mono' to create prediction_file
# generate_prediction( ... ,split=... ,outfile=... )

"""### TODO: Create new dataset

1. Combine the prediction data with monolingual data
2. Use the original spm model to tokenize data into Subword Units
3. Binarize data with fairseq
"""

# Combine prediction_file (.en) and mono.zh (.zh) into a new dataset.
#
# hint: tokenize prediction_file with the spm model
# spm_model.encode(line, out_type=str)
# output: ./DATA/rawdata/mono/mono.tok.en & mono.tok.zh
#
# hint: use fairseq to binarize these two files again
# binpath = Path('./DATA/data-bin/synthetic')
# src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
# tgt_dict_file = src_dict_file
# monopref = ./DATA/rawdata/mono/mono.tok # or whatever path after applying subword tokenization, w/o the suffix (.zh/.en)
# if binpath.exists():
#     print(binpath, "exists, will not overwrite!")
# else:
#     !python -m fairseq_cli.preprocess\
#         --source-lang 'zh'\
#         --target-lang 'en'\
#         --trainpref {monopref}\
#         --destdir {binpath}\
#         --srcdict {src_dict_file}\
#         --tgtdict {tgt_dict_file}\
#         --workers 2

# create a new dataset from all the files prepared above
# !cp -r ./DATA/data-bin/ted2020/ ./DATA/data-bin/ted2020_with_mono/

# !cp ./DATA/data-bin/synthetic/train.zh-en.zh.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.bin
# !cp ./DATA/data-bin/synthetic/train.zh-en.zh.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.idx
# !cp ./DATA/data-bin/synthetic/train.zh-en.en.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.bin
# !cp ./DATA/data-bin/synthetic/train.zh-en.en.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.idx
raise

"""Created new dataset "ted2020_with_mono"

1. Change the datadir in **config** ("./DATA/data-bin/ted2020_with_mono")
2. Switch back the source_lang and target_lang in **config** ("en", "zh")
2. Change the savedir in **config** (eg. "./checkpoints/transformer-bt")
3. Train model

# References

1. <a name=ott2019fairseq></a>Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., ... & Auli, M. (2019, June). fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations) (pp. 48-53).
2. <a name=vaswani2017></a>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017, December). Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 6000-6010).
3. <a name=reimers-2020-multilingual-sentence-bert></a>Reimers, N., & Gurevych, I. (2020, November). Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 4512-4525).
4. <a name=tiedemann2012parallel></a>Tiedemann, J. (2012, May). Parallel Data, Tools and Interfaces in OPUS. In Lrec (Vol. 2012, pp. 2214-2218).
5. <a name=kudo-richardson-2018-sentencepiece></a>Kudo, T., & Richardson, J. (2018, November). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 66-71).
6. <a name=sennrich-etal-2016-improving></a>Sennrich, R., Haddow, B., & Birch, A. (2016, August). Improving Neural Machine Translation Models with Monolingual Data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 86-96).
7. <a name=edunov-etal-2018-understanding></a>Edunov, S., Ott, M., Auli, M., & Grangier, D. (2018). Understanding Back-Translation at Scale. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 489-500).
8. https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus
9. https://ithelp.ithome.com.tw/articles/10233122
10. https://nlp.seas.harvard.edu/2018/04/03/attention.html
11. https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW05/HW05.ipynb
"""

