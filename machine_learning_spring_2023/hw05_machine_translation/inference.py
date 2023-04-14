import logging
from argparse import Namespace
from pathlib import Path

import fairseq
import torch
import utils
import yaml
from fairseq.models.transformer import base_architecture
from model import build_model
from tqdm import tqdm
from utils import same_seeds

from dataset import get_data_iterator_kwargs, setup_task

proj = "hw5.seq2seq"
logger = logging.getLogger(proj)

# env = fairseq.utils.CudaEnvironment()
# fairseq.utils.CudaEnvironment.pretty_print_cuda_env_list([env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('params.yaml', 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

seed = hparams['seed']
same_seeds(seed)

config = Namespace(
    datadir = hparams['env']['binarized'],
    savedir = "./checkpoints/rnn",
    source_lang = 'en',
    target_lang = 'zh',

    # cpu threads when fetching & processing data.
    num_workers=2,

    # max number of tokens simultaneously loaded in GPU.
    max_tokens=8192,
    **hparams['decoding'],
)

task = setup_task(data_dir=config.datadir, source_lang=config.source_lang, target_lang=config.target_lang)

def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"

    fpath = Path(config.savedir) / name
    if not fpath.exists():
        raise FileNotFoundError(f"no checkpoints found at {fpath}!")

    ckpt = torch.load(fpath)

    model.load_state_dict(ckpt["model"])
    stats = ckpt["stats"]
    step = "unknown"

    if optimizer != None:
        optimizer._step = step = ckpt["optim"]["step"]

    logger.info(f"Loaded checkpoint {fpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")

def decode(toks, dictionary):
    # convert from Tensor to human readable sentence
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(generator, sample, model):
    gen_out = generator.generate([model], sample)

    srcs, hyps, refs = list(), list(), list()
    for i in range(len(gen_out)):
        # for each sample, collect the input, hypothesis and reference, later be used to calculate BLEU
        srcs.append(decode(
            fairseq.utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 indicates using the top hypothesis in beam
            task.target_dictionary,
        ))
        refs.append(decode(
            fairseq.utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
            task.target_dictionary,
        ))

    return srcs, hyps, refs

@torch.no_grad()
def generate_prediction(model, task, outfile="./prediction.txt"):
    # utilities
    itr_kwargs = get_data_iterator_kwargs(task, max_tokens=config.max_tokens)

    # fairseq's beam search generator
    # given model and input seqeunce, produce translation hypotheses by beam search
    generator = task.build_generator([model], config)

    # load test dataset, compact them as translation task
    task.load_dataset(split='test', epoch=1)
    itr = task.get_batch_iterator(
        dataset=task.dataset('test'), epoch=1, **itr_kwargs
    ).next_epoch_itr(shuffle=False)

    # inference with test dataset.
    idxs = []
    hypothesis = []

    model.eval()
    for sample in tqdm(itr, desc=f"prediction", ncols=0):
        sample = fairseq.utils.move_to_cuda(sample, device=device)
        s, h, r = inference_step(generator, sample, model)

        hypothesis.extend(h)
        idxs.extend(list(sample['id']))

    # sort based on the order before preprocess
    hypothesis = [x for _, x in sorted(zip(idxs, hypothesis))]
    with open(outfile, "w") as f:
        for h in hypothesis:
            f.write(h + "\n")

if __name__ == "__main__":
    # Averaging model
    # checkdir=config.savedir
    # python ./fairseq/scripts/average_checkpoints.py \
    #   --inputs {checkdir} \
    #   --num-epoch-checkpoints 5 \
    #   --output {checkdir}/avg_last_5_checkpoint.pt

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
    model = build_model(arch_args, task)
    model = model.to(device)
    # checkpoint_last.pt : latest epoch
    # checkpoint_best.pt : highest validation bleu
    # avg_last_5_checkpoint.pt: the average of last 5 epochs
    try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
    # validate(model, task, criterion, log_to_wandb=False)

    generate_prediction(model, task)
