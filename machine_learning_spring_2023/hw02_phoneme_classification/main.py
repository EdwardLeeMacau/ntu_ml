import argparse
import json
import os
from typing import Dict

import torch
import torch.nn as nn
import yaml
from dataset import TEST, TRAIN, VAL, LibriSeqDataset, preprocess_seqdata
from model import SeqTagger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import _optimizer, _scheduler, flatten, same_seeds

# data parameters
with open('params.yaml', 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

# dataset
dataset_dir = hparams['env']['dataset']

# store dir
checkpoint_dir = hparams['env']['checkpoint']

n_frames = hparams['model']['n-frames'] # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = hparams['ratio']          # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = hparams['seed']                                                  # random seed
batch_size = hparams['batch-size']['train']                             # batch size
num_epoch = hparams['epochs']                                           # the number of training epoch
model_path = os.path.join(checkpoint_dir, 'model.ckpt')                 # the path where the checkpoint will be saved
# last_model_path = os.path.join(checkpoint_dir, './model_end.ckpt')    # the path where the last iterated model will be saved

optimizer_params = hparams['optimizer']
optimizer_kwargs = optimizer_params['kwargs']

scheduler_params = hparams['scheduler']
scheduler_kwargs = scheduler_params['kwargs']

# model
model_params = hparams['model']
model_params['input_dim'] = 39 * n_frames

del model_params['n-frames']

same_seeds(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loader(splits=[TRAIN, VAL]) -> Dict[str, DataLoader]:
    kwargs = {
        'feature_dir': f'{dataset_dir}/feat',
        'phone_path': dataset_dir,
        'concat_nframes': n_frames,
        'train_ratio': train_ratio,
        'random_seed': seed
    }

    dataset = {
        s: LibriSeqDataset(preprocess_seqdata(split=s, **kwargs)) for s in splits
    }

    # preprocess data
    return {
        s: DataLoader(
            dataset[s], collate_fn=(dataset[s].test_collate_fn if s == TEST else dataset[s].collate_fn),
            batch_size=batch_size, shuffle=(True if s == TRAIN else False)
        ) for s in splits
    }

def train():
    # create dataloader
    dataloader = get_loader()
    train_dataloader = dataloader[TRAIN]
    val_dataloader = dataloader[VAL]

    # modify params before constructing model, optimizer and scheduler
    # also dropped arguments not recognizable by Torch
    if scheduler_params['kwargs']['step_unit'] == "epoch":
        scheduler_params['kwargs']['step_size'] *= len(train_dataloader)

    del scheduler_params['kwargs']['step_unit']

    # create logger
    # Reference: https://pytorch.org/docs/stable/tensorboard.html
    writer = SummaryWriter()

    # create model, define a loss function, and optimizer
    model = SeqTagger(**model_params).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataloader.dataset.IGN)
    optimizer = _optimizer[optimizer_params['name']](model.parameters(), **optimizer_kwargs)
    scheduler = _scheduler[scheduler_params['name']](optimizer, **scheduler_kwargs)

    # utility for calculating accuracy
    val_size = val_dataloader.dataset.num_tokens

    progress_bar = tqdm(total=num_epoch * len(dataloader[TRAIN]), desc='Training: ')

    best_acc, best_loss = 0.0, float('inf')
    val_acc, val_loss = 0.0, float('inf')

    for epoch in range(num_epoch):
        # training
        model.train()
        for i, batch in enumerate(dataloader[TRAIN]):
            x, y = batch

            mask = x['mask']
            size = mask.sum()

            x['sequence'] = x['sequence'].to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # get the index of the class with the highest probability
            _, pred = torch.max(outputs, 1)
            y, pred = y[mask], pred[mask]
            accuracy = (pred.detach() == y.detach()).sum().item() / size

            # record the loss and accuracy.
            n_iter = epoch * len(dataloader[TRAIN]) + i
            writer.add_scalar('Loss/train', loss.item(), n_iter)
            writer.add_scalar('Accuracy/train', accuracy, n_iter)

            progress_bar.update()
            progress_bar.set_postfix(next=len(dataloader[TRAIN]) - i - 1, acc=val_acc, loss=val_loss)

        # TODO: Decrease evaluation frequency
        # validation
        val_acc = 0.0
        val_loss = 0.0

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                x, y = batch

                mask = x['mask']
                x['sequence'] = x['sequence'].to(device)
                y = y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                # accumulate loss for statistics
                val_loss += loss.item()

                # get the index of the class with the highest probability
                _, pred = torch.max(outputs, 1)
                y, pred = y[mask], pred[mask]

                val_acc += (pred.cpu() == y.cpu()).sum().item()

        # TODO: handle case with size of validation set is 0 (Train without validation set)
        # Need to suppress ZeroDivisionError
        val_acc = val_acc / val_size
        val_loss = val_loss / len(dataloader[VAL])

        n_iter = (epoch + 1) * len(dataloader[TRAIN])
        writer.add_scalar('Loss/validation', val_loss, n_iter)
        writer.add_scalar('Accuracy/validation', val_acc, n_iter)

        progress_bar.set_postfix(next=len(dataloader[TRAIN]), acc=val_acc, loss=val_loss)

        # if the model improves, save a checkpoint at this epoch
        # TODO: move model to cpu before saving checkpoint
        # TODO: save checkpoint
        if val_acc > best_acc:
            best_acc, best_loss = val_acc, val_loss
            torch.save(model.state_dict(), model_path)

        # TODO: implement early-stop
        pass

    # TODO: handle case with size of validation set is 0 (Train without validation set)
    # model = model.cpu()
    # torch.save(model.state_dict(), last_model_path)

    with open('metrics.json', 'w') as f:
        json.dump({ 'accuracy': best_acc, 'loss': best_loss }, f)

    writer.add_hparams(
        flatten(hparams), metric_dict={'hparam/accuracy': best_acc, 'hparam/loss': best_loss }
    )

@torch.no_grad()
def test():
    # create dataloader
    dataloader = get_loader([TEST])[TEST]

    # create model
    model = SeqTagger(**model_params)
    model.load_state_dict(torch.load(model_path))
    # model.to(device)

    # inference
    predictions = []
    model.eval()
    for i, x in enumerate(tqdm(dataloader, desc='Inferring: ')):
        length = x['length']

        # x['sequence'] = x['sequence'].to(device)
        outputs = model(x)

        _, pred = torch.max(outputs, 1)
        pred = pred.cpu().numpy()
        for p, l in zip(pred, length):
            predictions.extend(p[:l].tolist())

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(predictions):
            f.write('{},{}\n'.format(i, y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()
