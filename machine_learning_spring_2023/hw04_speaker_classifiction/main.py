import random
import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import yaml
from dataset import InferenceDataset, MyDataset, metadata
from model import Classifier, AdditiveMarginSoftmaxLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import _optimizer, flatten, same_seeds


# TODO: Modify validation dataset such that it behaves similar to InferenceDataset.
def get_dataloader(data_dir: str, batch_size: int, n_workers: int, ratio: float = 0.9, segment_len: int = 128):
    """ Generate dataloader. """

    def collate_batch(batch):
        # Process features within a batch.
        """ Collate a batch of data. """
        mel, length, speaker = zip(*batch)
        length = torch.LongTensor(length)

        # Because we train the model batch by batch, we need to pad the features
        # in the same batch to make their lengths the same pad log 10^(-20) which
        # is very small value.
        mel = pad_sequence(mel, batch_first=True, padding_value=-20)

        # mel: (batch size, length, 40)
        return mel, length, torch.FloatTensor(speaker).long()

    speaker_num, _, data = metadata(data_dir)

    # Split dataset into training dataset and validation dataset
    random.shuffle(data)

    size = int(ratio * len(data))
    train_set = MyDataset(data_dir, data[:size], segment_len)
    valid_set = MyDataset(data_dir, data[size:], float('inf'))

    kwargs = {
        'num_workers': n_workers,
        'pin_memory': True,
        'collate_fn': collate_batch,
    }

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, batch_size=batch_size, **kwargs)
    valid_loader = DataLoader(valid_set, batch_size=1, **kwargs)

    return train_loader, valid_loader, speaker_num


"""# Learning rate schedule
- For transformer architecture, the design of learning rate schedule is different from that of CNN.
- Previous works show that the warmup of learning rate is useful for training models with transformer architectures.
- The warmup schedule
  - Set learning rate to 0 in the beginning.
  - The learning rate increases linearly from 0 to initial learning rate during warmup period.
"""

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.

        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.

        num_training_steps (:obj:`int`):
            The total number of training steps.

        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).

        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model: nn.Module, criterion, device):
    """ Forward a batch through the model. """
    mels, length, labels = batch

    mels = mels.to(device)
    length = length.to(device)
    labels = labels.to(device)

    outs = model(mels, length)

    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    prediction = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((prediction == labels).float())

    return loss, accuracy

@torch.no_grad()
def validate(dataloader: DataLoader, model: nn.Module, criterion: nn.Module, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    for batch in dataloader:
        loss, accuracy = model_fn(batch, model, criterion, device)

        running_loss += loss.item()
        running_accuracy += accuracy.item()

    model.train()

    return running_accuracy / len(dataloader), running_loss / len(dataloader)

def train():
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    seed = hparams['seed']
    same_seeds(seed)

    data_dir = hparams['env']['dataset']
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(hparams['env']['checkpoint'], timestamp)
    batch_size = hparams['batch-size']

    # Update 'epochs' in params.yaml
    valid_steps = 1000
    warmup_steps = 1000
    save_steps = 1000
    total_steps = 35000

    os.makedirs(save_path)

    # create training dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, speaker_num = get_dataloader(
        data_dir, batch_size, 8, ratio=hparams['ratio'], segment_len=hparams['segment_len']
    )
    train_iterator = iter(train_loader)

    # create model and trainer
    model_kwargs = hparams['model']
    model = Classifier(n_speakers=speaker_num, **model_kwargs).to(device)
    # print(model)

    # TODO: Modify loss to enhance performance
    criterion = AdditiveMarginSoftmaxLoss()

    optimizer_params = hparams['optimizer']
    optimizer_kwargs = optimizer_params['kwargs']
    optimizer = _optimizer[optimizer_params['name']](model.parameters(), **optimizer_kwargs)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # utilities
    writer = SummaryWriter(log_dir=os.path.join('runs', timestamp))

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=total_steps, ncols=0, desc="Train")
    for step in range(1, total_steps + 1):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)

        # Update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        writer.add_scalars('Loss', {'train': loss.item()}, step)
        writer.add_scalars('Accuracy', {'train': accuracy.item()}, step)
        pbar.update()

        # Do validation
        if step % valid_steps == 0:
            accuracy, loss = validate(valid_loader, model, criterion, device)
            writer.add_scalars('Loss', {'validation': loss}, step)
            writer.add_scalars('Accuracy', {'validation': accuracy}, step)

            pbar.set_postfix(loss=f"{loss:.2f}", accuracy=f"{accuracy:.2%}", step=step)

            # keep the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state_dict = model.state_dict()

        # Save the best model so far.
        if step % save_steps == 0 and best_state_dict is not None:
            ckpt_path = os.path.join(save_path, f"model-{step:08d}.pt")
            torch.save(best_state_dict, ckpt_path)
            pbar.write(f"Step {step}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()

    # Log hyperparameters
    del hparams['env']
    writer.add_hparams(
        flatten(hparams), metric_dict={'hparam/accuracy': best_accuracy }
    )


@torch.no_grad()
def test(model_path):
    def inference_collate_batch(batch):
        """ Collate a batch of data. """
        feat_paths, length, mels = zip(*batch)
        return feat_paths, torch.LongTensor(length), torch.stack(mels)

    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = hparams['env']['dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )

    speaker_num = len(mapping["id2speaker"])

    model_kwargs = hparams['model']
    model = Classifier(n_speakers=speaker_num, **model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = [["Id", "Category"]]
    for feat_paths, length, mels in tqdm(dataloader, ncols=0, desc="Inferring"):
        mels = mels.to(device)
        length = length.to(device)

        outs = model(mels, length)
        pred = outs.argmax(1).cpu().numpy()
        for feat_path, pred in zip(feat_paths, pred):
            results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load-ckpt', type=str)
    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test(model_path=args.load_ckpt)

if __name__ == "__main__":
    main()