import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from dataset import InferenceDataset, MyDataset
from model import Classifier
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from tqdm.notebook import tqdm
from utils import same_seeds

same_seeds(87)

"""## Dataloader
- Split dataset into training dataset(90%) and validation dataset(10%).
- Create dataloader to iterate the data.
"""



def get_dataloader(data_dir, batch_size, n_workers):
    """ Generate dataloader. """

    def collate_batch(batch):
        # Process features within a batch.
        """ Collate a batch of data. """
        mel, speaker = zip(*batch)

        # Because we train the model batch by batch, we need to pad the features
        # in the same batch to make their lengths the same pad log 10^(-20) which
        # is very small value.
        mel = pad_sequence(mel, batch_first=True, padding_value=-20)

        # mel: (batch size, length, 40)
        return mel, torch.FloatTensor(speaker).long()

    dataset = MyDataset(data_dir)
    speaker_num = dataset.get_speaker_number()

    # Split dataset into training dataset and validation dataset
    train_len = int(0.9 * len(dataset))
    length = [train_len, len(dataset) - train_len]
    train_set, validation_set = random_split(dataset, length)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    valid_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

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


def model_fn(batch, model, criterion, device):
    """ Forward a batch through the model. """
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    prediction = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((prediction == labels).float())

    return loss, accuracy

@torch.no_grad()
def validate(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        loss, accuracy = model_fn(batch, model, criterion, device)
        running_loss += loss.item()
        running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)

def parse_args():
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    config = {
        "data_dir": hparams['env']['dataset'],
        "save_path": "model.ckpt",
        "batch_size": hparams['batch-size'],
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 70000,
    }

    return config


def train(
    data_dir,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    total_steps,
    save_steps,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!",flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = validate(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()

def inference_collate_batch(batch):
    """ Collate a batch of data. """
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)

# def parse_args():
#     """arguments"""
#     config = {
#         "data_dir": "./Dataset",
#         "model_path": "./model.ckpt",
#         "output_path": "./output.csv",
#     }

#     return config

@torch.no_grad()
def test(data_dir, model_path, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

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
    print(f"[Info]: Finish loading data!",flush = True)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!",flush = True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load-ckpt', type=str)
    args = parser.parse_args()

    if args.train:
        train(**parse_args())

    if args.test:
        test(model_path=args.load_ckpt, **parse_args())

if __name__ == "__main__":
    main()