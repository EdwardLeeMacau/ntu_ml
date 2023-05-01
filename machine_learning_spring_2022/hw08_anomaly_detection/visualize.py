"""
Visualize the learned visual representations of the CNN model on the validation set
by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output of
both top & mid layers (You need to submit 2 images).
"""

import argparse

import pandas as pd
import os
import numpy as np
import torch
import yaml
from model import GaussianNoise
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from torchvision.utils import make_grid, save_image
from utils import same_seeds

with open('params.yaml', 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

root_dir = hparams['env']['dataset']

def prepare_dataset(x):
    return TensorDataset(torch.from_numpy(np.load(
        os.path.join(root_dir, x), allow_pickle=True
    )))

def prepare_dataloader(x):
    return DataLoader(
        TensorDataset(torch.from_numpy(np.load(
            os.path.join(root_dir, x), allow_pickle=True
        ))), batch_size=256, shuffle=True)

@torch.no_grad()
def visualize_dataset():
    # helper function to load dataset
    TRAIN, TEST = 'train', 'test'

    dataset = {
        'train': prepare_dataloader('trainingset.npy'),
        'test': prepare_dataloader('testingset.npy'),
    }

    # visualize training dataset
    x = next(iter(dataset[TRAIN]))[0]
    x = x.permute(0, 3, 1, 2)
    grid = make_grid(x, nrow=32, padding=2, pad_value=0) / 255
    save_image(grid, 'trainingset.png')

    # visualize testing dataset
    x = next(iter(dataset[TEST]))[0]
    x = x.permute(0, 3, 1, 2)
    grid = make_grid(x, nrow=32, padding=2, pad_value=0) / 255
    save_image(grid, 'testingset.png')

# Visualize image transformation
def visualize_transform():
    transform = transforms.Compose([
        GaussianNoise(0, 0.1),
    ])

    dataset = prepare_dataloader('trainingset.npy')
    x = next(iter(dataset))[0]
    x = x.permute(0, 3, 1, 2)
    x = 2 * (x / 255) - 1
    x = transform(x)

    x = (x + 1) / 2
    grid = make_grid(x, nrow=32, padding=2, pad_value=0)
    save_image(grid, 'transform.png')

    return

# Visualize ranking
def visualize_ranking():
    df = pd.read_csv('prediction.csv')
    df = df.sort_values(by=['score'], ascending=False)

    # print max and min score
    max_score, min_score = df['score'].max(), df['score'].min()
    print(f'{max_score=}, {min_score=}')

    # get top-k anomaly images
    k = 256
    top_k = df[:k]['ID'].values
    dataset = prepare_dataset('testingset.npy')

    x = torch.stack([dataset[i][0] for i in top_k])
    x = x.permute(0, 3, 1, 2)
    grid = make_grid(x, nrow=32, padding=2, pad_value=0) / 255
    save_image(grid, 'ranking.png')

    # iterate over the first 10 rows
    # for i, row in df[:10].iterrows():
    #     print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--visualize-dataset', action='store_true')
    parser.add_argument('--visualize-transform', action='store_true')
    parser.add_argument('--visualize-rank', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)

    args = parser.parse_args()

    same_seeds(args.seed)

    if args.visualize_dataset:
        visualize_dataset()

    if args.visualize_transform:
        visualize_transform()

    if args.visualize_rank:
        visualize_ranking()
