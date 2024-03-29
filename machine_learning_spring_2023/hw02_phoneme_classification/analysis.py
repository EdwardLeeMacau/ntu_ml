import torch
import yaml
import os
from dataset import TRAIN, preprocess_data
from typing import Tuple
from matplotlib import pyplot as plt
from itertools import product
from torchsummary import summary
from model import SeqTagger
from utils import count_parameters
import json
import numpy as np


def z_score() -> Tuple[torch.Tensor, torch.Tensor]:
    """ Calculate mean and standard deviation of training dataset. """
    # data parameters
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    seed = hparams['seed']
    train_ratio = 0.75                          # the ratio of data used for training, the rest will be used for validation

    preprocess_kwargs = {
        'feature_dir': './libriphone/feat',
        'phone_path': './libriphone',
        'concat_nframes': 1,
        'train_ratio': train_ratio,
        'random_seed': seed
    }

    # Expected to do z-score normalization with input data, seems they are normalized already.
    #
    # Return value:
    # mean=tensor([ 5.3525e-10, -3.4838e-10,  6.2296e-10, -5.1877e-10, -2.9086e-10,
    #         -1.0833e-09, -5.1536e-10, -2.8555e-10, -3.8021e-10, -1.1773e-10,
    #         -8.5643e-10,  6.0129e-10, -6.7021e-10,  1.0716e-09, -1.7705e-10,
    #         -1.6639e-09, -8.0378e-10, -1.0526e-09,  4.7562e-10, -8.6516e-10,
    #         -3.3107e-10, -9.3545e-10, -5.5725e-10, -7.9151e-10,  1.2622e-10,
    #         -1.6905e-10,  1.8240e-10, -1.6727e-10,  2.2702e-09, -9.2239e-11,
    #          3.9839e-10, -2.1627e-10,  6.7777e-10,  8.0844e-10,  2.6468e-10,
    #          8.1159e-10, -2.4370e-10,  1.0251e-10,  4.9391e-10])
    # std=tensor([0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992,
    #         0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992,
    #         0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992,
    #         0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992, 0.9992,
    #         0.9992, 0.9992, 0.9992])
    x, _ = preprocess_data(split=TRAIN, **preprocess_kwargs)
    return torch.std_mean(x, dim=0)

def plot_q1_learning_curve():
    """ Plot different learning curves and save to local dir. """
    metrics = 'accuracy'
    for split in ('train', 'validation'):
        dirname = os.path.join('report', f'{metrics}.{split}')

        data = {}

        fname = os.path.join(dirname, f'shallow.json')
        with open(fname, 'r') as f:
            tmp = np.array(json.load(f)).T
            data['shallow'] = tmp[1:]

        fname = os.path.join(dirname, f'dropout-50.json')
        with open(fname, 'r') as f:
            tmp = np.array(json.load(f)).T
            data['deep'] = tmp[1:]

        fig = plt.figure(figsize=(12.8, 7.2))
        for k, v in data.items():
            plt.plot(v[0], v[1], label=k)

        plt.title(f'Shallow model v.s. Deep model')
        plt.legend()
        plt.savefig(f'q1.{split}.png')
        plt.cla()

    return

def plot_q2_learning_curve():
    """ Plot different learning curves and save to local dir. """
    for metrics, split in product(('accuracy', 'loss'), ('train', 'validation')):
        dirname = os.path.join('report', f'{metrics}.{split}')

        data = {}
        for dropout_rate in (25, 50, 75):
            fname = os.path.join(dirname, f'dropout-{dropout_rate}.json')
            with open(fname, 'r') as f:
                tmp = np.array(json.load(f)).T
                data[dropout_rate] = tmp[1:]

        fig = plt.figure(figsize=(12.8, 7.2))
        for k, v in data.items():
            plt.plot(v[0], v[1], label=f'dropout=0.{k}')

        plt.title(f'{metrics}.{split}')
        plt.legend()
        plt.savefig(f'{metrics}.{split}.png')
        plt.cla()

    return

def find_trainable_parameters():
    # data parameters
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    n_frames = hparams['model']['n-frames'] # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
    frame_dim = 39 * n_frames

    model_params = hparams['model']
    model_params['input_dim'] = frame_dim
    del model_params['n-frames']

    model = SeqTagger(**model_params)
    model = model.train()
    trainable_param = count_parameters(model)
    print(f'{trainable_param=}')


plot_q1_learning_curve()
