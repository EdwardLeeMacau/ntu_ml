import torch
import yaml
from dataset import TRAIN, preprocess_data

# data parameters
with open('params.yaml', 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

seed = hparams['seed']
n_frames = hparams['model']['n-frames']     # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
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
std, mean = torch.std_mean(x, dim=0)
print(f'{mean=}')
print(f'{std=}')

