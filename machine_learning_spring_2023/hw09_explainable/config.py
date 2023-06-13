"""
Declare global variables here.
"""

import argparse
import yaml
import torch
from dataset import FoodDataset, get_paths_labels
from model import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='param.yaml', type=str)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


args = argparse.Namespace(**config)                     # Export this variable to be used
                                                        # in other files

model = Classifier().cuda()                             # Export this variable to be used
                                                        # in other files
ckpt = torch.load(args.checkpoint)
model.load_state_dict(ckpt['model_state_dict'])

train_paths, train_labels = get_paths_labels(args.dataset_dir)
train_set = FoodDataset(train_paths, train_labels, mode='eval')

img_indices = [i for i in range(10)]
images, labels = train_set.get_batch(img_indices)       # Export this variable to be used
                                                        # in other files

def initialize():
    return args, model, img_indices, images, labels
