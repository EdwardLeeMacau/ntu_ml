import argparse
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import pysnooper
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from model import Model, source_transform, target_transform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from trainer import (DecisionBoundaryIterativeRefinementTrainer,
                     DomainAdversarialTrainer)
from utils import same_seeds

# load hyperparameters from yaml
with open("params.yaml", "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

# Determine unique experiment ID
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# fix random seed for reproducibility
same_seeds(hparams["seed"])
root_dir = hparams['env']['dataset']
ckpt_dir = os.path.join(hparams['env']['checkpoint'], timestamp)

# write back to hparam
hparams['env']['checkpoint'] = ckpt_dir

def adaptive_lambda(curr: int, total: int, k: float = 10):
    """ adaptive lambda function, return lambda in range [0, 1] given x """
    x = curr / total

    # sigmoid function with shifting and scaling
    lambda_ = (2 / (1 + np.exp(-k*x))) - 1
    return lambda_

def train():
    # Imbalance case
    # Labeled data: 5000 images
    # Unlabeled data: 100k images
    source_dataset = ImageFolder(
        os.path.join(root_dir, 'train_data'), transform=source_transform
    )
    target_dataset = ImageFolder(
        os.path.join(root_dir, 'test_data'), transform=target_transform
    )

    # Initialize model from scratch
    model = Model()

    # Initialize checkpoint directory
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load model from checkpoint
    if 'load-dir' in hparams['training']:
        param_dir = hparams['training']['load-dir']

        state_dict = torch.load(param_dir)
        model.load_state_dict(state_dict)

        print(f'Load model from {param_dir}')

    # Domain Adversarial Training
    if 'dann' in hparams['training']:
        config = hparams['training']['dann']

        model = DomainAdversarialTrainer(
            config, { 'source': source_dataset, 'target': target_dataset }
        ).fit()

        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model-dann.pth'))

    # Decision Boundary Iterative Refinement Training
    if 'dirt-t' in hparams['training']:
        config = hparams['training']['dirt-t']

        teacher = deepcopy(model)
        model = DecisionBoundaryIterativeRefinementTrainer(
            config, { 'source': source_dataset, 'target': target_dataset },
            model=model, teacher=teacher
        ).fit()

        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model-dirtt.pth'))

    return model

def inference(model: nn.Module):
    # Inference
    result = []

    model = model.cuda()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(
        os.path.join(root_dir, 'test_data'), transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
    for (x, _) in tqdm(dataloader, ncols=0, desc='Inference'):
        x = x.cuda()

        logits = model(x)

        x = torch.argmax(logits, dim=1).cpu().detach().numpy()
        result.append(x)

    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv('DaNN_submission.csv', index=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--load-ckpt', type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        model = train()
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))

        inference(model)

    if args.inference:
        model = Model()

        state_dict = torch.load(args.load_ckpt)
        model.load_state_dict(state_dict)

        inference(model)

if __name__ == '__main__':
    main()
