import argparse
import json
import os
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from dataset import FoodDataset
from model import GaussianNoise
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler, autocast
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm import tqdm
from tqdm.auto import tqdm
from utils import (ModelCheckpointPreserver, _optimizer, _scheduler, flatten,
                   mixup, plot_confusion_matrix, same_seeds, sizeof_fmt)
from visualize import visualize_representation

"""# Configurations"""
with open('params.yaml', 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

# dataset
dataset_dir = hparams['env']['dataset']

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
#
# For ResNet family, initialize residual from zero leads slightly
# performance improvement. To use this feature, set argument
# zero_init_residual as True.
# https://arxiv.org/pdf/1706.02677.pdf
model = models.resnet50(num_classes=11, weights=None, progress=None)
model = model.to(device)

# The number of batch size.
seed = hparams['seed']
img_size = tuple(hparams['model']['resize'])
batch_size = hparams['batch-size']

# The number of training epochs.
n_epochs = hparams['epochs']

# Config for mixup augmentation
use_mixup = hparams['mixup']['enable']
alpha = hparams['mixup']['alpha']

# If no improvement in 'patience' epochs, early stop.
patience = hparams['early-stop']

# For the classification task, we use cross-entropy as the measurement of performance.
#
# Replace CrossEntropy by BCEWithLogitsLoss to have better flexibility
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
criterion = nn.BCEWithLogitsLoss()

same_seeds(seed)

# Reference: https://pytorch.org/vision/stable/transforms.html
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
validation_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_transform = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize(img_size),
    # You may add some transforms here.
    transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
    transforms.GaussianBlur(13),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomPosterize(bits=4, p=0.5),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0),
    GaussianNoise(mean=0, sigma=(0.01, 0.1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train():
    # Construct train and valid datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set = FoodDataset(f"{dataset_dir}/train", split="train", transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_set = FoodDataset(f"{dataset_dir}/valid", split="val", transform=validation_transform)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    optimizer_params = hparams['optimizer']
    optimizer_kwargs = optimizer_params['kwargs']

    scheduler_params = hparams['scheduler']
    scheduler_kwargs = scheduler_params['kwargs']

    if scheduler_params['kwargs']['step_unit'] == "epoch":
        scheduler_params['kwargs']['step_size'] *= len(train_loader)

    del scheduler_params['kwargs']['step_unit']

    # Create logger
    writer = SummaryWriter()

    # Imply mixup if enabled
    mx = (lambda x: mixup(x, alpha=alpha)) if use_mixup else (lambda x: x)

    # Create utilities
    # Automatic Mixed Precision: https://pytorch.org/docs/stable/amp.html#torch.autocast
    scaler = GradScaler()

    # store dir
    checkpoint_dir = os.path.join(hparams['env']['checkpoint'], str(uuid.uuid1()))
    os.makedirs(checkpoint_dir)
    preserver = ModelCheckpointPreserver(key='accuracy', k=hparams['env']['k-max-ckpt'], dirname=checkpoint_dir)

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = _optimizer[optimizer_params['name']](model.parameters(), **optimizer_kwargs)
    scheduler = _scheduler[scheduler_params['name']](optimizer, **scheduler_kwargs)

    progress_bar = tqdm(total=n_epochs * len(train_loader), desc='Training: ')

    """# Start Training"""

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0

    valid_acc, valid_loss = 0.0, float('inf')
    for epoch in range(n_epochs):
        # ------------------------------ Training ------------------------------
        model.train()

        # These are used to record information in training.
        for i, batch in enumerate(train_loader):
            x, y = mx(batch)
            x, y = x.to(device), y.to(device)

            # Forward the data.
            # with autocast():
            logits = model(x)
            loss = criterion(logits, y)

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            # scaler.scale(loss).backward()
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            scheduler.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()

            # record the loss and accuracy.
            n_iter = epoch * len(train_loader) + i
            writer.add_scalar('Loss/train', loss.item(), n_iter)
            writer.add_scalar('Accuracy/train', acc.item(), n_iter)

            postfix = {
                'acc': valid_acc,
                'loss': valid_loss,
                'memusage': sizeof_fmt(torch.cuda.max_memory_allocated()),
                'nextval': len(train_loader) - i - 1,
            }
            progress_bar.update()
            progress_bar.set_postfix(**postfix)

        # ------------------------------ Validation ------------------------------
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        prediction = []
        ground_truth = []

        # Iterate the validation set by batches.
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch

                x, y = x.to(device), y.to(device)
                logits = model(x)

                # compute the loss.
                loss = criterion(logits, y)
                pred = np.argmax(logits.cpu().data.numpy(), axis=1)

                # compute the metrics for current batch.
                prediction += pred.squeeze().tolist()
                ground_truth += y.argmax(dim=-1).cpu().tolist()
                acc = (logits.argmax(dim=-1) == y.argmax(dim=-1)).float().mean()

                # record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc.item())

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # cm = confusion_matrix(ground_truth, prediction)
        # cm_img = np.array(
        #     Image.open(plot_confusion_matrix(cm, list(range(11)))).convert("RGB")
        # ).transpose(2, 0, 1)

        # record the loss and accuracy.
        n_iter = (epoch + 1) * len(train_loader)
        writer.add_scalar('Loss/validation', valid_loss, n_iter)
        writer.add_scalar('Accuracy/validation', valid_acc, n_iter)
        # writer.add_image('ConfusionMatrix/validation', cm_img, n_iter)

        # save model if it performs well in validation set.
        stale = 0 if preserver.update(model, valid_acc, n_iter) else (stale + 1)
        if stale > patience:
            break

    best_iter, best_acc = preserver.get_best()
    with open('metrics.json', 'w') as f:
        json.dump({ 'accuracy': best_acc }, f)

    del hparams['env']['checkpoint']
    del hparams['env']['dataset']
    del hparams['env']['k-max-ckpt']
    del hparams['model']['resize']
    writer.add_hparams(
        flatten(hparams), metric_dict={'hparam/accuracy': best_acc }
    )

    # TODO: Enable t-SNE visualization for debugging
    best_model_dir = os.path.join(checkpoint_dir, f"model-{best_iter:08d}.pt")
    tsne = np.array(Image.open(visualize_representation(best_model_dir)).convert("RGB"))
    writer.add_image('t-SNE/validation', tsne)

@torch.no_grad()
def test(fpath: str):
    # Construct datasets.
    # The argument "loader" tells how torchvision reads the data.
    deterministic_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    diversified_transform = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize(img_size),
        # You may add some transforms here.
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.GaussianBlur(13),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomPosterize(bits=2),
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0),
        GaussianNoise(mean=0, sigma=(0.01, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FoodDataset(f"{dataset_dir}/test", split="test", transform=deterministic_transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    # Setup model.
    model = models.resnet50(num_classes=11, weights=None, progress=None)
    model.load_state_dict(torch.load(fpath))
    model.to(device)
    model.eval()

    # Utility
    sigmoid = nn.Sigmoid()

    # Deterministic part
    deterministic_prob = np.empty((len(dataset), 11))
    for i, (x, _) in enumerate(tqdm(dataloader, desc='Inferring (deterministic)')):
        deterministic_prob[i] = sigmoid(model(x.to(device))).cpu().data.numpy()

    # Probabilistic part
    dataset.transform = diversified_transform
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    probabilistic_prob = np.empty((hparams['test-time-augmentation']['candidates'], len(dataset), 11))
    for c in range(hparams['test-time-augmentation']['candidates']):
        for i, (x, _) in enumerate(tqdm(dataloader, desc='Inferring (augmentation)')):
            probabilistic_prob[c][i] = sigmoid(model(x.to(device))).cpu().data.numpy()
    probabilistic_prob = np.mean(probabilistic_prob, axis=0)

    # Weight averaged prob to and select label
    w = hparams['test-time-augmentation']['weight']
    prob = w * probabilistic_prob + (1 - w) * deterministic_prob
    label = np.argmax(prob, axis=1).squeeze().tolist()

    # Study how TTA affects accuracy
    # diff = np.mean(np.abs(deterministic_prob - probabilistic_prob), axis=0)
    # print(f"{diff=}")

    # create test csv
    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)

    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(dataset))]
    df["Category"] = label
    df.to_csv("submission.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load-ckpt', type=str, help="load target checkpoint.")

    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test(args.load_ckpt)
