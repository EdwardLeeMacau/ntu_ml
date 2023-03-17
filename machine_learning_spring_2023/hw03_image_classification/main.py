# -*- coding: utf-8 -*-
"""ML2023Spring - HW3
# HW3 Image Classification

## We strongly recommend that you run with [Kaggle](https://www.kaggle.com/t/86ca241732c04da99aca6490080bae73) for this homework

If you have any questions, please contact the TAs via TA hours, NTU COOL, or email to mlta-2023-spring@googlegroups.com
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from dataset import FoodDataset
from model import Classifier
import json
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm import tqdm
# This is for the progress bar.
from tqdm.auto import tqdm

seed = 6666  # set a random seed for reproducibility

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

"""# Configurations"""
with open('params.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)

# The number of batch size.
# batch_size = 64
batch_size = params['tuning']['batch-size']
image_size = tuple(params['model']['resize'])

# The number of training epochs.
n_epochs = params['epochs']
weight_decay = params['tuning']['weight-decay']
learning_rate = params['tuning']['learning-rate']

# If no improvement in 'patience' epochs, early stop.
patience = 5

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

same_seeds(seed)
"""# Dataloader"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
# a possible solution: AutoAugment(AutoAugmentPolicy.IMAGENET)
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize(image_size),
    # You may add some transforms here.
    transforms.GaussianBlur(5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.8, p=0.5),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.RandomPosterize(bits=2),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

def train():
    # Construct train and valid datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set = FoodDataset("./data/train", transform=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_set = FoodDataset("./data/valid", transform=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    """# Start Training"""

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"model.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvement {patience} consecutive epochs, early stopping")
                break

    with open('metrics.json', 'w') as f:
        json.dump({ 'accuracy': best_acc.item() }, f)

def test():
    # Construct test datasets.
    # The argument "loader" tells how torchvision reads the data.
    test_set = FoodDataset("data/test", transform=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"model.ckpt"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data,_ in tqdm(test_loader):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    # create test csv
    def pad4(i):
        return "0"*(4-len(str(i)))+str(i)
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set))]
    df["Category"] = prediction
    df.to_csv("submission.csv",index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()
