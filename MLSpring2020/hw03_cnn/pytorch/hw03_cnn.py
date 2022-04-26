import os
from random import shuffle
import time
from tkinter import image_names

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, path: str, label: bool, transform=None):
        self.transform = transform
        self.images = sorted(os.listdir(path))
        self.label = label
        self.path = path
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Return image in np.array and its label
        """
        file = self.images[idx]

        x = cv2.resize(cv2.imread(os.path.join(self.path, file)), (128, 128))
        if self.transform is not None:
            x = self.transform(x)

        if self.label == False:
            return x

        y = int(file.split('_')[0])

        return (x, y)

class Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Layer, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )

    def forward(self, x):
        return self.f(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Self-defined model architecture
        filter_nums = (3, 64, 128, 256, 512, 512)
        filter_nums = [(filter_nums[idx], filter_nums[idx + 1]) for idx in range(len(filter_nums) - 1)]

        self.cnn = nn.Sequential(
            *[Layer(in_features, out_features) for (in_features, out_features) in filter_nums]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.cnn(x).view(batch_size, -1)
        x = self.fc(x)
        return x

def load_model(fname: str) -> Classifier:
    model = Classifier()
    model.load_state_dict(torch.load(fname))
    model.eval()
    return model

def save_model(model: Classifier, fname: str):
    torch.save(model.state_dict(), fname)
    return

def inference():
    batch_size = 128
    model = load_model('./hw03_model/classifier.pt')

    test_set = ImageDataset('../hw03_data/validation', label=True,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    ))

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for idx, x in enumerate(test_loader):
            x = x.cuda()
            pred = torch.argmax(model(x), )


    return

def train():
    # Hyper-parameters
    num_epoches = 30
    batch_size = 128

    # Create dataset and specify data augmentation method
    # Then, create DataLoader
    train_set = ImageDataset('../hw03_data/training', label=True,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    ))

    val_set = ImageDataset('../hw03_data/validation', label=True,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    ))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Create model
    model = Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.cuda()
    for epoch in range(num_epoches):
        # Loop initialization
        train_acc, train_loss = 0.0, 0.0
        val_acc, val_loss = 0.0, 0.0

        # Training
        model.train()
        for idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            optimizer.step()

            train_acc += torch.sum(torch.argmax(pred, axis=1) == y)
            train_loss += loss.item()

        print(f"[{(epoch + 1):03}/{num_epoches:03}] Train Acc:{(train_acc / len(train_set)):.2%} Loss:{(train_loss / len(train_set)):.6}")

        # Validation
        model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(val_loader):
                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                loss = criterion(pred, y)

                val_acc += torch.sum(torch.argmax(pred, axis=1) == y)
                val_loss += loss.item()

        print(f"[{(epoch + 1):03}/{num_epoches:03}] Validation Acc:{(val_acc / len(val_set)):.2%} Loss:{(val_loss / len(val_set)):.6}")

    # Save model
    model.cpu()
    save_model(model, './hw03_model/classifier.pt')

    return

def main():
    train()
    return

if __name__ == "__main__":
    main()