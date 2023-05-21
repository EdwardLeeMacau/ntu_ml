"""# Visualization
We use t-SNE plot to observe the distribution of extracted features.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import os
from model import Model
import torch
from sklearn import manifold
from model import source_transform, target_transform
from torch.utils.data import ImageFolder, DataLoader

"""## Step1: Load checkpoint and evaluate to get extracted features"""

# Hints:
# Set features_extractor to eval mode
# Start evaluation and collect features and labels

# Extract feature extractor from model
model = Model()

state_dict = torch.load('model.pth')['model']
model.load_state_dict(state_dict)
model = model.feature_extractor
model.eval()

root_dir = "/tmp2/edwardlee/dataset/real_or_drawing/"

src = ImageFolder(
    os.path.join(root_dir, 'train_data'), transform=source_transform
)[:]

tgt = ImageFolder(
    os.path.join(root_dir, 'test_data'), transform=target_transform
)[:5000]

img = torch.cat([src, tgt], 0)

"""## Step2: Apply t-SNE and normalize"""

# process extracted features with t-SNE
X = model(img)
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

# Normalization the processed features
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

x, y = X_norm[:, 0], X_norm[:, 1]

"""## Step3: Visualization with matplotlib"""

# Data Visualization
# Use matplotlib to plot the distribution
# The shape of X_norm is (N,2)

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c='red', s=1, label='Source')
plt.savefig('tsne.png')