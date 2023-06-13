import os

import numpy as np
import torch
import yaml
from config import initialize
from model import compute_saliency_maps
from matplotlib import pyplot as plt
from skimage.segmentation import slic

args, model, img_indices, images, labels = initialize()

# images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

# visualize
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([images, saliencies]):
    for column, img in enumerate(target):
        if row == 0:
            axs[row][column].imshow(img.permute(1, 2, 0).numpy())
        else:
            axs[row][column].imshow(img.numpy(), cmap=plt.cm.hot)

os.makedirs('.explain', exist_ok=True)
plt.savefig('.explain/saliency.png')
plt.close()
