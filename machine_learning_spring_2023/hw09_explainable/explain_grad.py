import os

import numpy as np
import torch
import yaml
from config import initialize
from model import smooth_grad
from matplotlib import pyplot as plt
from skimage.segmentation import slic

args, model, img_indices, images, labels = initialize()

smooth = []
for i, l in zip(images, labels):
    smooth.append(smooth_grad(i, l, model, 500, 0.4))
smooth = np.stack(smooth)

# visualize
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([images, smooth]):
    for column, img in enumerate(target):
        axs[row][column].imshow(np.transpose(img.reshape(3,128,128), (1,2,0)))

os.makedirs('.explain', exist_ok=True)
plt.savefig('.explain/smooth_grad.png')
plt.close()
