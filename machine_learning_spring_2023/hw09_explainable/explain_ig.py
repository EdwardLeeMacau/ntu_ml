import os

import numpy as np
from config import initialize
from model import IntegratedGradients, normalize
from matplotlib import pyplot as plt
from skimage.segmentation import slic

args, model, img_indices, images, labels = initialize()
images = images.cuda()

IG = IntegratedGradients(model)
integrated_grads = []
for i, img in enumerate(images):
    img = img.unsqueeze(0)
    integrated_grads.append(IG.generate_integrated_gradients(img, labels[i], 10))

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))

for i, img in enumerate(images):
    axs[0][i].imshow(img.cpu().permute(1, 2, 0))

for i, img in enumerate(integrated_grads):
    axs[1][i].imshow(np.moveaxis(normalize(img),0,-1))

os.makedirs('.explain', exist_ok=True)
plt.savefig('.explain/integrated_gradients.png')
plt.close()
