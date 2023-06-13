import argparse
import os

import numpy as np
import torch
import yaml
from config import initialize
from dataset import FoodDataset, get_paths_labels
from lime import lime_image
from matplotlib import pyplot as plt
from model import Classifier
from skimage.segmentation import slic

args, model, img_indices, images, labels = initialize()

@torch.no_grad()
def predict(input: np.ndarray) -> np.ndarray:
    # input: numpy array, (batches, height, width, channels)

    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    # pytorch tensor, (batches, channels, height, width)

    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    # split the image into 200 pieces with the help of segmentation from skimage
    return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)


fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
# fix the random seed to make it reproducible
np.random.seed(16)
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    # numpy array for lime

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)

    # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance
    lime_img, mask = explanation.get_image_and_mask(
        label=label.item(),
        positive_only=False,
        hide_rest=False,
        num_features=11,
        min_weight=0.05
    )

    # turn the result from explainer to the image
    # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask

    axs[idx].imshow(lime_img)

os.makedirs('.explain', exist_ok=True)
plt.savefig('.explain/lime.png')
plt.close()
