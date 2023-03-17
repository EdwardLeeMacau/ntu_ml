"""
Visualize the learned visual representations of the CNN model on the validation set
by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output of
both top & mid layers (You need to submit 2 images).
"""

import argparse

import matplotlib.cm as cm
import numpy as np
import torch
import yaml
from dataset import FoodDataset
from matplotlib import pyplot as plt
from model import Classifier
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

_exp_name = "sample"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def visualize():
    with open('params.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Load the trained model
    state_dict = torch.load(f"{_exp_name}_best.ckpt")

    model = Classifier().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(model)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load the validation set
    valid_loader = DataLoader(
        FoodDataset("data/valid", transform=transform), batch_size=64, num_workers=0, pin_memory=True
    )

    # Extract the representations for the specific layer of model
    index = ... # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
    features = []
    labels = []
    for batch in tqdm(valid_loader):
        X, Y = batch

        logits = model.cnn[:index](X.to(device))
        logits = logits.view(logits.size()[0], -1)
        labels.extend(Y.cpu().numpy())
        logits = np.squeeze(logits.cpu().numpy())

        features.extend(logits)

    features = np.array(features)
    colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

    # Apply t-SNE to the features
    features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        x = features_tsne[labels == label, 0]
        y = features_tsne[labels == label, 1]

        plt.scatter(x, y, label=label, s=5)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize()