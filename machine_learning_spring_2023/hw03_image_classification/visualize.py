"""
Visualize the learned visual representations of the CNN model on the validation set
by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output of
both top & mid layers (You need to submit 2 images).
"""

import argparse
import io
import random
from collections import defaultdict

import matplotlib.cm as cm
import numpy as np
import torch
import yaml
from dataset import FoodDataset
from matplotlib import pyplot as plt
from model import GaussianNoise
from PIL import Image
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from utils import mixup, same_seeds, sizeof_fmt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def visualize_transform():
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    img_size = tuple(hparams['model']['resize'])
    dataset_dir = hparams['env']['dataset']

    transform = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize(img_size),
        # You may add some transforms here.
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        # transforms.RandomEqualize(),
        # transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(13),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # transforms.RandomPosterize(bits=2),
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0),
        GaussianNoise(mean=0, sigma=(0.01, 0.1)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = FoodDataset(f"{dataset_dir}/train", split='train', transform=transform)

    writer = SummaryWriter(f'runs/transform')

    idx = random.randint(0, len(dataset) - 1)
    fname = dataset.files[idx]
    print(f'{fname=}')

    batch = torch.empty((25, 3, *img_size))
    for i in range(25):
        batch[i] = dataset[idx][0]
    batch = make_grid(batch, nrow=5)

    writer.add_image('test', batch, 0)
    writer.close()

@torch.no_grad()
def visualize_dataset():
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    img_size = tuple(hparams['model']['resize'])
    dataset_dir = hparams['env']['dataset']

    transform = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    dataset = FoodDataset(f"{dataset_dir}/train", split='train', transform=transform)

    writer = SummaryWriter(f'runs/dataset')
    files = defaultdict(list)
    for fname, label in dataset.instances:
        files[label].append(fname)

    print(dataset.statistics())
    for k, v in files.items():
        print(f'{k:02d}: {len(v)}')

    batch = torch.zeros((5 * 11, 3, *img_size))
    for i in range(11):
        for j in range(5):
            idx = i * 5 + j
            batch[idx] = transform(Image.open(files[i][j]))
    batch = make_grid(batch, nrow=5)

    writer.add_image('dataset/train', batch, 0)
    writer.close()

@torch.no_grad()
def visualize_representation(fpath: str) -> io.BytesIO:
    """
    Parameters
    ----------
    fpath: str
        Path to model checkpoint
    """
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    image_size = hparams['model']['input-size']['test']
    image_size = (image_size['height'], image_size['width'])
    dataset_dir = hparams['env']['dataset']

    # Load the trained model or pass by argument
    state_dict = torch.load(fpath)
    model = models.resnext101_32x8d(
        num_classes=11, weights=None, progress=None
    ).to(device)
    model.load_state_dict(state_dict)

    # Print entire model structure
    print(model)

    # Remove classification head
    model.fc = nn.Identity()
    model.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the validation set
    valid_loader = DataLoader(
        FoodDataset(f"{dataset_dir}/valid", split='val', transform=transform),
        batch_size=512, num_workers=8, pin_memory=True
    )

    # Extract the representations for the specific layer of model
    # index = ... # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
    features = []
    labels = []
    with tqdm(valid_loader, desc='Validating') as pbar:
        for batch in pbar:
            X, Y = batch

            logits = model(X.to(device))
            logits = logits.view(logits.size()[0], -1)
            labels.extend(Y.cpu().numpy())
            logits = np.squeeze(logits.cpu().numpy())

            features.extend(logits)
            pbar.set_postfix(MemUsage=sizeof_fmt(torch.cuda.max_memory_allocated()))

    features = np.array(features)
    colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

    # Apply t-SNE to the features
    features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        x = features_tsne[labels == label, 0]
        y = features_tsne[labels == label, 1]

        plt.scatter(x, y, label=label, s=5, color=colors_per_class[label])

    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--repr', action='store_true')
    parser.add_argument('--dataset', action='store_true')
    parser.add_argument('--transform', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--load-ckpt', type=str, help="load target checkpoint.")

    args = parser.parse_args()

    same_seeds(args.seed)

    if args.transform:
        visualize_transform()

    if args.dataset:
        visualize_dataset()

    if args.repr:
        Image.open(visualize_representation(args.load_ckpt)).save('tsne.png')
