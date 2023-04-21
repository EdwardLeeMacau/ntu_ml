import torch
import torch.nn as nn
import argparse
import os
import shutil
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from typing import Callable, Union, List
from dataset import AdvDataset
from pytorchcv.model_provider import get_model
from model import fgsm, ifgsm, mifgsm, EnsembleNet
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

# the mean and std are the calculated statistics from CIFAR10 dataset
CIFAR10_MEAN = (0.491, 0.482, 0.447)
CIFAR10_STD = (0.202, 0.199, 0.201)

mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(device)
std = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(device)

# Constraint
epsilon = 8 / 255 / std

# Attack agnostic parameters
ATTACKERS = { 'fgsm': fgsm, 'ifgsm': ifgsm, 'mifgsm': mifgsm }
batch_size = 8

# Hyperparameters for attacker
alpha = 0.8 / 255 / std
method = 'ifgsm'

# Check supported pre-trained models as proxy network
#
# Reference:
# https://pypi.org/project/pytorchcv/
model_names = [
    'nin_cifar10',
    'resnet20_cifar10',
    'preresnet20_cifar10'
]

eval_model_names = model_names.copy()
eval_model_names.extend([
    'resnet110_cifar10'
])

root = './data' # directory for storing benign images

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

@torch.no_grad()
def inference(model: nn.Module, loader: DataLoader, loss_fn):
    acc, _loss = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        acc += (pred.argmax(dim=1) == y).sum().item()
        _loss += loss.item() * x.shape[0]

    return acc / len(loader.dataset), _loss / len(loader.dataset)

def generate_adversarial_instances(model, loader, attack: Callable, loss_fn):
    """
    perform untargeted adversarial attack and generate adversarial examples
    """

    adv_names = []
    acc, loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10)

        pred = model(x_adv)
        loss = loss_fn(pred, y)

        acc += (pred.argmax(dim=1) == y).sum().item()
        loss += loss.item() * x.shape[0]

        # store adversarial examples
        adv_ex = ((x_adv) * std + mean).clamp(0, 1) # to 0-1 scale
        adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
        adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
        adv_ex = adv_ex.transpose((0, 2, 3, 1)) # transpose (bs, C, H, W) back to (bs, H, W, C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]

    return adv_examples, acc / len(loader.dataset), loss / len(loader.dataset)

# create directory which stores adversarial examples
def create_dir(data_dir, adv_dir, x_adv, fnames: List[str]):
    if not os.path.exists(adv_dir):
        _ = shutil.copytree(data_dir, adv_dir)

    for example, name in zip(x_adv, fnames):
        im = Image.fromarray(example.astype(np.uint8)) # image pixel value should be unsigned int
        im.save(os.path.join(adv_dir, name))

def create_model_instance(name: Union[List[str], str], ensemble=False) -> nn.Module:
    if ensemble and isinstance(name, list) and len(name) == 1:
        # Auto unpack variable
        ensemble, name = False, name[0]

    if not ensemble and isinstance(name, list) and len(name) > 1:
        raise ValueError

    model = EnsembleNet(name) if ensemble else get_model(name, pretrained=True)

    model.to(device)
    model.eval()

    return model

def visualize(model: nn.Module):
    plt.figure(figsize=(10, 20))
    cnt = 0

    for i, cls_name in enumerate(classes):
        path = f'{cls_name}/{cls_name}1.png'

        # benign image
        cnt += 1
        plt.subplot(len(classes), 4, cnt)
        im = Image.open(f'./data/{path}')
        logit = model(transform(im).unsqueeze(0).to(device))[0]
        predict = logit.argmax(-1).item()
        prob = logit.softmax(-1)[predict].item()
        plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
        plt.axis('off')
        plt.imshow(np.array(im))

        # adversarial image
        cnt += 1
        plt.subplot(len(classes), 4, cnt)
        im = Image.open(f'./fgsm/{path}')
        logit = model(transform(im).unsqueeze(0).to(device))[0]
        predict = logit.argmax(-1).item()
        prob = logit.softmax(-1)[predict].item()
        plt.title(f'adversarial: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
        plt.axis('off')
        plt.imshow(np.array(im))

    plt.tight_layout()
    plt.show()

    """## Report Question
    * Make sure you follow below setup: the source model is "resnet110_cifar10", applying the vanilla fgsm attack on `dog2.png`. You can find the perturbed image in `fgsm/dog2.png`.
    """

    # original image
    path = f'dog/dog2.png'
    im = Image.open(f'./data/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'benign: dog2.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
    plt.tight_layout()
    plt.show()

    # adversarial image
    adv_im = Image.open(f'./fgsm/{path}')
    logit = model(transform(adv_im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(adv_im))
    plt.tight_layout()
    plt.show()

def passive_defense(model: nn.Module, adv_im: Image):
    """## Passive Defense - JPEG compression
    JPEG compression by imgaug package, compression rate set to 70

    Reference: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html#imgaug.augmenters.arithmetic.JpegCompression
    """

    # pre-process image
    x = transforms.ToTensor()(adv_im) * 255
    x = x.permute(1, 2, 0).numpy()
    x = x.astype(np.uint8)

    # TODO: use "imgaug" package to perform JPEG compression (compression rate = 70)
    x = x

    logit = model(transform(x).unsqueeze(0).to(device))[0]
    pred = logit.argmax(-1).item()
    prob = logit.softmax(-1)[pred].item()

    plt.title(f'JPEG adversarial: dog2.png\n{classes[pred]}: {prob:.2%}')
    plt.axis('off')

    plt.imshow(x)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', action='store_true')
    parser.add_argument('--defense', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    eval_models = { n: create_model_instance(n, ensemble=False) for n in eval_model_names }
    target_model = create_model_instance(model_names, ensemble=True)

    if args.attack:
        # Prepare to be attacked instances
        dataset = AdvDataset(root, transform=transform)
        benign_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Prepare the attacking algorithms
        attacker = ATTACKERS[method]

        # Generate adversarial examples given a model as guidance
        print(f"Guidance: {model_names}")
        x_adv, acc, loss = generate_adversarial_instances(
            target_model, benign_dl, attacker, loss_fn
        )
        create_dir(root, method, x_adv, benign_dl.dataset.__getname__())

        # Evaluate the adversarial examples on unseen models
        dataset = AdvDataset(method, transform=transform)
        adv_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        metrics = {}
        for k, m in eval_models.items():
            benign_acc, benign_loss = inference(m, benign_dl, loss_fn)
            adv_acc, adv_loss = inference(m, adv_dl, loss_fn)

            name = ('*' if k in model_names else '') + k
            metrics[name] = {
                'benign_acc': benign_acc, 'adv_acc': adv_acc,
                'benign_loss': benign_loss, 'adv_loss': adv_loss
            }

        metrics = pd.DataFrame(metrics).T
        print(metrics)

    if args.visualize:
        visualize(target_model)

    if args.defense:
        passive_defense(target_model)

if __name__ == '__main__':
    main()
