from typing import Dict

import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torchvision import transforms

# To adapt the data from source domain to target domain, we decide to
# apply these transformation as our data preprocessing strategy.

source_transform = transforms.Compose([
    # Turn RGB to grayscale. (Because Canny do not support RGB images.)
    transforms.Grayscale(),
    # cv2 do not support skimage.Image, so we transform it to np.array,
    # and then adopt cv2.Canny algorithm.
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # Transform np.array back to the skimage.Image.
    transforms.ToPILImage(),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    # Turn RGB to grayscale.
    transforms.Grayscale(),
    # Resize: size of source data is 32x32, thus we need to
    #  enlarge the size of target data from 28x28 to 32x32。
    transforms.Resize((32, 32)),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])


# Reference:
# Deep Learning for Computer Vision, Spring 2019, Homework3 - Domain Adaptation
# https://github.com/EdwardLeeMacau/ntu_dlcv_spring_2019_hw3/blob/master/TransferLearning/dann.py
class GradientReverseF(Function):
    """
    This function supports model training with partially gradient ascent and partially
    gradient descent.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None

def grad_reverse(x, alpha):
    return GradientReverseF.apply(x, alpha)

# Reference:
#
# Zhi-Hu: Explain VADA (Virtual Adversarial Domain Adaptation) and DIRT-T
# https://zhuanlan.zhihu.com/p/60420771
#
# DIRT-T Pytorch Implementation
# https://github.com/Solacex/pytorch-implementation-of-DIRT-T/blob/master/DIRT-T%20pytorch/models/vat_loss.py
class ConditionalEntropy(nn.Module):
    """
    Estimates the conditional entropy of the target distribution.

    This loss function is frequented applied in semi-supervised learning, which restricts the model
    to predict a confident class label for each input sample.
    """
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        p     = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)

        H = - (p * log_p).sum(dim=1).mean(dim=0)
        return H

class KLDivWithLogitsLoss(nn.Module):
    """ Measure KL divergence to discriminator's output logits."""
    def __init__(self):
        super(KLDivWithLogitsLoss, self).__init__()

        # See Doc: Warning
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Arguments
        ---------
        input, target : torch.Tensor
            Logits in shape (N, C)
        """
        input  = F.log_softmax(input, dim=1)        # Q(x)
        target = F.softmax(target, dim=1)           # P(x)

        return self.kl_div(input, target)           # D_KL(P(x) || Q(x))

def normalize_perturbation(p: torch.Tensor) -> torch.Tensor:
    """ Normalize tensor p to have max perturbation 1. """
    n = p.size(0)

    eps = p.new_tensor(1e-12)

    _p2 = p.view(n, -1) ** 2
    _p2 = torch.max(_p2.sum(dim=-1), eps)
    _p2 = torch.sqrt(_p2)
    _p2 = _p2.view(n, 1, 1, 1)

    out = p / _p2

    return out

class VirtualAdversarialLoss(nn.Module):
    """ Restricted models' local lipschitzness. """
    def __init__(self, model: nn.Module, radius=1) -> None:
        super(VirtualAdversarialLoss, self).__init__()

        self.model = model
        self.radius = radius
        self.nll = KLDivWithLogitsLoss()

    def _perturb(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """ Generate perturbation. """

        # start adversarial attack from random initial point
        delta = torch.randn_like(x)
        delta = 1e-6 * normalize_perturbation(delta)
        delta.requires_grad = True

        # forward and backward to retrieve the adversarial perturbation to x
        prob = self.model(x + delta)

        loss = self.nll(prob, logits.detach())
        loss.backward()

        perturbation = normalize_perturbation(delta.grad)
        return (x + self.radius * perturbation).detach()

    def forward(self, x: torch.Tensor, logits: torch.Tensor):
        """ Minimize the adversarial risk to leads model to be locally lipschitz."""
        x_adv = self._perturb(x, logits)

        prob = self.model(x_adv)
        loss = self.nll(prob, logits)

        return loss

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class Reconstructor(nn.Module):
    """ Convolutional autoencoder """
    def __init__(self):
        super(Reconstructor, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(-1, 512, 1, 1)
        x = self.decoder(x)

        return x

class LabelPredictor(nn.Module):
    """ Classifier. """
    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    # TODO: Apply Gradient Reversal Layer to co-training all modules.
    def forward(self, z, lamda: float = 0.1):
        z = grad_reverse(z, lamda)
        return self.layer(z)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.label_predictor = LabelPredictor()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.label_predictor(x)

        return x

# Reference:
#
# WeightEMA implementation.
# https://github.com/Solacex/pytorch-implementation-of-DIRT-T/blob/561aae1bc3c517989ee421c279497653f0ba3985/models/optimizer.py#L43
#
# Implementation of EMA. From homework 6 diffusion model.
class WeightEMA:
    def __init__(self, params, src_params, alpha: float = 0.998):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for dest, src in zip(self.params, self.src_params):
            dest.data[:] = src.data[:]

    def step(self):
        gamma = 1.0 - self.alpha            # exponential moving average weight

        for dest, src in zip(self.params, self.src_params):
            dest.data.mul_(self.alpha)
            dest.data.add_(src.data * gamma)

    def zero_grad(self):
        pass

if __name__ == "__main__":
    def test_conditional_entropy():
        f = ConditionalEntropy()

        logits = torch.zeros((1, 2))
        loss = f(logits)

        assert(abs(loss - 0.693147) <= 1e-4), loss

        logits = torch.Tensor([[0, 1]])
        loss = f(logits)

        assert(abs(loss - 0.582203) <= 1e-4), loss

    def test_kldiv_with_logits():
        f = KLDivWithLogitsLoss()

        zeros = torch.zeros((1, 2))

        logits = torch.zeros((1, 2))
        loss = f(logits, logits)

        assert(abs(loss) <= 1e-4), loss

        logits = torch.Tensor([[0.2, 0.5]])
        loss = f(logits, zeros)

        assert(abs(loss - 0.011208) <= 1e-4), loss

    def test_virtual_adversarial_loss():
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
        )

        with torch.no_grad():
            model[0].weight = nn.Parameter(torch.Tensor([[1, 0], [0, 1]]) * 100.)

        # Verify model with example near the decision boundary.
        x = torch.tensor([[1, 1]], dtype=torch.float)
        z = model(x)

        # assert(z.item() == torch.Tensor([0.5])), z.item()

        f = VirtualAdversarialLoss(model, radius=1)
        loss = f(x, z)
        print(f'{loss=}')

    test_conditional_entropy()
    test_kldiv_with_logits()
    test_virtual_adversarial_loss()
