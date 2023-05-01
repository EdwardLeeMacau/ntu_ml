import numbers
from collections.abc import Sequence

import torch
from pytorchcv.model_provider import get_model
from torch import Tensor, nn
from torchvision import transforms


# GaussianNoise optimization
#
# Reference: https://github.com/pytorch/vision/pull/6233
# A modification still in pull request...
#
# (waiting be implemented to torch)
# TODO: using torch.normal() instead
class GaussianNoise(torch.nn.Module):
    """Adds Gaussian noise to the image with specified mean and standard deviation.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        mean (float or sequence): Mean of the sampling gaussian distribution .
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            sampling the gaussian noise. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    Returns:
        PIL Image or Tensor: Input image perturbed with Gaussian Noise.
    """

    def __init__(self, mean, sigma=(0.1, 0.5)):
        super().__init__()

        if mean < 0:
            raise ValueError("Mean should be a positive number")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.mean = mean
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, image: Tensor) -> Tensor:
        """
        Args:
            image (PIL Image or Tensor): image to be perturbed with gaussian noise.
        Returns:
            PIL Image or Tensor: Image added with gaussian noise.
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        image += torch.randn_like(image) * sigma + self.mean
        return image

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma})"
        return s

class EnsembleNet(nn.Module):
    """
    * ## Example of Ensemble Attack
    * Ensemble multiple models as your proxy model to increase the black-box transferability
      ([paper](https://arxiv.org/abs/1611.02770))
    """
    def __init__(self, model_names):
        super().__init__()

        self.num = len(model_names)
        self.models = nn.ModuleList([
            get_model(name, pretrained=True) for name in model_names
        ])

    def forward(self, x):
        device = x.device
        size = (self.num, x.size(0), 10, )

        logits = torch.empty(size, device=device)
        for i, m in enumerate(self.models):
            logits[i] = m(x)

        return logits.mean(dim=0)

def fgsm(model: nn.Module, x, y, loss_fn, epsilon=8, **kwargs):
    """ Fast gradient sign method, use gradient ascent on x_adv to maximize loss. """
    device = next(model.parameters()).device

    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv to find best perturbation
    epsilon = epsilon.to(device)

    loss = loss_fn(model(x_adv), y)
    loss.backward()

    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()

    return x_adv

def ifgsm(model, x, y, loss_fn, epsilon=8, alpha=0.8, num_iter=20, **kwargs):
    """
    Iterated fast gradient sign method.

    # alpha and num_iter can be decided by yourself
    """
    x_adv = x
    for _ in range(num_iter):
        # need to obtain gradient of x_adv, thus set required grad
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True

        # find gradient of x_adv as oracle
        loss = loss_fn(model(x_adv), y)
        loss.backward()

        # fgsm: use gradient ascent on x_adv to maximize loss
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()

        # clip new x_adv back to [x-epsilon, x+epsilon]
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)

    return x_adv

def mifgsm(model, x, y, loss_fn, epsilon=8, alpha=0.8, num_iter=40, decay=1.0, **kwargs):
    """ Multistep Iterated fast gradient sign method.

    The algorithm can be degenerated to I-FGSM and FGSM using the following hyper-parameters.

             decay = 0           num_iter = 1
    mifgsm ------------> ifgsm ----------------> fgsm
    """

    # Consider adding guassian noise to input image as augmentation.
    #
    # 4.1 ATTACKS AGAINST ADVERSARIALLY TRAINED NETWORKS
    # A new randomized single-step attack.
    # https://arxiv.org/pdf/1705.07204.pdf

    # Diversifying input
    p = kwargs.get('p', 0.5)
    T = transforms.Compose([
        # Guess which defense is used for target model.
        transforms.GaussianBlur(3, sigma=0.5),
        # Augmentation to avoid overfitting
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomResizedCrop(size=32, scale=(0.8, 1.25), antialias=True),
            ]),
        ], p=p)
    ])

    # parameter initialization
    device = next(model.parameters()).device
    x_adv = x

    # initialize momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)

    # write a loop of num_iter to represent the iterative times
    for _ in range(num_iter):
        # need to obtain gradient of x_adv, thus set required grad
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True

        # find gradient of x_adv as oracle
        logits = model(T(x_adv))
        loss = loss_fn(logits, y)
        loss.backward()

        # Algorithm 1 - MI-FGSM
        # Reference: https://arxiv.org/pdf/1710.06081.pdf
        #
        # Why does the graidient takes L1-norm normalization?
        grad = x_adv.grad.detach()
        momentum = decay * momentum + grad / torch.norm(grad, p=1)
        x_adv = x_adv + alpha * momentum.sign()

        # clip new x_adv back to [x-epsilon, x+epsilon]
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon)

    return x_adv

# 4.5. NIPS 2017 Adversarial Competition
# Defense: see https://arxiv.org/pdf/1803.06978.pdf