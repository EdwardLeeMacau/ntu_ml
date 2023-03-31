import numbers
from collections.abc import Sequence

import torch
from torch import Tensor
from torch import functional as F
from torch import nn


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

class Regularization(nn.Module):
    """ Workaround for weight decay with skipping layers. """
    def __init__(self, l1norm: float = 0.0, l2norm: float = 0.0):
        super().__init__()
        self.l2norm = l2norm

    def forward(self, model: nn.Module):
        loss = 0

        for name, p in model.named_parameters():
            if 'bn' in name:
                continue
            loss += torch.norm(p)

        return self.l2norm * loss
