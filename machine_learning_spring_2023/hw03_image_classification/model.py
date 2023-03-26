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


# * Pretrained model are not allowed in this task.
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input.shape [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)

        return self.fc(out)
