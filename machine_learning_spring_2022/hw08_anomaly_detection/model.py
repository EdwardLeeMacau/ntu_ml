import numbers
from collections.abc import Sequence
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.autograd import Variable
from torchvision.models import resnet18, resnet34, resnet50


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

class FCNAutoEncoder(nn.Module):
    """ Fully-connected neural network autoencoder"""
    def __init__(self):
        super(FCNAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 64 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 32 * 64 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        # Store the original size format
        size = x.size()

        # Flatten input if its not flattened yet.
        n = x.shape[0]
        x = x.view(n, -1)

        x = self.encoder(x)
        x = self.decoder(x)

        # Restore to original size format
        x = x.view(size)
        return x

def strip_resnet_fc(model: nn.Module) -> nn.Module:
    """ Strip fully-connected layer of resnet """
    model.fc = nn.Identity()

    return model

# Learn architecture from:
#
# Reference:
# - StyleGAN and StyleGAN2, heuristic for designing blocks
#   https://zhuanlan.zhihu.com/p/263554045
#
# - Multi-scale autoencoder
#   https://arxiv.org/pdf/1904.03851.pdf
class ConvAutoEncoder(nn.Module):
    """ Convolutional neural network autoencoder"""
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 64, 2, 4)
        x = self.decoder(x)

        return x

class MultiEncoderAutoEncoder(nn.Module):
    """ Convolutional neural network autoencoder, with multiple encoders """

    _backbone = {
        'resnet18': (resnet18, 512, strip_resnet_fc),
        'resnet34': (resnet34, 512, strip_resnet_fc),
        'resnet50': (resnet50, 2048, strip_resnet_fc),
    }

    def __init__(self) -> None:
        super(MultiEncoderAutoEncoder, self).__init__()

        # Hyperparameters
        encoders_name = ['resnet18', 'resnet34', 'resnet50']

        # Model construction
        self.encoders, self.embedded_dim = self._construct_decoder(encoders_name)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=48, out_channels=6),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(kernel_size=1, in_channels=6, out_channels=48),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # nn.ConvTranspose2d(24, 24, 3, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # nn.ConvTranspose2d(12, 12, 3, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(3, 3, 3, padding=1),
            nn.Tanh(),
        )

    def _construct_decoder(self, names: List[str]) -> nn.Module:
        encoders, embedded_dim = [], []

        for name in names:
            model, dim, strip_fc = self._backbone[name]

            encoders.append(strip_fc(model(weights=None)))
            embedded_dim.append(dim)

        return nn.ModuleList(encoders), embedded_dim

    def forward(self, x):
        device = x.device
        b = x.shape[0]

        start, end = 0, 0
        z = torch.empty(size=(b, sum(self.embedded_dim)), device=device)
        for encoder, dim in zip(self.encoders, self.embedded_dim):
            start, end = end, end + dim
            z[:, start:end] = encoder(x)

        z = z.view(-1, 48, 8, 8)
        x = self.bottleneck(z)
        x = self.decoder(z)

        return x

class VAELoss(nn.Module):
    """ Variational autoencoder loss function """
    def __init__(self, criterion: nn.Module = nn.MSELoss(), alpha: float = 1.0):
        """
        criterion : nn.Module()
            loss metrics for reconstruction loss

        alpha : float
            weight of distribution similarity
        """
        super(VAELoss, self).__init__()

        self.alpha = alpha
        self.criterion = criterion

    def forward(self, out, target):
        """
        input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Output from VAE

        target : torch.Tensor
            target image to be reconstructed
        """
        reconstruct, mu, log_var = out

        # reconstruction loss
        mse = self.criterion(reconstruct, target)

        # distribution similarity
        if self.alpha == 0:
            KLD = 0
        else:
            KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            KLD = torch.sum(KLD_element).mul_(-0.5)

        return mse + KLD

class VAE(nn.Module):
    """ Variational autoencoder """
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = resnet50(weights=None)
        self.encoder.fc = nn.Identity()
        self.enc_out_1 = nn.Sequential(
            nn.Linear(2048, 3072),
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Linear(2048, 3072),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(h1.size(0), -1)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)

        z = self.reparametrize(mu, log_var)
        z = z.view(z.size(0), 48, 8, 8)

        return self.decode(z), mu, log_var

def create_model(name: str) -> Tuple[nn.Module, nn.Module]:
    """ Create model based on name """
    models = {
        'fcn': (FCNAutoEncoder, nn.MSELoss()),
        'cnn': (ConvAutoEncoder, nn.MSELoss()),
        'vae': (VAE, VAELoss()),
    }

    if name not in models:
        raise ValueError(f"Model {name} not found!")

    constructor, criteria = models[name]
    return (constructor(), criteria)
