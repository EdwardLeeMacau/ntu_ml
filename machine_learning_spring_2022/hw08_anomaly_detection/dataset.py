import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

class CustomTensorDataset(TensorDataset):
    """ TensorDataset with support of transforms. """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors

        # b, h, w, c -> b, c, h, w
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)

        # mapping images to [-1.0, 1.0]
        self.normalizer = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),
            transforms.Lambda(lambda x: 2. * x/255. - 1.),
        ])

        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]
        x = self.normalizer(x)

        # apply transform
        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)
