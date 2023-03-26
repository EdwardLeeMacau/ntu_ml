import os

import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    def __init__(self, path, transform):
        # The data is labelled by the name, so we load images and label while calling '__getitem__'
        super(FoodDataset).__init__()

        self.num_classes = 11

        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        fname = os.path.basename(fname)
        if (underscore := fname.find('_')) == -1:
            # Test set has no label, a dummy variable only.
            onehot = 0
        else:
            label = torch.tensor(int(fname[:underscore]), dtype=torch.long)
            onehot = F.one_hot(label, self.num_classes).type(torch.float)

        return im, onehot
