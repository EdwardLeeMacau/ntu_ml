import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    def __init__(self, path, transform, files=None):
        # The data is labelled by the name, so we load images and label while calling '__getitem__'
        super(FoodDataset).__init__()

        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        # if files != None:
        #     self.files = files

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im, label