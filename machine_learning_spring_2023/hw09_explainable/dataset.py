import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# It might take some time, if it is too long, try to reload it.
# Dataset definition
class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'

        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform

    # pytorch dataset class
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # help to get images for visualizing
    def get_batch(self, indices):
        images = []
        labels = []

        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)

        return torch.stack(images), torch.tensor(labels)

# help to get data path and label
def get_paths_labels(path):
    def my_key(name):
        return int(name.replace(".jpg", "").split("_")[1]) + 1000000 * int(name.split("_")[0])

    img_names = os.listdir(path)
    img_names.sort(key=my_key)
    img_paths = []
    labels = []

    for name in img_names:
        img_paths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))

    return img_paths, labels
