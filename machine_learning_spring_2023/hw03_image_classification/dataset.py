import os
from typing import Callable, Dict, Optional
from collections import Counter

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


class FoodDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        super(FoodDataset).__init__()

        files = sorted([os.path.join(root, x) for x in os.listdir(root) if x.endswith(".jpg")])
        # Test set has no label, a dummy variable only.
        if split in ("train", "val"):
            label = [int(os.path.basename(f).split('_')[0]) for f in files]
        else:
            label = [0 for _ in range(len(files))]

        self.split = split
        self.instances = list(zip(files, label))
        self.transform = transform
        self.num_classes = 11

    def statistics(self) -> Dict:
        if self.split not in ("train", "val"):
            raise ValueError

        counter = Counter()
        for _, c in self.instances:
            counter[c] += 1

        return counter

    def sampler(self) -> Sampler:
        average_instances = len(self.instances) / 11

        weight = {k: average_instances / v for k, v in self.statistics().items()}
        sample_weight = [weight.get(l, 1) for _, l in self.instances]
        return WeightedRandomSampler(sample_weight, num_samples=len(self.instances))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        fname, label = self.instances[idx]

        im = Image.open(fname)
        if self.transform is not None:
            im = self.transform(im)

        # Onehot as float to allow label smoothing, mixup operations....
        label = torch.tensor(label, dtype=torch.long)
        onehot = F.one_hot(label, self.num_classes).type(torch.float)

        return im, onehot


def test_trainval_dataset():
    import yaml
    import random

    """# Configurations"""
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    dataset_dir = hparams['env']['dataset']
    train_set = FoodDataset(f"{dataset_dir}/train", split="train", transform=None)
    valid_set = FoodDataset(f"{dataset_dir}/valid", split="val", transform=None)

    instances = []

    # Copy instances and shuffle again
    instances.extend(train_set.instances)
    instances.extend(valid_set.instances)
    random.shuffle(instances)

    # Reassign instance list
    num = len(train_set)
    train_set.instances, valid_set.instances = instances[:num], instances[num:]

if __name__ == "__main__":
    test_trainval_dataset()