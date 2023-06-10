import os
from collections import Counter
from typing import Callable, Optional, Dict

from PIL import Image
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

class FoodDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = transform):
        super(FoodDataset, self).__init__()

        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split name {split}")

        files = sorted([os.path.join(root, x) for x in os.listdir(root) if x.endswith(".jpg")])
        # Test set has no label, a dummy variable only.
        if split in ('train', 'val'):
            labels = [int(os.path.basename(f).split('_')[0]) for f in files]
        else:
            labels = [0 for _ in range(len(files))]

        self.split = split
        self.instances = list(zip(files, labels))
        self.transform = transform
        self.num_classes = 11

    def statistics(self) -> Dict:
        counter = Counter()
        for _, c in self.instances:
            counter[c] += 1

        return counter

    def sampler(self) -> Sampler:
        average_instances = len(self.instances) / self.num_classes

        weight = {k: average_instances / v for k, v in self.statistics().items()}
        sample_weight = [weight.get(l, 1) for _, l in self.instances]
        return WeightedRandomSampler(sample_weight, num_samples=len(self.instances))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self,idx):
        fname, label = self.instances[idx]

        im = Image.open(fname)
        im = self.transform(im)

        # TODO: support one-hot label to allow label smoothing, mixup operations....

        return im, label