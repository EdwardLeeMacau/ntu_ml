import torch
import yaml
from collections import Counter, OrderedDict
from dataset import FoodDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


@torch.no_grad()
def test_weighted_sampler():
    with open('params.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    img_size = tuple(hparams['model']['resize'])
    dataset_dir = hparams['env']['dataset']

    transform = transforms.Compose([
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    dataset = FoodDataset(f"{dataset_dir}/train", split='train', transform=transform)
    sampler = dataset.sampler()

    origin_class_distribution = OrderedDict(sorted(dataset.statistics().items()))
    print(f'{origin_class_distribution=}')

    counter = Counter()
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=False, sampler=sampler)
    for _, y in tqdm(dataloader):
        y = y.argmax(dim=-1).squeeze().item()
        counter[y] += 1

    resampled_class_distribution = OrderedDict(sorted(counter.items()))
    print(f"{resampled_class_distribution=}")

if __name__ == "__main__":
    test_weighted_sampler()