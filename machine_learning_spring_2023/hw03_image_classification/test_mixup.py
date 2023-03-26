import torch
import os
from utils import mixup, same_seeds
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

def main():
    same_seeds(3407)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    inverse = transforms.ToPILImage()

    x = torch.empty((2, 3, 224, 224), dtype=torch.float)
    y = torch.zeros((2, 11), dtype=torch.float)

    # dataloader
    dirname = '/tmp2/edwardlee/food-11/train'
    for i, fname in enumerate(('0_0.jpg', '1_12.jpg')):
        fname = os.path.join(dirname, fname)

        x[i] = transform(Image.open(fname))
        y[i] = F.one_hot(
            torch.tensor(int(fname.split("/")[-1].split("_")[0]), dtype=torch.long), num_classes=11
        )

    x, y = mixup((x, y), alpha=0.4)
    print(f'{y.shape=}')
    print(f'{y=}')

    for i in range(x.shape[0]):
        inverse(x[i]).save(f'mixup-{i}.jpg')

    return

if __name__ == "__main__":
    main()