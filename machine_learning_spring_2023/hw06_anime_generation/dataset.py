from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

class MyDataset(Dataset):
    def __init__(self, folder: str, image_size):
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(folder).glob(f'**/*.jpg')]

        # In https://arxiv.org/pdf/2302.07944.pdf, they apply randomHorizontalFlip as standard
        # data augmentation.
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            # T.GaussianBlur(kernel_size=3),
            # T.RandomRotation(degrees=5, expand=False),
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

