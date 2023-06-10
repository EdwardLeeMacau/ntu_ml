"""### Architecture_Design

In this homework, you have to design a smaller network and make it perform well. Apparently, a well-designed architecture is crucial for such task. Here, we introduce the depthwise and pointwise convolution. These variants of convolution are some common techniques for architecture design when it comes to network compression.

<img src="https://i.imgur.com/LFDKHOp.png" width=400px>

* explanation of depthwise and pointwise convolutions:
    * [prof. Hung-yi Lee's slides(p.24~p.30, especially p.28)](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/tiny_v7.pdf)

* other useful techniques
    * [group convolution](https://www.researchgate.net/figure/The-transformations-within-a-layer-in-DenseNets-left-and-CondenseNets-at-training-time_fig2_321325862) (Actually, depthwise convolution is a specific type of group convolution)
    * [SqueezeNet](!https://arxiv.org/abs/1602.07360)
    * [MobileNet](!https://arxiv.org/abs/1704.04861)
    * [ShuffleNet](!https://arxiv.org/abs/1707.01083)
    * [Xception](!https://arxiv.org/abs/1610.02357)
    * [GhostNet](!https://arxiv.org/abs/1911.11907)

After introducing depthwise and pointwise convolutions, let's define the **student network architecture**. Here, we have a very simple network formed by some regular convolution layers and pooling layers. You can replace the regular convolution layers with the depthwise and pointwise convolutions. In this way, you can further increase the depth or the width of your network architecture.
"""

"""### Knowledge_Distillation

<img src="https://i.imgur.com/H2aF7Rv.png=100x" width="400px">

Since we have a learned big model, let it teach the other small model. In implementation, let the training target be the prediction of big model instead of the ground truth.

**Why it works?**
* If the data is not clean, then the prediction of big model could ignore the noise of the data with wrong labeled.
* There might have some relations between classes, so soft labels from teacher model might be useful. For example, Number 8 is more similar to 6, 9, 0 than 1, 7.


**How to implement?**
* $Loss = \alpha T^2 \times KL(p || q) + (1-\alpha)(\text{Original Cross Entropy Loss}), \text{where } p=softmax(\frac{\text{student's logits}}{T}), \text{and } q=softmax(\frac{\text{teacher's logits}}{T})$
* very useful link: [pytorch docs of KLDivLoss with examples](!https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
* original paper: [Distilling the Knowledge in a Neural Network](!https://arxiv.org/abs/1503.02531)

**Please be sure to carefully check each function's parameter requirements.**
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from dataset import FoodDataset
from model import GaussianNoise, KnowledgeDistillationLoss, get_student_model
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm import trange
from tqdm.auto import tqdm
from utils import cycle, same_seeds

with open('params.yaml', 'r') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

cfg = {
    'save_dir': './outputs',
    'exp_name': "simple_baseline",
    'grad_norm_max': 10,
}

# Fix random seed for reproducibility
seed = hparams['seed']
same_seeds(seed)

# Env hyperparameters
root_dir = hparams['env']['dataset']

# Training hyperparameters
batch_size = hparams['batch-size']

# Loss function hyperparameters
temperature = hparams['knowledge-distillation']['temperature']
alpha = hparams['knowledge-distillation']['alpha']

save_path = os.path.join(cfg['save_dir'], cfg['exp_name']) # create saving directory
os.makedirs(save_path, exist_ok=True)

# define training/testing transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_tfm = transforms.Compose([
    # add some useful transform or augmentation here, according to your experience in HW3.
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(13),
    # The training input size of the provided teacher model is (3, 224, 224).
    # Thus, Input size other then 224 might hurt the performance. please be careful.
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0),
    GaussianNoise(mean=0, sigma=(0.01, 0.1)),
    normalize,
])

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    # Hyperparameters
    grad_norm_max = cfg['grad_norm_max']

    # Initialize tracker
    writer = SummaryWriter()

    # Form train/valid dataloaders
    train_set = FoodDataset(os.path.join(root_dir, "training"), split='train', transform=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    valid_set = FoodDataset(os.path.join(root_dir, "validation"), split='val', transform=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # DO NOT modify this block and please make sure that this block can run successfully.
    student_model = get_student_model()
    summary(student_model, (3, 224, 224), device='cpu')

    # Load provided teacher model (model architecture: resnet18, num_classes=11, test-acc ~= 89.9%)
    teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None, num_classes=11)
    teacher_ckpt_path = os.path.join(root_dir, "resnet18_teacher.ckpt")
    teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location='cpu'))

    criteria = KnowledgeDistillationLoss(alpha=alpha, temperature=temperature)

    # The number of training epochs and patience.
    n_epochs = hparams['epochs']
    # patience = cfg['patience'] # If no improvement in 'patience' epochs, early stop

    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    # Initialize optimizer.
    optimizer_param = hparams['optimizer']['kwargs']
    optimizer = torch.optim.Adam(student_model.parameters(), **optimizer_param)

    # TODO: Initialize scheduler.

    # Initialize trackers
    num_iters = n_epochs * len(train_loader)
    num_validation = len(train_loader)

    pbar = trange(num_iters, desc='Training')
    train_loader = cycle(train_loader)
    for i in pbar:
        student_model.train()

        imgs, labels = next(train_loader)

        # A batch consists of image data and corresponding labels.
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward the data. (Make sure data and model are on the same device.)
        # TODO: Apply test time augmentation to make predictions more robust.
        with torch.no_grad():
            teacher_logits = teacher_model(imgs)

        logits = student_model(imgs)
        loss = criteria(logits, labels, teacher_logits)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_norm_max)
        optimizer.step()

        # Record the loss and accuracy.
        writer.add_scalars('Loss', { 'train': loss.item() }, i)
        writer.add_scalars('Grad Norm', { 'train': grad_norm }, i)
        # writer.add_scalars('Accuracy', { 'train': acc.item() }, i)

        # Validation section
        if i % num_validation == 0:
            student_model.eval()

            with torch.no_grad():
                valid_loss = []
                valid_accs = []
                valid_lens = []

                for imgs, labels in tqdm(valid_loader, desc='Validating', leave=False):
                    imgs = imgs.to(device)
                    labels = labels.to(device)


                    logits = student_model(imgs)
                    teacher_logits = teacher_model(imgs)

                    loss = criteria(logits, labels, teacher_logits)
                    acc = (logits.argmax(dim=-1) == labels).float().sum()

                    # Record the loss and accuracy.
                    batch_len = len(imgs)
                    valid_loss.append(loss.item() * batch_len)
                    valid_accs.append(acc)
                    valid_lens.append(batch_len)

                # The average loss and accuracy for entire validation set is the average of the recorded values.
                valid_loss = sum(valid_loss) / sum(valid_lens)
                valid_acc = sum(valid_accs) / sum(valid_lens)

            pbar.set_postfix(loss=valid_loss, acc=valid_acc.item())

            writer.add_scalars('Loss', {'validation': valid_loss }, i)
            writer.add_scalars('Accuracy', {'validation': valid_acc.item() }, i)

            student_model.train()

    # Save the trained model.
    student_model.cpu()
    state_dict = student_model.state_dict()
    torch.save(state_dict, os.path.join(save_path, "model.ckpt"))

@torch.no_grad()
def inference():
    def pad4(i):
        return "0"*(4-len(str(i))) + str(i)

    dataset = FoodDataset(os.path.join(root_dir, "evaluation"), split='test', transform=test_tfm)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    # Load model from {exp_name}/student_best.ckpt
    student_model_best = get_student_model() # get a new student model to avoid reference before assignment.
    ckpt_path = f"{save_path}/model.ckpt" # the ckpt path of the best student model.
    student_model_best.load_state_dict(torch.load(ckpt_path, map_location='cpu')) # load the state dict and set it to the student model
    student_model_best.to(device) # set the student model to device
    student_model_best.eval()

    # storing predictions of the evaluation dataset
    prediction = []
    for imgs, _ in tqdm(dataloader, desc="Inferring"):
        logits = student_model_best(imgs.to(device))
        pred = logits.argmax(dim=-1).squeeze().cpu().numpy().tolist()

        prediction += pred

    ids = [pad4(i) for i in range(len(dataset))]
    categories = prediction

    df = pd.DataFrame()
    df['Id'] = ids
    df['Category'] = categories
    df.to_csv(f"{save_path}/submission.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--load-ckpt', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        train()
        inference()

    if args.inference:
        inference()

if __name__ == "__main__":
    main()
