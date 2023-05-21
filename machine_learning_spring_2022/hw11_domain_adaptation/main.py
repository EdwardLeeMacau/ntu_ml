import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pysnooper
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from model import (ConditionalEntropy, DomainClassifier, FeatureExtractor,
                   Model, Reconstructor, VirtualAdversarialLoss,
                   source_transform, target_transform)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm, trange
from utils import create_optimizer, create_scheduler, cycle, same_seeds

# load hyperparameters from yaml
with open("params.yaml", "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

# Determine unique experiment ID
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# fix random seed for reproducibility
same_seeds(hparams["seed"])
root_dir = hparams['env']['dataset']
ckpt_dir = os.path.join(hparams['env']['checkpoint'], timestamp)

def adaptive_lambda(curr: int, total: int, k: float = 10):
    """ adaptive lambda function, return lambda in range [0, 1] given x """
    x = curr / total

    # sigmoid function with shifting and scaling
    lambda_ = (2 / (1 + np.exp(-k*x))) - 1
    return lambda_

class ModelTrainer:
    """ Class wrapped domain transfer algorithm. """
    def __init__(self, config: dict, dataset: dict) -> None:
        super().__init__()

        # hyper-parameters
        self.config = config
        self.epochs = config["iterations"]['train']
        self.batch_size = config["batch-size"]

        # model instance initialization
        self.model = Model()
        self.reconstructor = None
        self.domain_classifier = DomainClassifier()

        # store dataset to self
        self.dataset = dataset

    def pretrain(self) -> FeatureExtractor:
        # check cuda availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # create model for reconstruction
        reconstructor = Reconstructor()
        extractor = self.model.feature_extractor

        # create optimizer and scheduler
        params = list(reconstructor.parameters()) + list(extractor.parameters())
        optimizer = create_optimizer(params, **self.config["optimizer"])
        scheduler = create_scheduler(optimizer, **self.config["scheduler"])

        # move model to device
        reconstructor.to(device)
        extractor.to(device)

        # create writer
        writer = SummaryWriter()

        # create dataloader from dataset, use target dataset as unsupervised dataset here
        dataloader = DataLoader(
            self.dataset["target"], batch_size=self.batch_size, pin_memory=True, num_workers=4,
            shuffle=True, drop_last=True
        )

        # compute number of iterations for each epoch
        num_iter = self.config['iterations']['pretrain']

        # create loss function
        criterion = nn.MSELoss()

        # create training loop
        dataloader = cycle(dataloader)
        pbar = trange(num_iter, ncols=0, desc='Pretraining')
        for i in pbar:
            # prepare data
            (x, _) = next(dataloader)

            # move data to device
            x = x.to(device)

            # forward
            z = extractor(x)
            x_hat = reconstructor(z)

            # compute loss
            loss = criterion(x_hat, x)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # log to tensorboard
            metrics = { "loss": loss.item(), }
            writer.add_scalars("pretrain", metrics, i)

            # log to progress bar
            postfix = { "loss": f'{loss.item():.4f}', }
            pbar.set_postfix(postfix)

        # move model to cpu, and deprecate the reconstructor.
        reconstructor.cpu()
        extractor.cpu()

        # save model
        torch.save(reconstructor.state_dict(), 'reconstructor.pth')
        torch.save(extractor.state_dict(), 'extractor.pth')

        # store reconstructor to self
        self.reconstructor = reconstructor

        return extractor

    def domain_adversarial_training(self) -> Model:
        # add a pretrain section
        # self.model.feature_extractor = self.pretrain()

        # load pretrain model
        # self.model.feature_extractor.load_state_dict(torch.load('extractor.pth'))

        # check cuda availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # create optimizer and scheduler
        params = list(self.model.parameters()) + list(self.domain_classifier.parameters())
        optimizer = create_optimizer(params, **self.config["optimizer"])
        scheduler = create_scheduler(optimizer, **self.config["scheduler"])

        # create writer
        writer = SummaryWriter()

        # move model to device
        self.model.to(device)
        self.domain_classifier.to(device)

        # hyper-parameters
        lamda = self.config["lambda"]

        # create dataloader from dataset
        src_dl = DataLoader(
            self.dataset["source"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        tgt_dl = DataLoader(
            self.dataset["target"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        dataloader = zip(src_dl, tgt_dl)

        # compute number of iterations for each epoch
        num_iter = min((len(x) for x in (src_dl, tgt_dl))) * self.epochs

        # create loss function
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()

        # create training loop
        pbar = trange(num_iter, ncols=0, desc='Training')
        for i in pbar:
            # prepare data
            try:
                (src_x, src_y), (tgt_x, _) = next(dataloader)
            except StopIteration:
                dataloader = zip(src_dl, tgt_dl)
                (src_x, src_y), (tgt_x, _) = next(dataloader)

            # move data to device
            src_x = src_x.to(device)
            src_y = src_y.to(device)
            tgt_x = tgt_x.to(device)

            # domain: 1 => source, 0 => target
            x = torch.cat([src_x, tgt_x], dim=0)
            domain_label = torch.zeros([self.batch_size * 2, 1]).to(device)
            domain_label[:self.batch_size] = 1

            # feature extraction
            z = self.model.feature_extractor(x)

            # domain classification, and image label prediction
            # note that the lamda is applied in forward function of domain classifier.
            domain_logits = self.domain_classifier(z, lamda)
            class_logits = self.model.label_predictor(z[:self.batch_size])

            # compute prediction loss and domain classification loss
            loss_F = class_criterion(class_logits, src_y)
            loss_D = domain_criterion(domain_logits, domain_label)

            # notes the model applied gradient reversal layer.
            # the optimize should tune the discriminator to minimize the loss_D, and tune
            # the feature extractor to maximize the loss_D.
            loss = loss_F + loss_D

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # compute accuracy (in source domain)
            source_acc = torch.sum(torch.argmax(class_logits, dim=1) == src_y).item()
            source_acc /= self.batch_size

            # log to tensorboard
            metrics = { "loss_F": loss_F.item(), "loss_D": loss_D.item(), "acc": source_acc, }
            writer.add_scalars("train", metrics, i)

            # log to progress bar
            postfix = {
                "loss_F": f'{loss_F.item():.4f}',
                "loss_D": f'{loss_D.item():.4f}',
                "acc": f'{source_acc:.2%}',
            }
            pbar.set_postfix(postfix)

            # store checkpoint

        return self.model

    def virtual_adversarial_domain_adaptation(self) -> Model:
        # pretrain on target domain.
        # self.model.feature_extractor = self.pretrain()

        # check cuda availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # create optimizer and scheduler
        params = list(self.model.parameters()) + list(self.domain_classifier.parameters())
        optimizer = create_optimizer(params, **self.config["optimizer"])
        scheduler = create_scheduler(optimizer, **self.config["scheduler"])

        # create writer
        writer = SummaryWriter()

        # move model to device
        self.model.to(device)
        self.domain_classifier.to(device)

        # hyper-parameters
        # lamda_D = self.config["lambda"]
        lamda_C = 0 # self.config["lambda"]
        lamda_V = 0 # self.config["lambda"]

        # create dataloader from dataset
        src_dl = DataLoader(
            self.dataset["source"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        tgt_dl = DataLoader(
            self.dataset["target"], batch_size=self.batch_size, pin_memory=True, num_workers=8,
            shuffle=True, drop_last=True
        )

        dataloader = zip(src_dl, tgt_dl)

        # compute number of iterations for each epoch
        num_iter = min((len(x) for x in (src_dl, tgt_dl))) * self.epochs

        # create loss function
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()
        conditional_criterion = ConditionalEntropy()
        virtual_adversarial_criterion = VirtualAdversarialLoss(
            model=self.model, radius=1
        )

        # create training loop
        pbar = trange(num_iter, ncols=0, desc='Training')
        for i in pbar:
            # prepare data
            try:
                (src_x, src_y), (tgt_x, _) = next(dataloader)
            except StopIteration:
                dataloader = zip(src_dl, tgt_dl)
                (src_x, src_y), (tgt_x, _) = next(dataloader)

            # utility variable
            # n: batch size for source domain
            n = self.batch_size

            # determine weight of domain adversarial loss
            lamda_D = adaptive_lambda(i, num_iter)

            # move data to device
            src_x = src_x.to(device)
            src_y = src_y.to(device)
            tgt_x = tgt_x.to(device)

            # domain: 1 => source, 0 => target
            x = torch.cat([src_x, tgt_x], dim=0)
            domain_label = torch.zeros([self.batch_size * 2, 1]).to(device)
            domain_label[:self.batch_size] = 1

            # feature extraction
            z = self.model.feature_extractor(x)

            # domain classification, and image label prediction
            # note that the lamda is applied in forward function of domain classifier.
            domain_logits = self.domain_classifier(z, lamda_D)
            class_logits = self.model.label_predictor(z)

            # compute prediction loss and domain classification loss
            # notes the model applied gradient reversal layer.
            # the optimize should tune the discriminator to minimize the loss_D, and tune
            # the feature extractor to maximize the loss_D.
            loss_F = class_criterion(class_logits[:n], src_y)
            loss_D = domain_criterion(domain_logits, domain_label)

            # compute conditional entropy loss and virtual adversarial loss
            loss_C = 0 # conditional_criterion(class_logits[n:])
            loss_V = 0 # virtual_adversarial_criterion(x, class_logits)

            loss = loss_F + loss_D + lamda_C * loss_C + lamda_V * loss_V

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # compute accuracy (in source domain)
            source_acc = torch.sum(torch.argmax(class_logits[:n], dim=1) == src_y).item()
            source_acc /= self.batch_size

            # log to tensorboard
            metrics = {
                "loss_F": loss_F.item(), "loss_D": loss_D.item(),
                # "loss_C": loss_C.item(), "loss_V": loss_V.item(),
                "acc": source_acc,
            }
            writer.add_scalars("train", metrics, i)
            writer.add_scalars("hparams", { 'lambda_D': lamda_D }, i)

            # log to progress bar
            postfix = {
                "loss_F": f'{loss_F.item():.4f}',
                "loss_D": f'{loss_D.item():.4f}',
                # "loss_C": f'{loss_C.item():.4f}',
                # "loss_V": f'{loss_V.item():.4f}',
                "acc": f'{source_acc:.2%}',
            }
            pbar.set_postfix(postfix)

            # store checkpoint

        return self.model

    def fit(self) -> Model:
        return self.virtual_adversarial_domain_adaptation()


def train():
    # Imbalance case
    # Labeled data: 5000 images
    # Unlabeled data: 100k images
    source_dataset = ImageFolder(
        os.path.join(root_dir, 'train_data'), transform=source_transform
    )
    target_dataset = ImageFolder(
        os.path.join(root_dir, 'test_data'), transform=target_transform
    )

    model = ModelTrainer(
        hparams, {'source': source_dataset, 'target': target_dataset }
    ).fit()

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))

    return model

def inference(model: nn.Module):
    # Inference
    result = []

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(
        os.path.join(root_dir, 'test_data'), transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
    for (x, _) in tqdm(dataloader, ncols=0, desc='Inference'):
        x = x.cuda()

        logits = model(x)

        x = torch.argmax(logits, dim=1).cpu().detach().numpy()
        result.append(x)

    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv('DaNN_submission.csv', index=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--load-ckpt', type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        model = train()
        inference(model)

    if args.inference:
        model = Model()

        state_dict = torch.load('model.pth')['model']
        model.load_state_dict(state_dict)

        inference(model)

if __name__ == '__main__':
    main()
