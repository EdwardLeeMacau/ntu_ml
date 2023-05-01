import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml
from dataset import CustomTensorDataset
from matplotlib import pyplot as plt
from model import VAELoss, create_model
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange
from utils import cycle, same_seeds

# send to GPU by default
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load hyperparameters
with open('params.yaml') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

# fix random seed for reproducibility
same_seeds(hparams['seed'])


def train():
    # create checkpoint directory
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    ckpt_dir = hparams['env']['checkpoint']
    ckpt_dir = os.path.join(ckpt_dir, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    # load hyperparameters
    root_dir = hparams['env']['dataset']
    num_iters = hparams['iterations']
    batch_size = hparams['batch-size']['train']
    model_type = hparams['model']['type']
    optimizer_kwargs = hparams['optimizer']['kwargs']

    # generic transforms, apply to both input and ground truth
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    # utility instances
    writer = SummaryWriter()

    # utility functions
    def preprocess(x):
        return x[:, 8:40, :, :]

    def prepare_dataset(fname: str) -> DataLoader:
        dataset = np.load(fname, allow_pickle=True)
        dataset = preprocess(dataset)
        dataset = torch.from_numpy(dataset)
        return CustomTensorDataset(dataset, transform=transform)

    # TODO: consider denoising autoencoder
    # specific transforms, only apply to input
    # additional_transform = transforms.Compose([
    #     GaussianNoise(0, 0.1),
    # ])

    # build train dataloader
    dataset = prepare_dataset(os.path.join(root_dir, 'trainingset.npy'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader = cycle(dataloader)

    # Load test images to preview anomality
    test_set = prepare_dataset(os.path.join(root_dir, 'testingset.npy'))
    test_img = next(
        iter(DataLoader(test_set, batch_size=batch_size, shuffle=False))
    ).float().to(device)
    test_img_cpu = test_img.clone().cpu()
    reconstruction_error = nn.MSELoss(reduction='none')

    # utility functions
    @torch.no_grad()
    def validate():
        model.eval()

        # forwarding to reconstruct images
        out = model(test_img)

        # use l2-norm as anomaly score, then sort images by score
        loss = reconstruction_error(out, test_img).sum([1, 2, 3])
        _, indices = torch.sort(loss, descending=True)

        # save image and write to tensorboard
        out, indices = out.cpu(), indices.cpu()

        grid = torch.zeros(size=(batch_size, 2, 3, 32, 64))
        grid[:, 0] = out[indices]
        grid[:, 1] = test_img_cpu[indices]
        grid = grid.view(-1, 3, 32, 64)

        grid = make_grid(grid, nrow=32, padding=2, pad_value=0, normalize=True, range=(-1, 1))
        writer.add_image('reconstruction', grid, i)

        # compute histogram and write to tensorboard
        loss = loss.cpu()
        writer.add_histogram('anomality', loss, i)

        # save model
        torch.save({
            'model': model.state_dict(),
        }, os.path.join(ckpt_dir, f'model-{i}.pth'))

        model.train()

    # Model, loss and optimizer
    model, criteria = create_model(model_type)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.95)

    model.to(device)
    model.train()

    # train autoencoder
    pbar = trange(1, num_iters, initial=1, ncols=0, desc='Training autoencoder')
    for i in pbar:
        # Prepare data
        img = next(dataloader).float()
        img = img.to(device)

        # forwarding
        out = model(img)

        # use multiple loss functions as guidance
        loss = criteria(out, img)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss = loss.item()

        # log metrics
        metrics = { 'loss': loss, }
        writer.add_scalars('loss', metrics, i)
        pbar.set_postfix({k: f'{v:.4f}' for k, v in metrics.items()})

        # regular checking anomaly performance
        if i % 1000 == 0:
            validate()

    model.eval()
    model.cpu()

    torch.save({
        'model': model.state_dict(),
    }, os.path.join(ckpt_dir, 'model.pth'))

@torch.no_grad()
def inference(checkpoint: str):
    """ Report the anomality score of target data """
    if checkpoint is None:
        raise ValueError('checkpoint path cannot be None')

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'checkpoint {checkpoint} does not exist')

    # load hyperparameters
    root_dir = hparams['env']['dataset']
    model_type = hparams['model']['type']
    batch_size = hparams['batch-size']['test']

    # utility functions
    def preprocess(x):
        return x[:, 8:40, :, :]

    def prepare_dataset(fname: str) -> DataLoader:
        dataset = np.load(fname, allow_pickle=True)
        dataset = preprocess(dataset)
        dataset = torch.from_numpy(dataset)
        return CustomTensorDataset(dataset)

    # build testing dataloader
    dataset = prepare_dataset(os.path.join(root_dir, 'testingset.npy'))
    test_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size, num_workers=1)
    criteria = VAELoss(nn.MSELoss(reduction='none'), alpha=0) if (model_type == 'vae') \
        else nn.MSELoss(reduction='none')

    # load trained model
    state_dict = torch.load(checkpoint)

    model, _ = create_model(model_type)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    # prediction file
    out_file = 'prediction.csv'

    anomality = list()
    for img in tqdm(dataloader, ncols=0):
        # ===================loading=====================
        img = img.float()
        img = img.to(device)

        # ===================forward=====================
        # TODO: Consider multi-sampling for VAE
        out = model(img)

        loss = criteria(out, img).sum([1, 2, 3])
        anomality.append(loss)

    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).view(-1).cpu().numpy()
    probability = np.zeros_like(anomality)

    vec = np.stack((anomality, probability), axis=1)
    df = pd.DataFrame(vec, columns=['score', 'probability'])

    # write out to file, only score is required
    df['score'].to_csv(out_file, index_label = 'ID')

    # plot score-prob correlation
    plt.figure(figsize=(12.8, 7.2))

    plt.scatter(df['score'], df['probability'])
    plt.xlabel('score')
    plt.ylabel('probability')
    plt.savefig('score-prob.png')

    plt.clf()

    # get top-k anomaly images
    k = 256
    df = df.sort_values(by=['score'], ascending=False)
    top_k = df[:k].index.values.tolist()

    img = torch.stack([dataset[i] for i in top_k])
    img = img.float()
    img = img.to(device)

    out = model(img)
    residual = torch.abs(out - img) / 2

    img = img.detach().cpu()
    out = out.detach().cpu()
    residual = residual.detach().cpu()
    residual = rgb_to_grayscale(residual)

    grid = make_grid(residual, nrow=32, padding=2, pad_value=0)
    save_image(grid, 'residual.png')

    # de-normalize output image
    grid = make_grid(out, nrow=32, padding=2, pad_value=0, normalize=True, value_range=(-1, 1))
    save_image(grid, 'reconstruct.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--load-ckpt', type=str)

    args = parser.parse_args()

    if args.train:
        train()

    if args.inference:
        inference(args.load_ckpt)
