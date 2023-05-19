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

# utility functions
def crop(*args):
    return (x[:, :, 8:40, :] for x in args)

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
    def prepare_dataset(fname: str) -> DataLoader:
        dataset = np.load(fname, allow_pickle=True)
        dataset = torch.from_numpy(dataset)
        return CustomTensorDataset(dataset, transform=transform)

    # build train dataloader
    dataset = prepare_dataset(os.path.join(root_dir, 'trainingset.npy'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader = cycle(dataloader)

    # Load test images to preview anomality
    test_set = prepare_dataset(os.path.join(root_dir, 'testingset.npy'))
    test_img = next(
        iter(DataLoader(test_set, batch_size=batch_size, shuffle=False))
    ).float().to(device)
    reconstruction_error = nn.MSELoss(reduction='none')

    # utility functions
    @torch.no_grad()
    def validate():
        model.eval()

        # forwarding to reconstruct images
        out = model(test_img)
        img = test_img                   # comment this line to boost performance
        # out, img = crop(out, test_img) # uncomment this line to boost performance

        # use l2-norm as anomaly score, then sort images by score
        loss = reconstruction_error(out, img).sum([1, 2, 3])
        _, indices = torch.sort(loss, descending=True)

        # save image and write to tensorboard
        out, img, indices = out.cpu(), img.cpu(), indices.cpu()

        # replace H to 32 to boost performance
        grid = torch.zeros(size=(batch_size, 2, 3, 64, 64))
        grid[:, 0] = out[indices]
        grid[:, 1] = img[indices]
        grid = grid.view(-1, 3, 64, 64)

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

        # crop images, only penalize the center part of generated images
        # leads some parameters waste at the output layer
        # out, img = crop(out, test_img) # uncomment this line to boost performance
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

    def prepare_dataset(fname: str) -> DataLoader:
        dataset = np.load(fname, allow_pickle=True)
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
        # prepare data, try some ensemble method.
        img = img.float()
        img = img.to(device)

        # forwarding
        out = model(img)
        # out, img = crop(out, test_img) # uncomment this line to boost performance
        loss = criteria(out, img).sum([1, 2, 3])

        # img = F.hflip(img)
        # out = model(img)
        # loss += criteria(out, img).sum([1, 2, 3])

        anomality.append(loss)

    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).view(-1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['score'])

    # write out to file, only score is required
    df['score'].to_csv(out_file, index_label = 'ID')

def analyze(df: pd.DataFrame):
    """ Analyze the anomaly score and visualize the result """
    root_dir = hparams['env']['dataset']

    # plot histogram
    plt.hist(df['score'], bins=100)
    plt.savefig('histogram.png')

    # get top-k anomaly images, uncropped
    dataset = np.load(os.path.join(root_dir, 'testingset.npy'), allow_pickle=True)
    dataset = torch.from_numpy(dataset)
    dataset = CustomTensorDataset(dataset, transform=transforms.Resize(32))

    k = 1024
    df = df.sort_values(by=['score'], ascending=False)

    top_k = df[:k].index.values.tolist()
    img = torch.stack([dataset[i] for i in top_k])
    grid = make_grid(img, nrow=64, padding=2, pad_value=0, normalize=True, value_range=(-1, 1))
    save_image(grid, 'anomaly.png')

    # get top-k normal images
    top_k = df[-k:].index.values.tolist()
    img = torch.stack([dataset[i] for i in top_k])
    grid = make_grid(img, nrow=64, padding=2, pad_value=0, normalize=True, value_range=(-1, 1))
    save_image(grid, 'normal.png')

def demo(checkpoint: str):
    # load hyperparameters
    root_dir = hparams['env']['dataset']
    model_type = hparams['model']['type']
    if model_type != 'fcn':
        raise ValueError('Demo only support FCN model, recv: {model_type}')

    # initialize model architecture and load model parameter
    model, _ = create_model(model_type)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    def prepare_img(fname: str):
        dataset = np.load(fname, allow_pickle=True)
        dataset = torch.from_numpy(dataset)
        dataset = CustomTensorDataset(dataset)
        img = dataset[0].unsqueeze(0).float()
        return img

    # load single image from training set
    img = prepare_img(os.path.join(root_dir, 'trainingset.npy'))
    img = img.reshape(1, -1).to(device)

    # retrieve latent vector, make 2 variant, reconstruct these latent vector
    latent = torch.zeros(3, 64, device=device)
    latent[:] = model.encoder(img)

    # the first variant: negate z[32]
    latent[1, 32] *= -1

    # the second variant: negate z[63]
    latent[2, -1] *= -1

    # reconstruct the image and de-normalize
    img = img.repeat(4, 1, 1, 1).reshape(4, 3, 64, 64)
    img[1:4] = model.decoder(latent).reshape(-1, 3, 64, 64)
    img = (img + 1) / 2

    # store the image
    img = make_grid(img, nrow=4, padding=2, pad_value=0, normalize=True, value_range=(0, 1))
    save_image(img, f'demo.png')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--load-ckpt', type=str)

    args = parser.parse_args()

    if args.train:
        train()

    if args.inference:
        inference(args.load_ckpt)
        # analyze(pd.read_csv('prediction.csv'))

    if args.demo:
        demo(args.load_ckpt)
