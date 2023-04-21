import numpy as np
import torch
import os
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import argparse
import pandas as pd
from dataset import CustomTensorDataset
from model import FCNAutoEncoder, ConvAutoEncoder, VAE, loss_vae
from utils import same_seeds
import yaml

"""# Loading data"""

root_dir = "../../../dataset/anomaly"

train_set = np.load(os.path.join(root_dir, 'trainingset.npy'), allow_pickle=True)
test_set = np.load(os.path.join(root_dir, 'testingset.npy'), allow_pickle=True)

same_seeds(48763)

# Training hyperparameters
num_epochs = 50
batch_size = 2000
learning_rate = 1e-3

# Build training dataloader
x = torch.from_numpy(train_set)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'vae'
model_classes = {
    'fcn': FCNAutoEncoder(), 'cnn': ConvAutoEncoder(), 'vae': VAE()
}

def train():
    model = model_classes[model_type]
    model = model.cuda()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = np.inf
    model.train()

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        tot_loss = list()

        for data in train_dataloader:

            # ===================loading=====================
            img = data.float().cuda()
            if model_type in ['fcn']:
                img = img.view(img.shape[0], -1)

            # ===================forward=====================
            output = model(img)
            if model_type in ['vae']:
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)

            tot_loss.append(loss.item())

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================save_best====================
        mean_loss = np.mean(tot_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, 'best_model_{}.pt'.format(model_type))

        # ===================log========================
        pbar.set_postfix({
            'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
            'loss': f'{mean_loss:.4f}',
        })

        # ===================save_last========================
        torch.save(model, 'last_model_{}.pt'.format(model_type))


@torch.no_grad()
def inference():
    eval_batch_size = 200

    # build testing dataloader
    data = torch.tensor(test_set, dtype=torch.float32)
    test_dataset = CustomTensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
    eval_loss = nn.MSELoss(reduction='none')

    # load trained model
    checkpoint_path = f'last_model_{model_type}.pt'
    model = torch.load(checkpoint_path)
    model.eval()

    # prediction file
    out_file = 'prediction.csv'

    anomality = list()
    for i, data in enumerate(test_dataloader):
        img = data.float().cuda()
        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)

        output = model(img)
        if model_type in ['vae']:
            output = output[0]

        if model_type in ['fcn']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)

    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test_set), 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['score'])
    df.to_csv(out_file, index_label = 'ID')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        inference()
