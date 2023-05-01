import os
import pickle

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from utils import same_seeds

# load hyperparameters
with open('params.yaml') as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

# load root directory
root_dir = hparams['env']['dataset']

# fix random seed
same_seeds(0)

def preprocess(x):
    x = gaussian_filter(x, sigma=(0, 1, 1, 0))
    return x[:, 8:40, :, :]

def rgb2gray(x):
    """ Convert image from RGB to Grayscale """
    return np.dot(x[..., :3], [0.299, 0.587, 0.114])

def normalize(x):
    """ Normalize image from 0 ~ 255 to -1 ~ 1 """
    return 2. * (x / 255.) - 1.

def anomality(X, m: PCA, reduction: str = 'mean') -> np.ndarray:
    """ Compute anomality of input x """
    if reduction not in ['mean', 'sum']:
        raise ValueError('Invalid reduction method.')

    Z = m.transform(X)
    x_hat = m.inverse_transform(Z)

    # measure reconstruction error with mean-square error
    err = np.power(X - x_hat, 2)

    if reduction == 'mean':
        err = np.mean(err, axis=1)
    elif reduction == 'sum':
        err = np.sum(err, axis=1)

    return err

def train():
    fname = os.path.join(root_dir, 'trainingset.npy')

    X = np.load(fname, allow_pickle=True)
    X = preprocess(X)

    print(f"{X.shape=}")

    X = normalize(X)
    X = X.reshape(X.shape[0], -1)

    print('Training PCA model...')

    pca = PCA(n_components=512)
    pca.fit(X)

    return pca

def main():
    if not os.path.exists("./pca.pkl"):
        pca = train()
        with open("./pca.pkl", "wb") as f:
            pickle.dump(pca, f)

        print('PCA model saved.')
    else:
        with open("./pca.pkl", "rb") as f:
            pca = pickle.load(f)

        print('PCA model loaded.')

    fname = os.path.join(root_dir, 'testingset.npy')
    X = np.load(fname, allow_pickle=True)
    X = preprocess(X)

    print(f'{X.shape=}')

    X = normalize(X)
    X = X.reshape(X.shape[0], -1)

    mse = anomality(X, pca, 'sum')
    df = pd.DataFrame(mse, columns=['score'])

    # save reconstruction error
    out_file = 'prediction.csv'
    df['score'].to_csv(out_file, index_label='ID')

    # plot anomaly score distribution
    plt.figure(figsize=(12.8, 7.2))
    plt.hist(df['score'], bins=100)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.savefig('anomaly_score.png')
    plt.clf()

    # plot 256 most anomalous images
    imgs = np.argsort(df['score'])[::-1][:256].tolist()
    X = X.reshape(X.shape[0], 32, 64, 3)
    X = (X + 1.) / 2.
    plt.figure(figsize=(12.8, 7.2))
    for i, idx in enumerate(imgs):
        plt.subplot(16, 16, i + 1)
        plt.imshow(X[idx])
        plt.axis('off')

    plt.savefig('anomaly.png')
    plt.clf()


if __name__ == '__main__':
    main()
