import argparse
import csv
import math
import os
from pyexpat import features
from re import L
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

def train_test_split(x: np.array, y: np.array, test_size=0.25) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Parameters
    ----------
    x, y : np.array
        Input data and corresponding labels
    test_size : float, default=0.25
        Ratio of test set size

    Returns
    -------
    x_train, x_test, y_train, y_test : np.array

    Raises
    ------
    AssertionError
        if x.shape[0] is not equal to y.shape[0]
    """
    assert (x.shape[0] == y.shape[0])

    split = math.floor(len(x) * (1 - test_size))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    return (x_train, x_test, y_train, y_test)

def z_normalize(x: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    Parameters
    ----------
    x : np.array
        Input data stored in 2D array
    """
    mean = np.mean(x, axis=0).reshape(-1)
    std = np.std(x, axis=0).reshape(-1)
    x = (x - mean) / (std + 1e-8)
    return x, mean, std

def shuffle(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    x, y : np.array
        Input data and corresponding labels

    Returns
    -------
    x, y : np.array
        Shuffled input data and corresponding labels

    Raises
    ------
    AssertionError
        if x.shape[0] is not equal to y.shape[0]
    """
    assert (x.shape[0] == y.shape[0])

    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)

    return (x[idx], y[idx])

def sigmoid(z: np.array) -> np.array:
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)

def logistic_regression(x: np.array, w: np.array, b: np.array) -> np.array:
    return sigmoid(np.matmul(x, w) + b)

def predict(x: np.array, w: np.array, b: np.array) -> np.array:
    return np.round(logistic_regression(x, w, b)).astype(np.int)

def accuracy(predict: np.array, label: np.array) -> float:
    """
    Parameters
    ----------
    predict, label : np.array
        Predicted label and golden answer

    Returns
    -------
    acc : float
        accuracy
    """
    return 1 - np.mean(np.abs(predict - label))

def cross_entropy(predict, label) -> float:
    return -np.dot(label, np.log(predict)) - np.dot((1 - label), np.log(1 - predict))

def gradient(x: np.array, y: np.array, w: np.array, b: np.array) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    x, y, w, b : np.array

    Returns
    -------
    w_grad, b_grad : np.array
    """
    predict = logistic_regression(x, w, b)
    err = y - predict
    w_grad = -np.sum(err * x.transpose(), 1)
    b_grad = -np.sum(err)

    return w_grad, b_grad

def inference(w, b):
    # Load test set
    x_fpath = './hw02_data/X_test'
    y_fpath = './hw02_data/output_{}.csv'

    with open(x_fpath, 'r') as f:
        next(f)
        x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=np.float)

    # Prediction
    pred = predict(x, w, b)

    # Write to files
    with open(y_fpath.format('logistic'), 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(pred):
            f.write('{},{}\n'.format(i, label))

    return pred

def load_model(fname: str) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Load model from .npy file
    """
    w = np.load(fname)
    w, mean_x, std_x, b = w[:-3], w[-3], w[-2], w[-1]
    return (w, mean_x, std_x, b)

def save_model(fname: str, mean_x: np.array, std_x: np.array, w: np.array, b: np.array):
    """
    Save model to .npy file
    """
    np.save(fname, np.concatenate([w, mean_x, std_x, b]))
    return

def train_generative():
    x_fpath = './hw02_data/X_train'
    y_fpath = './hw02_data/Y_train'

    with open(x_fpath) as f:
        next(f)
        x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=np.float)

    with open(y_fpath) as f:
        next(f)
        y = np.array([line.strip('\n').split(',')[1] for line in f], dtype=np.float)

    # x, mean, std = z_normalize(x)
    mean = 0
    std = 1

    # Input data dimension
    features_num = x.shape[1]

    # Compute in-class mean and variance
    mask = y == 0
    x_0, x_1 = x[mask], x[1 - mask]

    mean_0 = np.mean(x_0, axis=0)
    mean_1 = np.mean(x_1, axis=0)

    cov_0 = np.zeros((features_num, features_num))
    cov_1 = np.zeros((features_num, features_num))

    for x in x_0:
        cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / x_0.shape[0]

    for x in x_1:
        cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / x_1.shape[0]

    # Shared covariance is taken as a weighted average of individual in-class covariance.
    cov = (cov_0 * x_0.shape[0] + cov_1 * x_1.shape[0]) / x.shape[0]

    # Compute inverse of covariance matrix.
    u, s, v = np.linalg.svd(cov, full_matrices=False)
    inv = np.matmul(v.T * 1 / s, u.T)

    # Directly compute weights and bias
    w = np.dot(inv, mean_0 - mean_1)
    b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + (-0.5) * np.dot(mean_1, np.dot(inv, mean_1)) \
        + np.log(float(x_0.shape[0]) / x_1.shape[0])

    pred = 1 - predict(x, w, b)
    print('Training accuracy: {}'.format(accuracy(pred, y)))

    save_model(os.path.join('hw02_model', 'generative.npy'), mean, std, w, np.array([b]))

    return

def train_logistic():
    x_fpath = './hw02_data/X_train'
    y_fpath = './hw02_data/Y_train'

    with open(x_fpath) as f:
        next(f)
        x = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=np.float)

    with open(y_fpath) as f:
        next(f)
        y = np.array([line.strip('\n').split(',')[1] for line in f], dtype=np.float)

    # Z-Normalization
    x, mean, std = z_normalize(x)

    # Split dataset into train set and validation set
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

    # Input data dimension
    features_num = x.shape[1]
    train_size = x_train.shape[0]
    val_size = x_val.shape[0]

    # Initialize weight
    w = np.zeros((features_num, ))
    b = np.zeros((1, ))

    # Hyperparameters
    max_iter = 10
    batch_size = 8
    lr = 0.2

    # Metrics tracing
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    step = 1
    for epoch in range(max_iter):
        # Random shuffle at the beginning of each epoch
        x_train, y_train = shuffle(x_train, y_train)

        # Minibatch training
        for idx in range(math.floor(train_size / batch_size)):
            start_idx, end_idx = idx * batch_size, (idx + 1) * batch_size
            x, y = x_train[start_idx: end_idx], y_train[start_idx: end_idx]

            # Compute the gradient
            w_grad, b_grad = gradient(x, y, w, b)

            # Update parameters with gradient descent method
            # Learning rate decay with time
            w -= lr / np.sqrt(step) * w_grad
            b -= lr / np.sqrt(step) * b_grad

            step += 1

        y_pred = logistic_regression(x_train, w, b)
        train_acc.append(accuracy(np.round(y_pred), y_train))
        train_loss.append(cross_entropy(y_pred, y_train) / train_size)

        y_pred = logistic_regression(x_val, w, b)
        val_acc.append(accuracy(np.round(y_pred), y_val))
        val_loss.append(cross_entropy(y_pred, y_val) / val_size)

    print('Training loss: {}'.format(train_loss[-1]))
    print('Validation loss: {}'.format(val_loss[-1]))
    print('Training accuracy: {:.2%}'.format(train_acc[-1]))
    print('Validation accuracy: {:.2%}'.format(val_acc[-1]))

    save_model(os.path.join('hw02_model', 'logistic.npy'), mean, std, w, b)

    return

def main():
    parser = argparse.ArgumentParser(description='Income classification')
    parser.add_argument('--train', type=str, choices=['logistic', 'generative'], help='Train and store model.')
    parser.add_argument('--test', type=str, choices=['logistic', 'generative'], help='Inference data.')
    args = parser.parse_args()

    if args.train == "logistic":
        print(">>> Train logistic regression model")
        train_logistic()
        return

    if args.train == "generative":
        print(">>> Train generative model")
        train_generative()
        return

    if args.test == "logistic":
        print(">>> Test logistic regression model")
        w, b = load_model(os.path.join('hw02_model', 'weight_logistic.npy'))
        inference(w, b)
        return

    if args.test == "generative":
        print(">>> Test generative model")
        return

    return
if __name__ == "__main__":
    main()
