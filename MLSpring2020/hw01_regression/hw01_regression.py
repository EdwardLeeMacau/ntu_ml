import argparse
import csv
import math
import os
from typing import Tuple

import numpy as np
import pandas as pd


def read_rawdata(fname: str, is_testing: bool) -> np.array:
    """
    Parameters
    ----------
    fname : str
        name of the file to be loaded
    is_testing : bool
        is it testing data? (no output data)

    Returns
    -------
    raw_data : np.array
        data in format np.array
    """
    pd_header = None if is_testing else 'infer'
    data = pd.read_csv(fname, header=pd_header, encoding='big5')

    # Data pre-processing
    start_idx = 2 if is_testing else 3
    data = data.iloc[:, start_idx:]

    # Data cleaning
    data[data == 'NR'] = 0

    # Pack as raw_data
    raw_data = data.to_numpy()

    return raw_data

def segment_rawdata(raw_data: np.array) -> Tuple[np.array, np.array]:
    # Extract features
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]

        month_data[month] = sample

    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue

                x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
                y[month * 471 + day * 24 + hour, :] = month_data[month][9, day * 24 + hour + 9]

    return (x, y)

def save_model(dirname: str, mean_x: np.array, std_x: np.array, w: np.array):
    """
    Parameters
    ----------
    dirname : str
        directory to store the model
    mean_x : np.array
        mean value of vector x
    std_x : np.arary
        standard value of vector x
    weight : np.array
        weight of regression model of vector x
    """
    np.save(os.path.join(dirname, 'mean_x.npy'), mean_x)
    np.save(os.path.join(dirname, 'std_x.npy'), std_x)
    np.save(os.path.join(dirname, 'weight.npy'), w)

    return

def load_model(dirname: str) -> Tuple[np.array, np.array, np.array]:
    """
    Parameters
    ----------
    dirname : str
        directory to store the model

    Returns
    -------
    model: tuple
        mean_x, std_x, and weight of regression model
    """
    mean_x = np.load(os.path.join(dirname, 'mean_x.npy'))
    std_x = np.load(os.path.join(dirname, 'std_x.npy'))
    weight = np.load(os.path.join(dirname, 'weight.npy'))

    return (mean_x, std_x, weight)

def z_normalize_fit(x) -> Tuple[np.array, np.array]:
    return np.mean(x, axis=0), np.std(x, axis=0)

def z_normalize_transform(x, mean, std) -> np.array:
    # i : number of training data
    # j : number of features
    for i in range(len(x)):
        for j in range(len(x[0])):
            if std[j] != 0:
                x[i][j] = (x[i][j] - mean[j]) / std[j]

    return x

def inference():
    data = read_rawdata('hw01_data/test.csv', True)

    x = np.empty([240, 18 * 9], dtype=np.float)
    for i in range(240):
        x[i, :] = data[18 * i: 18 * (i + 1), :].reshape(1, -1)

    mean_x, std_x, w = load_model('hw01_model')
    x = z_normalize_transform(x, mean_x, std_x)

    # Inference
    dim = 18 * 9 + 1
    x = np.concatenate((np.ones([240, 1]), x), axis=1).astype(np.float)
    y = np.dot(x, w)

    return y

def train_test_split(x, y, test_size=0.2):
    assert (x.shape[0] == y.shape[0])

    split = math.floor(len(x) * (1 - test_size))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    return (x_train, x_test, y_train, y_test)

def regression_model_fit(x_train, y_train, x_val, y_val):
    # Statistics
    num_train, num_val = x_train.shape[0], x_val.shape[0]

    # Training
    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    adagrad = np.zeros([dim, 1])

    lr = 100            # Learning rate
    iterates = 1000     # Times to iterate
    eps = 1e-10         # Eplison for AdaGrad algorithm

    for t in range(iterates):
        # Loss: L2 Norm
        loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train, 2)) / num_train)

        grad = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train)
        adagrad += grad ** 2
        w = w - lr * grad / np.sqrt(adagrad + eps)

        # Train Loss
        if (t % 100 == 0):
            print("Train loss {} / {}: {}".format(t, iterates, loss))

        # Validation Loss
        if (t % 100 == 0):
            loss = np.sqrt(np.sum(np.power(np.dot(x_val, w) - y_val, 2)) / num_val)
            print("Validation loss {} / {}: {}".format(t, iterates, loss))

    return w

def train():
    x, y = segment_rawdata(read_rawdata('hw01_data/train.csv', False))

    # Add bias term
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(np.float)

    # Split dataset into train set and validation set
    # This function
    x_train, x_val, y_train, y_val = train_test_split(x, y, 0.2)

    # Z Normalization :
    # Get mean, std on train set. Then apply to both train and validation set
    mean_x, std_x = z_normalize_fit(x_train)
    x_train = z_normalize_transform(x_train, mean_x, std_x)
    x_val = z_normalize_transform(x_val, mean_x, std_x)

    # Train the regression model
    w = regression_model_fit(x_train, y_train, x_val, y_val)

    save_model('hw01_model', mean_x, std_x, w)

    return

def main():
    parser = argparse.ArgumentParser(description='PM2.5 prediction (regression model)')
    parser.add_argument('--train', action='store_true', default=False, help='Train and store model.')
    parser.add_argument('--test', action='store_true', default=False, help='Inference answer for test data.')
    args = parser.parse_args()

    if args.train is True:
        train()
        return

    if args.test is True:
        y = inference()

        with open('hw01_data/submit.csv', mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            header = ['id', 'value']
            csv_writer.writerow(header)
            for i in range(240):
                row = ['id_' + str(i), y[i][0]]
                csv_writer.writerow(row)

        return

    _, _, w = load_model('hw01_model')
    print(w)

    return

if __name__ == "__main__":
    main()
