import os
import random

import torch
from typing import List, Optional, Union
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


TRAIN = 'train'
TEST = 'test'
VAL = 'val'

def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

# Reference:
#
# My implementation from previous course (Applied Deep Learning, Fall 2022).
# https://nol.ntu.edu.tw/nol/coursesearch/print_table.php?course_id=922%20U4340&class=&dpt_code=9440&ser_no=32336&semester=111-1&lang=CH
#
# GitHub URL:
# https://github.com/EdwardLeeMacau/ntucsie_adl/blob/master/intent_classification_and_slot_tagging/dataset.py#L69
class LibriSeqDataset(Dataset):
    IGN = -1

    def __init__(self, data: List, max_len: Optional[int] = None):
        self.data = data
        self.max_len = max_len

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @property
    def num_tokens(self):
        return sum([len(label) for _, label in self.data])

    def collate_fn(self, samples: List):
        sequences = [s[0] for s in samples] # Mel-Frequency cepstrum
        label = [s[1] for s in samples]     # Phoneme class

        feature_size = sequences[0].shape[1]
        batch_size = len(samples)
        length = torch.tensor([x.shape[0] for x in sequences], dtype=torch.long).flatten()
        to_len = torch.max(length).item() if self.max_len is None else self.max_len

        pad_sequences = torch.zeros((batch_size, to_len, feature_size), dtype=torch.float)
        for i, seq in enumerate(sequences):
            pad_sequences[i, :seq.shape[0]] = seq
        label = torch.tensor(pad_to_len(label, to_len, self.IGN), dtype=torch.long)
        mask = label != self.IGN

        return ({ 'sequence': pad_sequences, 'length': length, 'mask': mask }, label)

    def test_collate_fn(self, samples: List):
        sequences = [s[0] for s in samples] # Mel-Frequency cepstrum

        feature_size = sequences[0].shape[1]
        batch_size = len(samples)
        length = torch.tensor([x.shape[0] for x in sequences], dtype=torch.long).flatten()
        to_len = torch.max(length).item() if self.max_len is None else self.max_len

        pad_sequences = torch.zeros((batch_size, to_len, feature_size), dtype=torch.float)
        mask = torch.zeros((batch_size, to_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            pad_sequences[i, :seq.shape[0]] = seq
            mask[i, :seq.shape[0]] = 1

        return { 'sequence': pad_sequences, 'length': length, 'mask': mask }

"""**Helper functions to pre-process the training data from raw MFCC features of each utterance.**

A phoneme may span several frames and is dependent to past and future frames. \
Hence we concatenate neighboring phonemes for training to achieve higher accuracy. The **concat_feat** function concatenates past and future k frames (total 2k+1 = n frames), and we predict the center frame.

Feel free to modify the data preprocess functions, but **do not drop any frame** (if you modify the functions, remember to check that the number of frames are the same as mentioned in the slides)
"""

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd

    if concat_n < 2:
        return x

    # (T, MFCC_dim)
    seq_len, feature_dim = x.size(0), x.size(1)

    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def parse_gt(fpath: str) -> List[int]:
    gt = {}

    with open(fpath, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            gt[line[0]] = list(map(int, line[1:]))

    return gt

def fetch_usage(split, phone_path, seed, ratio):
    mode = TEST if split == TEST else TRAIN

    if mode == TRAIN:
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()

        rng = np.random.RandomState(seed)
        rng.shuffle(usage_list)
        train_len = int(len(usage_list) * ratio)
        usage_list = usage_list[:train_len] if split == TRAIN else usage_list[train_len:]

    elif mode == TEST:
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]

    return usage_list

def preprocess_data(split, feature_dir, phone_path, concat_nframes, train_ratio=0.8, random_seed=1213):
    class_num = 41 # NOTE: pre-computed, should not need change

    if split not in (TRAIN, VAL, TEST):
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    mode = TEST if split == TEST else TRAIN

    usage_list = fetch_usage(split, phone_path, random_seed, train_ratio)
    label_dict = parse_gt(os.path.join(phone_path, f'{mode}_labels.txt')) if mode == TRAIN else {}
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode == TRAIN:
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        # x.shape = (T, 39)
        x = torch.load(os.path.join(feature_dir, mode, f'{fname}.pt'))
        T = x.shape[0]

        # x.shape = (T, n * 39)
        x = concat_feat(x, concat_nframes)
        if mode == TRAIN:
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx + T, :] = x
        if mode == TRAIN:
            y[idx: idx + T] = label

        idx += T

    # Truncate unnecessary dimension
    X = X[:idx, :]
    if mode == TRAIN:
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == TRAIN:
        print(y.shape)
        return X, y
    else:
        return X

def preprocess_seqdata(split, feature_dir, phone_path, concat_nframes, train_ratio=0.8, random_seed=1213) -> List:
    class_num = 41 # NOTE: pre-computed, should not need change

    if split not in (TRAIN, VAL, TEST):
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    mode = TEST if split == TEST else TRAIN

    usage_list = fetch_usage(split, phone_path, random_seed, train_ratio)
    label_dict = parse_gt(os.path.join(phone_path, f'{mode}_labels.txt')) if mode == TRAIN else {}
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    data = []
    for _, fname in tqdm(enumerate(usage_list)):
        # x.shape = (T, 39)
        x = torch.load(os.path.join(feature_dir, mode, f'{fname}.pt'))
        x = concat_feat(x, concat_nframes)
        y = label_dict[fname] if mode == TRAIN else None

        data.append((x, y))

    return data
