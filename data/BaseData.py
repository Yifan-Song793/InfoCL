import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


class BaseData:
    def __init__(self, args):
        self.args = args
        self.id2label, self.label2id = self._read_labels()
        self.train_data, self.val_data, self.test_data = None, None, None

    def _read_labels(self):
        id2label = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'id2label.json')))
        label2id = {label: i for i, label in enumerate(id2label)}
        return id2label, label2id

    def read_and_preprocess(self, **kwargs):
        raise NotImplementedError

    def filter(self, labels, split='train'):
        if not isinstance(labels, list):
            labels = [labels]
        if isinstance(labels[0], str):
            labels = [self.label2id[label] for label in labels]
        split = split.lower()
        res = []
        for label in labels:
            if split == 'train':
                res += self.train_data[label]
            elif split in ['dev', 'val']:
                res += self.val_data[label]
            elif split == 'test':
                res += self.test_data[label]

        return res


class BaseDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data[idx]['idxs'] = idx
        return self.data[idx]

