import os
import json
import jsonlines
from tqdm import tqdm
from collections import Counter
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from .BaseData import BaseData


class MAVENData(BaseData):
    def __init__(self, args):
        super().__init__(args)

    def read(self, file_name):
        raw_data = jsonlines.Reader(open(os.path.join(self.args.data_path, self.args.dataset_name, file_name)))

        res = {'sentence': [], 'labels': [], 'offset': []}
        for doc in raw_data:
            content = doc['content']
            for event in doc['events']:
                label = event['type']
                if label not in self.label2id:
                    continue
                label = self.label2id[label]
                for mention in event['mention']:
                    tokens = content[mention['sent_id']]['tokens']
                    offset_char_list = []
                    cur_pos = 0
                    for token in tokens:
                        offset_char_list.append((cur_pos, cur_pos + len(token)))
                        cur_pos += len(token) + 1
                    trigger_offset = [offset_char_list[mention['offset'][0]][0],
                                      offset_char_list[mention['offset'][1]-1][1]]
                    assert trigger_offset[0] < trigger_offset[1]
                    res['sentence'].append(' '.join(content[mention['sent_id']]['tokens']))
                    res['labels'].append(label)
                    res['offset'].append(trigger_offset)
        return res

    def preprocess(self, raw_data, tokenizer):
        res = []
        result = tokenizer(raw_data['sentence'], return_offsets_mapping=True)
        offset_mapping = result['offset_mapping']
        for idx in range(len(raw_data['sentence'])):
            start_pos, end_pos = -1, -1
            for i, item in enumerate(offset_mapping[idx]):
                if item[0] == 0 and item[1] == 0:
                    continue
                if item[0] == raw_data['offset'][idx][0]:
                    start_pos = i
                if item[1] == raw_data['offset'][idx][1]:
                    end_pos = i + 1
            assert start_pos != -1 and end_pos != -1
            assert start_pos <= end_pos
            res.append({
                'input_ids': result['input_ids'][idx],
                'attention_mask': result['attention_mask'][idx],
                'labels': raw_data['labels'][idx],
                'offset': [start_pos, end_pos]
            })
        return res

    def read_and_preprocess(self, tokenizer, seed=None):
        train_raw_data = self.preprocess(self.read('train.jsonl'), tokenizer)
        test_raw_data = self.preprocess(self.read('valid.jsonl'), tokenizer)

        self.train_data = {label: [] for label in range(len(self.id2label))}
        self.val_data = {label: [] for label in range(len(self.id2label))}
        self.test_data = {label: [] for label in range(len(self.id2label))}

        train_cnt = Counter()

        for item in train_raw_data:
            if train_cnt[item['labels']] >= 1000:
                continue
            self.train_data[item['labels']].append(item)
            train_cnt[item['labels']] += 1
        for item in test_raw_data:
            self.test_data[item['labels']].append(item)

