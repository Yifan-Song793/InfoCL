import os
import json
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from .BaseData import BaseData


class FewRelData(BaseData):
    def __init__(self, args):
        super().__init__(args)
        self.entity_markers = ["[E11]", "[E12]", "[E21]", "[E22]"]

    def preprocess(self, raw_data, tokenizer):
        subject_start_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[0])
        object_start_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[2])
        res = []
        result = tokenizer(raw_data['sentence'])
        for idx in range(len(raw_data['sentence'])):
            subject_start_pos = result['input_ids'][idx].index(subject_start_marker)
            object_start_pos = result['input_ids'][idx].index(object_start_marker)
            res.append({
                'input_ids': result['input_ids'][idx],
                'attention_mask': result['attention_mask'][idx],
                'subject_start_pos': subject_start_pos,
                'object_start_pos': object_start_pos,
                'labels': raw_data['labels'][idx],
            })
        return res

    def read_and_preprocess(self, tokenizer, seed=None):
        raw_data = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'data_with_marker.json')))

        train_data = {}
        val_data = {}
        test_data = {}

        if seed is not None:
            random.seed(seed)

        for label in tqdm(raw_data.keys(), desc="Load FewRel data"):
            cur_data = raw_data[label]
            random.shuffle(cur_data)
            train_raw_data = {"sentence": [], "labels": []}
            val_raw_data = {"sentence": [], "labels": []}
            test_raw_data = {"sentence": [], "labels": []}
            for idx, sample in enumerate(cur_data):
                sample["tokens"] = ' '.join(sample["tokens"])
                sample["relation"] = self.label2id[sample["relation"]]
                if idx < 420:
                    train_raw_data["sentence"].append(sample["tokens"])
                    train_raw_data["labels"].append(sample["relation"])
                elif idx < 420 + 140:
                    val_raw_data["sentence"].append(sample["tokens"])
                    val_raw_data["labels"].append(sample["relation"])
                else:
                    test_raw_data["sentence"].append(sample["tokens"])
                    test_raw_data["labels"].append(sample["relation"])

            train_data[self.label2id[label]] = self.preprocess(train_raw_data, tokenizer)
            val_data[self.label2id[label]] = self.preprocess(val_raw_data, tokenizer)
            test_data[self.label2id[label]] = self.preprocess(test_raw_data, tokenizer)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

