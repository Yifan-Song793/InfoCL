import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from .BaseData import BaseData


class HWU64Data(BaseData):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, raw_data, label, tokenizer):
        res = []
        result = tokenizer(raw_data)
        for idx in range(len(raw_data)):
            res.append({
                'input_ids': result['input_ids'][idx],
                'attention_mask': result['attention_mask'][idx],
                'labels': label,
            })
        return res

    def read_and_preprocess(self, tokenizer, seed=None):
        raw_data = pd.read_csv(open(os.path.join(self.args.data_path, self.args.dataset_name, 'NLU-Data-Home-Domain-Annotated-All.csv')), sep=";")

        if seed is not None:
            random.seed(seed)

        raw_data['label'] = raw_data['scenario'] + '_' + raw_data['intent']
        delete_columns = ['answer_annotation', 'scenario', 'intent', 'userid', 'answerid', 'suggested_entities', 'answer', 'question', 'status', 'notes']
        for column in delete_columns:
            del raw_data[column]

        raw_data = raw_data.dropna(axis=0, how='any')

        self.train_data = {label: [] for label in range(len(self.id2label))}
        self.val_data = {label: [] for label in range(len(self.id2label))}
        self.test_data = {label: [] for label in range(len(self.id2label))}

        for label in self.id2label:
            cur_data = raw_data[raw_data['label'] == label]['answer_normalised'].tolist()
            cur_data = self.preprocess(cur_data, self.label2id[label], tokenizer)
            random.shuffle(cur_data)
            total_cnt = len(cur_data) if len(cur_data) < 195 else 194
            test_cnt = total_cnt // 10
            self.train_data[self.label2id[label]] = cur_data[test_cnt:total_cnt]
            self.test_data[self.label2id[label]] = cur_data[:test_cnt]
