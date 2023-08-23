import numpy as np
from texttable import Texttable
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

import logging
from typing import Optional, Union, Callable, Tuple, Dict, List, Any
from types import SimpleNamespace
from dataclasses import dataclass, field
from tqdm import tqdm
from copy import deepcopy
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    DataCollator,
    TrainerCallback,
    EvalPrediction,
    set_seed,
    DataCollatorWithPadding,
)
from transformers.utils import PaddingStrategy

from data import BaseData


logger = logging.getLogger(__name__)


def confusion_matrix_view(true_label, pred_label, labels, logger):
    cf_matrix = confusion_matrix(true_label, pred_label)

    table = Texttable()
    table.add_row([" "] + [i[:8] for i in labels])
    table.set_max_width(2000)
    for idx, r in enumerate(cf_matrix):
        table.add_row([labels[idx][:8]] + [str(i) for i in cf_matrix[idx]])
    return table.draw()


def compute_forgetting_rate(detail_metrics, task_seq, id2label, mode='task'):
    label_metric_rec = {}
    for metric in detail_metrics:
        for label in metric:
            if label not in id2label:
                continue
            if label not in label_metric_rec:
                label_metric_rec[label] = []
            label_metric_rec[label].append(metric[label]['f1-score'] * 100)

    if mode == 'label':
        fr_per_label = []
        for label in label_metric_rec:
            if len(label_metric_rec) == 1:
                continue
            fr_tmp = max(label_metric_rec[label]) - label_metric_rec[label][-1]
            fr_per_label.append(fr_tmp)
        return np.mean(fr_per_label)
    elif mode == 'task':
        task_fr_rec = []
        for task in task_seq[:-1]:
            cur_task_metric = [label_metric_rec[id2label[label]] for label in task]
            cur_task_metric = np.array(cur_task_metric)
            cur_task_metric = np.mean(cur_task_metric, axis=0)
            cur_task_fr = np.max(cur_task_metric) - cur_task_metric[-1]
            task_fr_rec.append(cur_task_fr)
        return np.mean(task_fr_rec)


@torch.no_grad()
def get_hidden_states(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
        shuffle: bool = False,
        return_type: Optional[str] = 'np',
        return_labels: bool = False,
):
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

    len_dataloader = len(eval_dataloader)
    num_examples = len_dataloader * args.eval_batch_size

    hidden_states = []
    label_list = []
    model.eval()
    for step, inputs in enumerate(eval_dataloader):
        labels = inputs.pop('labels')
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden_state = outputs.hidden_states
        hidden_states.append(hidden_state)
        label_list.append(labels)

    hidden_states = torch.cat(hidden_states, dim=0)
    label_list = torch.cat(label_list, dim=0).to(hidden_states.device)
    if return_type == 'np':
        hidden_states = hidden_states.cpu().numpy()
        label_list = label_list.cpu().numpy()

    if return_labels:
        return hidden_states, label_list
    else:
        return hidden_states


def get_prototype(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
):
    hidden_states = get_hidden_states(model, args, data_collator, eval_dataset, return_type='pt')
    proto = torch.mean(hidden_states, dim=0)
    return proto


def select_exemplars(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
):
    memory_size = args.memory_size
    features = get_hidden_states(model, args, data_collator, eval_dataset, return_type='np')
    distances = KMeans(n_clusters=memory_size, n_init=memory_size, random_state=0).fit_transform(features)
    exemplars = []
    for k in range(memory_size):
        exemplar_idx = np.argmin(distances[:, k])
        exemplars.append(eval_dataset[exemplar_idx])
    return exemplars



@dataclass
class XMLMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    whole_data: BaseData = None
    mlm_probability: float = 0.3

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        
        labels = batch['labels']
        xmlm_inputs = []
        for label in labels.tolist():
            cur_data = self.whole_data.filter(label)
            xmlm_inputs.append(deepcopy(random.choice(cur_data)))

        xmlm_batch = self.tokenizer.pad(xmlm_inputs, return_tensors='pt', pad_to_multiple_of=self.pad_to_multiple_of)

        # xmlm_batch = deepcopy(batch)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = xmlm_batch.pop("special_tokens_mask", None)
        batch["xmlm_input_ids"], batch["xmlm_labels"] = self.mask_tokens(
            xmlm_batch['input_ids'], special_tokens_mask=special_tokens_mask
        )

        batch['xmlm_attention_mask'] = xmlm_batch['attention_mask']

        return batch

    def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels
