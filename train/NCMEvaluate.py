import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple, Dict, List
from types import SimpleNamespace
from tqdm import tqdm

import numpy as np
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    PreTrainedModel,
    DataCollator,
)

from utils import confusion_matrix_view, get_prototype
from data import BaseData, BaseDataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def ncm_evaluate(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
        cur_train_data: Optional[BaseData] = None,
        memory_data: Optional[Dict[int, List]] = None,
        cur_labels: Optional[List[int]] = None,
        seen_labels: Optional[List[int]] = None,
        verbose: bool = True,
        **kwargs,
):
    protos = []
    proto2id = []
    for label in seen_labels:
        if label in cur_labels:
            tmp_dataset = cur_train_data.filter(label, 'train')
        else:
            tmp_dataset = BaseDataset(memory_data[label])
        proto = get_prototype(model, args, data_collator, tmp_dataset)
        protos.append(proto)
        proto2id.append(label)

    protos = torch.stack(protos, dim=0)
    # protos = F.normalize(protos, p=2, dim=1)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    len_dataloader = len(eval_dataloader)
    num_examples = len_dataloader * args.eval_batch_size

    logger.info("***** Running ncm evaluating *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Eval batch size = {args.eval_batch_size}")

    progress_bar = tqdm(range(len_dataloader))

    golds = []
    preds = []

    model.eval()
    for step, inputs in enumerate(eval_dataloader):
        labels = inputs.pop('labels')
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        outputs = model(**inputs)

        hidden_states = outputs.hidden_states

        # hidden_states = F.normalize(hidden_states, p=2, dim=1)

        dist = F.cosine_similarity(hidden_states.unsqueeze(1), protos.unsqueeze(0), dim=-1)
        # logits = -euclidean_dist(hidden_states, protos)

        predicts = dist.max(dim=-1)[1]
        predicts = torch.tensor([proto2id[item] for item in predicts]).to(labels.device)

        predicts = predicts.cpu().tolist()
        labels = labels.cpu().tolist()
        pred_rel_t = [seen_labels.index(pred) for pred in predicts]
        gold_rel_t = [seen_labels.index(label) for label in labels]
        golds.extend(gold_rel_t)
        preds.extend(pred_rel_t)

        progress_bar.update(1)

    progress_bar.close()

    micro_f1 = f1_score(golds, preds, average='micro')
    logger.info("Micro F1 {}".format(micro_f1))

    if verbose:
        target_names = [args.id2label[label] for label in seen_labels]
        logger.info(
            '\n' + classification_report(golds, preds, labels=range(len(seen_labels)), target_names=target_names,
                                         zero_division=0))
        logger.info(f"confusion matrix\n{confusion_matrix_view(golds, preds, target_names, logger)}")

    return micro_f1
