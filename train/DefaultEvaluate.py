import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple, Dict, List
import hydra
from omegaconf import DictConfig
from types import SimpleNamespace
from pathlib import Path
from functools import partial
from copy import deepcopy
from tqdm import tqdm

import numpy as np
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import evaluate
import transformers
from transformers import (
    PreTrainedModel,
    DataCollator,
)

from utils import confusion_matrix_view

logger = logging.getLogger(__name__)


@torch.no_grad()
def default_evaluate(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
        seen_labels: Optional[List[int]] = None,
        verbose: bool = True,
        **kwargs,
):
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    len_dataloader = len(eval_dataloader)
    num_examples = len(eval_dataset)

    logger.info("***** Running evaluating *****")
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

        logits = outputs.logits

        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        logits_mask[:, seen_labels] = False

        logits = torch.masked_fill(logits, logits_mask, float('-inf'))

        predicts = logits.max(dim=-1)[1]

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

    details = None
    if verbose:
        target_names = [args.id2label[label] for label in seen_labels]
        details = classification_report(golds, preds, labels=range(len(seen_labels)), target_names=target_names,
                                         zero_division=0, output_dict=True)
        logger.info(
            '\n' + classification_report(golds, preds, labels=range(len(seen_labels)), target_names=target_names,
                                         zero_division=0))
        # logger.info(f"confusion matrix\n{confusion_matrix_view(golds, preds, target_names, logger)}")

    return micro_f1, details
