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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from transformers import (
    PreTrainedModel,
    DataCollator,
    TrainerCallback,
    EvalPrediction,
    set_seed,
    DataCollatorWithPadding,
)



logger = logging.getLogger(__name__)


@torch.no_grad()
def get_moco_hidden_states(
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
    num_examples = len(eval_dataset)

    hidden_states = []
    label_list = []
    model.eval()
    for step, inputs in enumerate(eval_dataloader):
        labels = inputs.pop('labels')
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        outputs = model(**inputs, moco=True)
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


def moco_train(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        num_train_epochs: int = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        optimizer: torch.optim.Optimizer = None,
):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    len_dataloader = len(train_dataloader)
    num_examples = len(train_dataset)
    max_steps = len_dataloader * num_train_epochs

    if optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        parameters = [
            {'params': [p for n, p in model.named_parameters() if 'bert' in n and not any(nd in n for nd in no_decay)],
             'lr': args.learning_rate, 'weight_decay': 1e-2},
             {'params': [p for n, p in model.named_parameters() if 'bert' in n and any(nd in n for nd in no_decay)],
             'lr': args.learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'bert' not in n and not any(nd in n for nd in no_decay)],
             'lr': args.classifier_learning_rate, 'weight_decay': 1e-2},
             {'params': [p for n, p in model.named_parameters() if 'bert' not in n and any(nd in n for nd in no_decay)],
             'lr': args.classifier_learning_rate, 'weight_decay': 0.0},
        ]
        optimizer = AdamW(parameters)

    lambda1 = lambda step: step / (args.warmup_epochs * len_dataloader) if step <= args.warmup_epochs * len_dataloader else 1
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    targets, target_labels = get_moco_hidden_states(
        model,
        args,
        data_collator=data_collator,
        eval_dataset=train_dataset,
        shuffle=True,
        return_type='pt',
        return_labels=True,
    )

    assert targets.size()[0] == target_labels.size()[0]
    if targets.size()[0] > args.moco_queue_size:
        targets = targets[:args.moco_queue_size]
        target_labels = target_labels[:args.moco_queue_size]

    model.init_target(targets, target_labels)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Train batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {max_steps}")

    progress_bar = tqdm(range(max_steps))

    for epoch in range(num_train_epochs):
        model.train()
        loss_rec = []
        for step, inputs in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            if len(set(inputs['labels'].cpu().tolist()) - set(model.queue_labels.cpu().tolist())) != 0:
                continue
            outputs = model(**inputs, moco=True)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)
            loss_rec.append(loss.item())
            if step % args.report_freq == 0:
                progress_bar.set_postfix({"Loss": np.array(loss_rec).mean(), "lr": scheduler.get_last_lr()[0]})
                loss_rec = []

    progress_bar.close()
