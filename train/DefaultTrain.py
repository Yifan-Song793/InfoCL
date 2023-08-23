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

from transformers import (
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    DataCollator,
    TrainerCallback,
    EvalPrediction,
    set_seed,
    DataCollatorWithPadding,
)


logger = logging.getLogger(__name__)


def default_train(
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
    num_examples = len_dataloader * args.train_batch_size
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
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            progress_bar.update(1)
            loss_rec.append(loss.item())
            if step % args.report_freq == 0:
                progress_bar.set_postfix({"Loss": np.array(loss_rec).mean()})
                loss_rec = []

    progress_bar.close()
