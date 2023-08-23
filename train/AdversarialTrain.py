import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple, Dict, List
from types import SimpleNamespace
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
)


logger = logging.getLogger(__name__)


class FreeLB:
    def __init__(self, args):
        self.args = args

        self.adv_K = args.adv_K
        self.adv_lr = args.adv_lr
        self.adv_max_norm = args.adv_max_norm
        self.adv_init_mag = args.adv_init_mag
        self.adv_norm_type = args.adv_norm_type

    def attack(self,
               model,
               input_ids,
               labels,
               attention_mask=None,
               **kwargs
    ):
        if attention_mask is None:
            attention_mask = input_ids != 0
        embeds_init = model.bert.embeddings.word_embeddings(input_ids)

        if self.adv_init_mag > 0:
            input_length = torch.sum(attention_mask, dim=1)
            if self.adv_norm_type == 'l2':
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * attention_mask.unsqueeze(2)
                dims = input_length * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == 'linf':
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag,
                                                               self.adv_init_mag) * attention_mask.unsqueeze(2)

        else:
            delta = torch.zeros_like(embeds_init)

        loss_rec = 0

        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs_embeds = delta + embeds_init
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
            loss = outputs[0]
            loss /= self.adv_K

            loss_rec += loss.item()

            loss.backward()

            if astep == self.adv_K - 1:
                break

            delta_grad = delta.grad.clone().detach()
            if self.adv_norm_type == 'l2':
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask
                                 + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == 'linf':
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()

            embeds_init = model.bert.embeddings.word_embeddings(input_ids)

        loss_rec /= self.adv_K

        return loss_rec



def adversarial_train(
        model: Union[PreTrainedModel, nn.Module] = None,
        args: SimpleNamespace = None,
        num_train_epochs: int = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        optimizer: torch.optim.Optimizer = None,
        **kwargs,
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

    freelb = FreeLB(args)

    logger.info("***** Running adversarial training *****")
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
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # loss.backward()

            loss = freelb.attack(model, **inputs)

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)
            loss_rec.append(loss)

        progress_bar.set_postfix({"Loss": np.array(loss_rec).mean(), "lr": scheduler.get_last_lr()[0]})

    progress_bar.close()
