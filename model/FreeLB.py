import numpy as np
from typing import Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import count


class FreeLB:
    def __init__(self, args):
        self.args = args
        self.adv_K = 20
        self.adv_lr = 1e-2
        self.adv_max_norm = 0
        self.adv_init_mag = 0
        self.adv_norm_type = 'l2'

    def attack(self,
               model,
               input_ids,
               labels,
               attention_mask=None,
               subject_start_pos=None,
               object_start_pos=None,
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
                subject_start_pos=subject_start_pos,
                object_start_pos=object_start_pos,
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
