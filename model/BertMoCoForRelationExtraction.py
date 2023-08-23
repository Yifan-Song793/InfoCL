from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertOnlyMLMHead,
    SequenceClassifierOutput,
)

from utils import InfoNCEWithQueue, SupConWithQueue, SupInfoNCE, SupConWithSelf


@dataclass
class NewToOldOutput(SequenceClassifierOutput):
    new_old_loss: Optional[torch.FloatTensor] = None


class BertMoCoForRelationExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ema_decay = config.global_args.ema_decay
        self.queue_size = config.global_args.moco_queue_size
        self.moco_lambda = config.global_args.moco_lambda
        self.moco_temperature = config.global_args.moco_temperature

        self.bert = BertModel(config)

        self.classifier_projection = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm([config.hidden_size]),
        )
        self.re_classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.online_pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.online_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm([config.hidden_size]),
            nn.Dropout(0.1)
        )
        self.online_prediction = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )

        self.register_buffer("target_encoder", None)
        self.register_buffer("target_pooler", None)
        self.register_buffer("target_projection", None)

        self.register_buffer("queue", torch.randn(self.queue_size, config.hidden_size))
        self.queue = F.normalize(self.queue, p=2, dim=1)
        self.register_buffer('queue_labels', torch.zeros(self.queue_size))

        # self.contrastive_loss_fct = SupConWithQueue(temp=self.moco_temperature)
        self.contrastive_loss_fct = SupInfoNCE(temp=self.moco_temperature)
        # self.contrastive_loss_fct = InfoNCEWithQueue(temp=self.moco_temperature)
        # self.contrastive_loss_fct = SupConWithSelf(temp=self.moco_temperature)

        self.post_init()


    def init_target(self, target=None, target_labels=None):
        self.init_target_net()
        self.init_queue(target, target_labels)

    def init_target_net(self):
        self.target_encoder = deepcopy(self.bert)
        self.target_pooler = deepcopy(self.online_pooler)
        self.target_projection = deepcopy(self.online_projection)

        self.target_encoder.eval()
        self.target_pooler.eval()
        self.target_projection.eval()

        for pm in self.target_encoder.parameters():
            pm.requires_grad = False
        for pm in self.target_pooler.parameters():
            pm.requires_grad = False
        for pm in self.target_projection.parameters():
            pm.requires_grad = False

    def init_queue(self, target=None, target_labels=None):
        if target is not None:
            # assert target.size() == (self.queue_size, self.config.hidden_size)
            self.queue = target.detach()
            self.queue_labels = target_labels
        else:
            # Todo: device
            self.queue = torch.randn(self.queue_size, self.config.hidden_size)
            self.queue_labels = torch.zeros(self.queue_size)
        self.queue = F.normalize(self.queue, p=2, dim=1)

        self.queue.requires_grad = False
        self.queue_labels.requires_grad = False

    @torch.no_grad()
    def target_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        subject_start_pos: Optional[torch.Tensor] = None,
        object_start_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = input_ids != 0

        outputs = self.target_encoder(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        last_hidden_states = outputs[0]

        idx = torch.arange(last_hidden_states.size(0)).to(last_hidden_states.device)
        ss_emb = last_hidden_states[idx, subject_start_pos]
        os_emb = last_hidden_states[idx, object_start_pos]
        # sent_emb = ((last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        raw_rel_hidden_states = torch.cat([ss_emb, os_emb], dim=-1)
        pooler_embedding = self.target_pooler(raw_rel_hidden_states)
        projection_embedding = self.target_projection(pooler_embedding)

        return projection_embedding

    @torch.no_grad()
    def ema_update(self):
        for op, tp in zip(self.bert.parameters(), self.target_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data

        for op, tp in zip(self.online_pooler.parameters(), self.target_pooler.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data

        for op, tp in zip(self.online_projection.parameters(), self.target_projection.parameters()):
            tp.data = self.ema_decay * tp.data + (1 - self.ema_decay) * op.data

    @torch.no_grad()
    def update_queue(self, key, labels):
        self.queue = torch.cat([key.detach(), self.queue], dim=0)
        self.queue = self.queue[0:self.queue_size]
        self.queue_labels = torch.cat([labels, self.queue_labels], dim=0)
        self.queue_labels = self.queue_labels[0:self.queue_size]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        subject_start_pos: Optional[torch.Tensor] = None,
        object_start_pos: Optional[torch.Tensor] = None,
        moco: Optional[bool] = False,
        new_old: Optional[bool] = False,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = input_ids != 0

        outputs = self.bert(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        last_hidden_states = outputs[0]

        idx = torch.arange(last_hidden_states.size(0)).to(last_hidden_states.device)
        ss_emb = last_hidden_states[idx, subject_start_pos]
        os_emb = last_hidden_states[idx, object_start_pos]
        # sent_emb = ((last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        raw_rel_hidden_states = torch.cat([ss_emb, os_emb], dim=-1)

        rel_hidden_states = self.classifier_projection(raw_rel_hidden_states)
        logits = self.re_classifier(rel_hidden_states)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if moco:
            online_pooler_embedding = self.online_pooler(raw_rel_hidden_states)
            online_projection_embedding = self.online_projection(online_pooler_embedding)
            query = self.online_prediction(online_projection_embedding)

            if not self.training:
                return SequenceClassifierOutput(
                    loss=None,
                    logits=logits,
                    hidden_states=online_projection_embedding,
                )

            self.ema_update()

            target = self.target_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                subject_start_pos=subject_start_pos,
                object_start_pos=object_start_pos,
            )
            target = target.detach()

            loss_contrastive = self.contrastive_loss_fct(query, target, self.queue, labels, self.queue_labels)
            loss += self.moco_lambda * loss_contrastive

            self.update_queue(target, labels)

        if new_old:
            online_pooler_embedding = self.online_pooler(raw_rel_hidden_states)
            online_projection_embedding = self.online_projection(online_pooler_embedding)
            query = self.online_prediction(online_projection_embedding)
            loss_new_old = self.contrastive_loss_fct(query, query, self.queue, labels, self.queue_labels)
            return NewToOldOutput(
                loss=loss,
                new_old_loss=loss_new_old,
                logits=logits,
                hidden_states=rel_hidden_states
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=rel_hidden_states
        )
