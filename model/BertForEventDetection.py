from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    SequenceClassifierOutput
)


class BertForEventDetection(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        self.linear_transform = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm([config.hidden_size]),
        )

        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.supcon_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, 64),
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        last_hidden_state = outputs.last_hidden_state

        trigger_mask = torch.zeros_like(attention_mask).to(attention_mask.device)
        batch_size = trigger_mask.size(0)
        for i in range(batch_size):
            trigger_mask[i, offset[i][0]:offset[i][1]] = 1

        hidden_states = ((last_hidden_state * trigger_mask.unsqueeze(-1)).sum(1) / trigger_mask.sum(-1).unsqueeze(-1))

        # hidden_states = self.linear_transform(hidden_states)

        loss = None
        logits = self.classifier(hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
