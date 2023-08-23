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


@dataclass
class REOutput(SequenceClassifierOutput):
    loss_mi: Optional[torch.FloatTensor] = None
    rel_hidden_states: Optional[torch.FloatTensor] = None
    supcon_hidden_states: Optional[torch.FloatTensor] = None


class BertForRelationExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        self.classifier_projection = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm([config.hidden_size]),
        )
        # self.linear_transform = nn.Linear(768 * 2, 768, bias=True)

        # self.dropout = nn.Dropout(drop)
        # self.linear = nn.Linear(768 * 2, 768, bias=True)
        # self.layer_normalization = nn.LayerNorm([768])
        self.re_classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

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
        subject_start_pos: Optional[torch.Tensor] = None,
        object_start_pos: Optional[torch.Tensor] = None,
        fine_class: bool = False,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = input_ids != 0
        mi = False
        # if len(input_ids.size()) == 3:
        #     mi = True
        # input_ids = input_ids.view((-1, input_ids.size(-1)))
        # attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
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

        # if fine_labels is not None:
        #     fine_rel_hidden_states = self.fine_linear_transform(raw_rel_hidden_states)

        # rel_hidden_states = self.dropout(rel_hidden_states)
        # rel_hidden_states = self.linear(rel_hidden_states)
        # rel_hidden_states = F.gelu(rel_hidden_states)
        # rel_hidden_states = self.layer_normalization(rel_hidden_states)
        # rel_hidden_states = self.dropout(rel_hidden_states)

        supcon_hidden_states = self.supcon_head(raw_rel_hidden_states)


        loss = None
        # logits = None
        if fine_class:
            logits = self.fine_classifier(rel_hidden_states)
        else:
            logits = self.re_classifier(rel_hidden_states)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return REOutput(
            loss=loss,
            logits=logits,
            hidden_states=rel_hidden_states,
            supcon_hidden_states=supcon_hidden_states,
        )
