import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCEWithQueue(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, query, keys, queue, *args, **kwargs):
        target = torch.LongTensor([i for i in range(query.shape[0])]).to(query.device)

        sim_matrix_pos = self.cos(query.unsqueeze(1), keys.unsqueeze(0))
        sim_matrix_neg = self.cos(query.unsqueeze(1), queue.unsqueeze(0))

        sim_matrix = torch.cat((sim_matrix_pos, sim_matrix_neg), dim=1) / self.temp

        loss = self.loss_fct(sim_matrix, target)
        return loss


class SupInfoNCE(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, query, keys, queue, query_labels, queue_labels):
        device = query.device

        sim_matrix_pos = self.cos(query.unsqueeze(1), keys.unsqueeze(0))
        sim_matrix_neg = self.cos(query.unsqueeze(1), queue.unsqueeze(0))

        logits = torch.cat((sim_matrix_pos, sim_matrix_neg), dim=1).to(device) / self.temp
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

        inv_diagonal = ~torch.eye(query_labels.size(0), dtype=torch.bool, device=device)
        inv_diagonal = torch.cat([inv_diagonal, torch.ones((query_labels.size(0), queue_labels.size(0)), dtype=torch.bool).to(device)], dim=1)

        target_labels = torch.cat([query_labels, queue_labels], dim=0)
        positive_mask = torch.eq(query_labels.unsqueeze(1).repeat(1, target_labels.size(0)), target_labels)
        positive_mask = positive_mask * inv_diagonal

        alignment = logits
        uniformity = torch.exp(logits) * inv_diagonal
        uniformity = uniformity * positive_mask + (uniformity * (~positive_mask) * inv_diagonal).sum(1, keepdim=True)
        uniformity = torch.log(uniformity + 1e-6)

        log_prob = alignment - uniformity

        log_prob = (positive_mask * log_prob).sum(1, keepdim=True) / \
                    torch.max(positive_mask.sum(1, keepdim=True), torch.ones_like(positive_mask.sum(1, keepdim=True)))

        loss = -log_prob
        loss = loss.mean()

        return loss


class SupConWithQueue(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    # def forward(self, query, keys, queue, query_labels, queue_labels):
    #     sim_matrix_pos = self.cos(query.unsqueeze(1), keys.unsqueeze(0))
    #     sim_matrix_neg = self.cos(query.unsqueeze(1), queue.unsqueeze(0))
    #
    #     sim_matrix = torch.cat((sim_matrix_pos, sim_matrix_neg), dim=1).to(query.device) / self.temp
    #     exp_sim_matrix = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach())
    #
    #     label_mask1 = torch.eq(query_labels.unsqueeze(1).repeat(1, query_labels.size(0)), query_labels)
    #     label_mask1 = label_mask1 & ~torch.eye(query_labels.size(0), dtype=torch.bool, device=label_mask1.device)
    #     label_mask2 = torch.eq(query_labels.unsqueeze(1).repeat(1, queue_labels.size(0)), queue_labels)
    #     label_mask = torch.cat([label_mask1, label_mask2], dim=1)
    #
    #     # target_labels = torch.cat([query_labels, queue_labels], dim=0)
    #     # label_mask = torch.eq(query_labels.unsqueeze(1).repeat(1, target_labels.size(0)), target_labels).to(query.device)
    #     label_cnt = torch.sum(label_mask, dim=1)
    #
    #     log_prob = -torch.log(exp_sim_matrix / exp_sim_matrix.sum(dim=1, keepdim=True))
    #     loss = torch.sum(log_prob * label_mask, dim=1) / label_cnt
    #     loss = torch.mean(loss)
    #
    #     return loss

    def forward(self, query, keys, queue, query_labels, queue_labels):
        device = query.device

        targets = torch.cat([keys, queue], dim=0)
        target_labels = torch.cat([query_labels, queue_labels], dim=0)
        logits = torch.div(self.cos(query.unsqueeze(1), targets.unsqueeze(0)), self.temp)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

        inv_diagonal = ~torch.eye(query_labels.size(0), dtype=torch.bool, device=device)
        inv_diagonal = torch.cat(
            [inv_diagonal, torch.ones((query_labels.size(0), queue_labels.size(0)), dtype=torch.bool).to(device)],
            dim=1)

        positive_mask = torch.eq(query_labels.unsqueeze(1).repeat(1, target_labels.size(0)), target_labels)
        positive_mask = positive_mask * inv_diagonal

        exp_logits = torch.exp(logits) * inv_diagonal
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        label_cnt = torch.sum(positive_mask, dim=1)

        loss = -torch.sum(log_prob * positive_mask, dim=1) / label_cnt
        loss = torch.mean(loss)

        return loss


class SupConWithSelf(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, query, keys, queue, query_labels, queue_labels):
        targets = torch.cat([keys, queue], dim=0)
        target_labels = torch.cat([query_labels, queue_labels], dim=0)
        logits = torch.div(self.cos(query.unsqueeze(1), targets.unsqueeze(0)), self.temp)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

        positive_mask = torch.eq(query_labels.unsqueeze(1).repeat(1, target_labels.size(0)), target_labels)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        label_cnt = torch.sum(positive_mask, dim=1)

        loss = -torch.sum(log_prob * positive_mask, dim=1) / label_cnt
        loss = torch.mean(loss)

        return loss

