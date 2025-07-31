import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class PhoBERTspectClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bert = bert

        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
        self.dep_type = DEP_type(opt.dep_dim)

        self.dropout = nn.Dropout(opt.bert_dropout)
        self.linear = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        # unpack inputs
        text_bert_indices, bert_segments_ids, attention_mask, deprel, asp_start, asp_end, src_mask, aspect_mask, short_mask, syn_dep_adj = inputs

        bert_output = self.bert(input_ids=text_bert_indices, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state  # (B, L, H)

        # Aspect pooling (average)
        aspect_mask = aspect_mask.unsqueeze(-1).expand_as(last_hidden_state)  # (B, L, H)
        aspect_len = aspect_mask.sum(dim=1).clamp(min=1e-8)
        aspect_rep = (last_hidden_state * aspect_mask).sum(dim=1) / aspect_len  # (B, H)

        logits = self.linear(self.dropout(aspect_rep))

        # ---------- se_loss ----------
        overall_max_len = text_bert_indices.shape[1]
        batch_size = text_bert_indices.shape[0]
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len]
        dep_input = self.dep_emb(deprel[:, :overall_max_len])  # (B, L, Dd)
        adj_pred = self.dep_type(dep_input, syn_dep_adj, overall_max_len, batch_size)
        se_loss = se_loss_batched(adj_pred, deprel[:, :overall_max_len], deprel.max().item() + 1)

        return logits, se_loss


class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super(DEP_type, self).__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, dep_input, syn_dep_adj, overall_max_len, batch_size):
        query = self.q(dep_input).squeeze(-1)  # (B, L)
        att_adj = F.softmax(query, dim=-1)     # (B, L)
        att_adj = att_adj.unsqueeze(1).expand(-1, overall_max_len, -1)  # (B, L, L)

        if syn_dep_adj.dtype == torch.bool or syn_dep_adj.max() <= 1:
            att_adj = att_adj * syn_dep_adj
        else:
            att_adj = torch.gather(att_adj, 2, syn_dep_adj)
            att_adj[syn_dep_adj == 0] = 0.

        return att_adj


def se_loss_batched(adj_pred, deprel_gold, num_relations):
    batch, seq_len, _ = adj_pred.size()
    adj_flat = adj_pred.view(-1, seq_len)
    rel_flat = deprel_gold.view(-1)

    mask = (rel_flat != 0)
    adj_flat = adj_flat[mask]
    rel_flat = rel_flat[mask]

    if rel_flat.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(adj_pred.device)

    logits = torch.log(adj_flat + 1e-9)
    se_loss = F.nll_loss(logits, rel_flat, reduction='mean')
    return se_loss
