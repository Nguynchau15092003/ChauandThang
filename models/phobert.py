import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class PhoBERTClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bert = bert

        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
        self.dep_type = DEP_type(opt.dep_dim)

        self.dropout = nn.Dropout(opt.bert_dropout)

        hidden_dim = 2 * opt.bert_dim  # sau pooling: mean + max
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, opt.bert_dim),
            nn.ReLU(),
            nn.Dropout(opt.linear_dropout),
        )
        self.linear = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
     text_bert_indices, bert_segments_ids, attention_mask, deprel, asp_start, asp_end, src_mask, aspect_mask, short_mask, syn_dep_adj = inputs

     bert_output = self.bert(input_ids=text_bert_indices, attention_mask=attention_mask)
     last_hidden_state = bert_output.last_hidden_state  # (B, L, H)

    # Aspect pooling (mean)
     aspect_mask_exp = aspect_mask.unsqueeze(-1).float()  # (B, L, 1)
     aspect_len = aspect_mask_exp.sum(dim=1).clamp(min=1e-8)  # (B, 1)
     aspect_rep = (last_hidden_state * aspect_mask_exp).sum(dim=1) / aspect_len  # (B, H)

     logits = self.linear(self.dropout(aspect_rep))

    # Các phần khác như se_loss bạn giữ nguyên

     return logits, None


class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super().__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, dep_input, syn_dep_adj, overall_max_len, batch_size):
        query = self.q(dep_input).squeeze(-1)       # (B, L)
        att_adj = F.softmax(query, dim=-1)
        att_adj = att_adj.unsqueeze(1).expand(-1, overall_max_len, -1)

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
