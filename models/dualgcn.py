import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # query, key, value, output
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value=None, mask=None):
        if value is None:
            value = key
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # [B,1,L]
            elif mask.dim() == 3:
                pass  # assume correct shape
            else:
                raise ValueError(f"Unexpected mask dimension {mask.shape}")

        nbatches = query.size(0)
        # Linear projections
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]  # shape [B, h, L, d_k]

        # Attention
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # [B,h,L,d_k]

        # Concat heads
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # [B,L,d_model]

        # Final linear layer
        return self.linears[-1](x), attn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        support = self.fc(x)
        out = torch.bmm(adj.float(), support)  
        return self.dropout(F.relu(out))


class SemGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout):
        super(SemGCN, self).__init__()
        self.attn = MultiHeadAttention(heads, input_dim, dropout=dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # mask: [B,L] with 1 for valid tokens, 0 for pad
        attn_output, _ = self.attn(x, x, x, mask=mask)
        out = self.fc(attn_output)
        return self.dropout(F.relu(out))


class DualGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super(DualGCNBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(opt.pretrained_bert_name)
        self.bert_hidden = self.bert.config.hidden_size

        self.hidden_dim = opt.hidden_dim
        self.num_classes = opt.polarities_dim

        # GCN Branch: Syntactic
        self.syn_gcn = GraphConvolution(self.bert_hidden, self.hidden_dim, opt.gcn_dropout)

        # GCN Branch: Semantic (Attn-based with custom MultiHeadAttention)
        self.sem_gcn = SemGCN(self.bert_hidden, self.hidden_dim, opt.attention_heads, opt.gcn_dropout)

        # Classifier
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, inputs):
        text_bert_indices, _, attention_mask, \
        deprel, asp_start, asp_end, src_mask, \
        aspect_mask, short_mask, syn_dep_adj = inputs

        # BERT embeddings
        outputs = self.bert(input_ids=text_bert_indices, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [B, L, 768]

        # GCN: Syntactic
        syn_out = self.syn_gcn(embeddings, syn_dep_adj)

        # GCN: Semantic (with custom MultiHeadAttention)
        sem_out = self.sem_gcn(embeddings, attention_mask)

        # Aspect pooling (using src_mask)
        mask = src_mask.unsqueeze(-1).float()  # [B, L, 1]
        aspect_len = mask.sum(dim=1).clamp(min=1e-10)

        syn_pool = (syn_out * mask).sum(dim=1) / aspect_len
        sem_pool = (sem_out * mask).sum(dim=1) / aspect_len

        out = torch.cat([syn_pool, sem_pool], dim=-1)
        logits = self.classifier(out)
        return logits, None
