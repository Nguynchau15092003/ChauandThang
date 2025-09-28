import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        support = self.fc(x)
        out = torch.bmm(adj.float(), support)  
        return self.dropout(F.relu(out))
def build_adj_from_deprel(deprel):
    """
    deprel: [B, L] chứa head index (0 <= head < L)
    Return: [B, L, L] adjacency matrix
    """
    B, L = deprel.size()
    adj = torch.zeros(B, L, L, device=deprel.device)
    for b in range(B):
        for i in range(L):
            head = deprel[b, i]
            if 0 <= head < L:
                adj[b, i, head] = 1.0
                adj[b, head, i] = 1.0  # bidirectional (tùy bạn)
    return adj
class SemGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout):
        super(SemGCN, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=~mask.bool())
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

        # GCN Branch: Dependency
        self.dep_gcn = GraphConvolution(self.bert_hidden, self.hidden_dim, opt.gcn_dropout)

        # GCN Branch: Semantic (Attn-based)
        self.sem_gcn = SemGCN(self.bert_hidden, self.hidden_dim, opt.attention_heads, opt.gcn_dropout)

        # Classifier: input dim = hidden_dim * 3
        self.classifier = nn.Linear(self.hidden_dim * 3, self.num_classes)

    def forward(self, inputs):
        text_bert_indices, _, attention_mask, \
        deprel, asp_start, asp_end, src_mask, \
        aspect_mask, short_mask, syn_dep_adj = inputs

        outputs = self.bert(input_ids=text_bert_indices, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [B, L, 768]

        syn_out = self.syn_gcn(embeddings, syn_dep_adj)

        # Chuyển deprel sang adjacency matrix
        dep_adj = build_adj_from_deprel(deprel)
        dep_out = self.dep_gcn(embeddings, dep_adj)

        sem_out = self.sem_gcn(embeddings, attention_mask)

        mask = src_mask.unsqueeze(-1).float()  # [B, L, 1]
        aspect_len = mask.sum(dim=1).clamp(min=1e-10)

        syn_pool = (syn_out * mask).sum(dim=1) / aspect_len
        dep_pool = (dep_out * mask).sum(dim=1) / aspect_len
        sem_pool = (sem_out * mask).sum(dim=1) / aspect_len

        out = torch.cat([syn_pool, dep_pool, sem_pool], dim=-1)
        logits = self.classifier(out)
        return logits, None
