

import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModel


# ----------- Tree structure & adj -----------
class TreeNode:
    def __init__(self, idx):
        self.idx = idx
        self.children = []

def head_to_tree(head, words, length):
    nodes = [TreeNode(i) for i in range(length)]
    root = None
    for i in range(length):
        h = head[i]
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].children.append(nodes[i])
    return root

def tree_to_adj(max_len, root, directed=True, self_loop=True):
    adj = np.zeros((max_len, max_len), dtype=np.float32)

    def dfs(node):
        for child in node.children:
            adj[node.idx][child.idx] = 1
            if not directed:
                adj[child.idx][node.idx] = 1
            dfs(child)

    if root is not None:
        dfs(root)

    if self_loop:
        for i in range(max_len):
            adj[i][i] = 1

    return adj

def inputs_to_tree_reps(head_batch, words_batch, lengths, max_len, directed=True, self_loop=True):
    batch_adj = []
    for i in range(len(lengths)):
        length = lengths[i]
        head = head_batch[i][:length].tolist()
        words = words_batch[i][:length]  # if words are needed
        tree = head_to_tree(head, words, length)
        adj = tree_to_adj(max_len, tree, directed=directed, self_loop=self_loop)
        batch_adj.append(adj[np.newaxis, :, :])  # [1, max_len, max_len]
    batch_adj = np.concatenate(batch_adj, axis=0)
    batch_adj = torch.from_numpy(batch_adj).float()
    return batch_adj.cuda() if torch.cuda.is_available() else batch_adj


# MultiHeadAttention như trong code bạn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


# GCN Layer

class GCN(nn.Module):
    def __init__(self, opt, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = opt.bert_hidden_dim
        self.attention_heads = opt.attention_heads

        # drop out
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layers
        self.W = nn.ModuleList()
        self.weight_list = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim*2)

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

        nn.init.xavier_uniform_(self.affine1)
        nn.init.xavier_uniform_(self.affine2)

    def forward(self, adj, inputs):
        # inputs: (tok_embeddings, mask, lengths)
        tok_emb, mask, lengths = inputs

        batch_size, maxlen, _ = tok_emb.size()

        src_mask = (mask != 0).unsqueeze(-2)  # [B, 1, maxlen]
        mask_ = mask.unsqueeze(-1).float()    # [B, maxlen, 1]

        denom_dep = adj.sum(2).unsqueeze(2) + 1  # degree normalization for dep adj

        attn_tensor = self.attn(tok_emb, tok_emb, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]

        adj_ag = None
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag /= self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = mask_ * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_ag = tok_emb
        outputs_dep = tok_emb

        for l in range(self.layers):
            # Dependency GCN
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # Attention GCN
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # Mutual biaffine
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)

            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)

            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, adj_ag



# DualGCNBERT main model

class DualGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.pretrained_bert_name)
        self.bert_hidden_dim = self.bert.config.hidden_size
        self.opt.bert_hidden_dim = self.bert_hidden_dim

        self.gcn = GCN(opt, mem_dim=opt.hidden_dim, num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        (text_bert_indices, bert_segments_ids, attention_mask,
         deprel, asp_start, asp_end, src_mask,
         aspect_mask, short_mask, syn_dep_adj) = inputs

        device = text_bert_indices.device

        # BERT
        bert_outputs = self.bert(
            input_ids=text_bert_indices,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state  # [B, L, H]
        batch_size, maxlen, _ = sequence_output.size()

        # Lengths from attention mask
        lengths = attention_mask.sum(dim=1).cpu()  # [batch]

        # Use mask (can be src_mask or aspect_mask or short_mask)
        mask = src_mask  # You can change this based on config

        # Dependency Tree Adjacency Matrix
        if syn_dep_adj is not None:
            adj_dep = syn_dep_adj.float().to(device)
        else:
            # NOTE: deprel = head indices, and use text_bert_indices as dummy tokens
            adj_dep = inputs_to_tree_reps(
                deprel.cpu(),
                text_bert_indices.cpu(),  # dummy tokens
                lengths,
                maxlen,
                directed=self.opt.direct,
                self_loop=self.opt.loop
            ).to(device)

        # GCN Forward
        gcn_inputs = (sequence_output, mask, lengths)
        outputs_ag, outputs_dep, adj_ag = self.gcn(adj_dep, gcn_inputs)

        # Aspect Pooling
        asp_wn = mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-10)
        outputs1 = (outputs_ag * mask.unsqueeze(-1).float()).sum(dim=1) / asp_wn
        outputs2 = (outputs_dep * mask.unsqueeze(-1).float()).sum(dim=1) / asp_wn

        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_outputs)

        # Regularization Penalty
        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).to(device).unsqueeze(0).expand_as(adj_ag)
        ortho = torch.bmm(adj_ag, adj_ag_T)

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).to(device)

        penal = None
        if self.opt.losstype == 'doubleloss':
            penal1 = torch.norm(ortho - identity) / adj_ag.size(0)
            penal2 = adj_ag.size(0) / torch.norm(adj_ag - adj_dep)
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2
        elif self.opt.losstype == 'orthogonalloss':
            penal = self.opt.alpha * (torch.norm(ortho - identity) / adj_ag.size(0))
        elif self.opt.losstype == 'differentiatedloss':
            penal = self.opt.beta * (adj_ag.size(0) / torch.norm(adj_ag - adj_dep))

        return logits, penal
