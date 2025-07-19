import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.si_utils import se_loss_batched


class ViewAttentionBlock(nn.Module):
    def __init__(self, view_num):
        super(ViewAttentionBlock, self).__init__()
        self.view_num = view_num
        self.fc1 = nn.Linear(view_num, 15)
        # self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, view_num)
        # initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        batch_size, num_view, num_node, feature_dim = x.size()
        weight = x.view(batch_size, self.view_num, -1).mean(dim=2)
        weight0 = weight.squeeze(-1)  # [batch_size , view_num]
        weight = torch.relu(self.fc1(weight0))
        weight = self.fc3(weight)  # residual connection
        weight = F.softmax(weight, dim=1)
        weight = weight.view(-1, self.view_num, 1, 1)
        out = x * weight
        out = out.sum(dim=1)
        return out


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MASGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(100, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, se_loss = self.gcn_model(inputs)
        logits = self.classifier(outputs1)

        return logits, se_loss


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, deprel, asp_start, asp_end, src_mask, aspect_mask, short_mask, syn_dep_adj = inputs
        h, se_loss = self.gcn(inputs)
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100)
        outputs1 = (h * aspect_mask).sum(dim=1) / asp_wn
        return outputs1, se_loss


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.attdim = 100
        self.W = nn.Linear(self.attdim, self.attdim)
        self.Wx = nn.Linear(self.attention_heads +
                            self.attdim * 2, self.attention_heads)
        self.Wxx = nn.Linear(self.bert_dim, self.attdim)
        self.Wi = nn.Linear(self.attdim, 50)
        self.aggregate_W = nn.Linear(
            self.attdim * (self.layers + 1), self.attdim)

        self.attn = MultiHeadAttention(opt.attention_heads, self.attdim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.gap_map = nn.Sequential(nn.Linear(self.attention_heads, 6), nn.Sigmoid(
        ), nn.Linear(6, 3), nn.Sigmoid(), nn.Linear(3, self.attention_heads))
        self.dep_type = DEP_type(opt.dep_dim)
        self.dep_emb = nn.Embedding(
            opt.dep_size, opt.dep_dim, padding_idx=0)  # dependency emb
        self.multi_view = ViewAttentionBlock(
            opt.attention_heads)  # multi-view attention

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, deprel, asp_start, asp_end, src_mask, aspect_mask, short_mask, syn_dep_adj = inputs
        overall_max_len = src_mask.size(1)
        src_mask = src_mask.unsqueeze(-2)
        batch = src_mask.size(0)
        len = src_mask.size()[2]

        sequence_output = self.bert(
            text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)['last_hidden_state']
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        # pooled_output = self.pooled_drop(pooled_output)

        gcn_inputs = self.Wxx(gcn_inputs)

        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100)
        aspect = (gcn_inputs * aspect_mask).sum(dim=1) / asp_wn

        attn_tensor = self.attn(gcn_inputs, gcn_inputs,
                                short_mask, aspect, src_mask)
        attn_adj_list = [attn_adj.squeeze(
            1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        multi_head_list = []
        outputs_dep = None
        adj_ag = None

        weight_adj = attn_tensor
        gcn_outputs = gcn_inputs
        layer_list = [gcn_inputs]

        syn_dep_adj = self.dep_type(self.dep_emb.weight, syn_dep_adj,
                                   overall_max_len, syn_dep_adj.shape[0]).float()
        se_loss = se_loss_batched(syn_dep_adj, deprel[:, :syn_dep_adj.shape[1]], deprel.max().item() + 1)
        syn_dep_adj = syn_dep_adj.unsqueeze(1)
        weight_adj += syn_dep_adj

        for i in range(self.layers):

            gcn_outputs = gcn_outputs.unsqueeze(1).expand(
                batch, self.attention_heads, len, self.attdim)
            Ax = torch.matmul(weight_adj, gcn_outputs)
            Ax = self.multi_view(Ax)

            Ax = self.W(Ax)
            weights_gcn_outputs = F.relu(Ax)

            gcn_outputs = weights_gcn_outputs
            layer_list.append(gcn_outputs)
            gcn_outputs = self.gcn_drop(
                gcn_outputs) if i < self.layers - 1 else gcn_outputs

            weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()
            node_outputs1 = gcn_outputs.unsqueeze(
                1).expand(batch, len, len, self.attdim)
            node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
            node = torch.cat([node_outputs1, node_outputs2], dim=-1)
            edge_n = torch.cat([weight_adj, node], dim=-1)
            edge = self.Wx(edge_n)
            edge = self.gcn_drop(edge) if i < self.layers - 1 else edge
            weight_adj = edge.permute(0, 3, 1, 2).contiguous()

        outputs = torch.cat(layer_list, dim=-1)
        # node_outputs=self.Wi(gcn_outputs)
        node_outputs = self.aggregate_W(outputs)
        node_outputs = F.relu(gcn_outputs)

        return node_outputs, None


def attention(query, key, short, aspect, weight_m, bias_m, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    batch = len(scores)
    p = weight_m.size(0)
    max = weight_m.size(1)
    weight_m = weight_m.unsqueeze(0).repeat(batch, 1, 1, 1)

    aspect_scores = torch.tanh(
        torch.add(torch.matmul(aspect, key.transpose(-2, -1)), bias_m))
    scores = torch.add(scores, aspect_scores)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = torch.add(scores, short)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, short, aspect, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)
        aspect = self.dense(aspect)
        aspect = aspect.unsqueeze(2).expand(
            batch, self.h, query.size()[2], self.d_k)
        attn = attention(query, key, short, aspect, self.weight_m,
                         self.bias_m, mask=mask, dropout=self.dropout)
        return attn


class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super(DEP_type, self).__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, input, syn_dep_adj, overall_max_len, batch_size):
        query = self.q(input).T
        att_adj = F.softmax(query, dim=-1)
        att_adj = att_adj.unsqueeze(0).repeat(batch_size, overall_max_len, 1)
        att_adj = torch.gather(att_adj, 2, syn_dep_adj)
        att_adj[syn_dep_adj == 0.] = 0.
        return att_adj
