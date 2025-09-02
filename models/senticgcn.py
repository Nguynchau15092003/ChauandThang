import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def load_senticnet(opt):
    # Xác định đường dẫn file dựa trên ngôn ngữ opt.sentic
    if hasattr(opt, 'sentic') and opt.sentic == 'vi':
        path = './Sentic/senticnet_vi/senticnet_vi.txt'
        print("Using Vietnamese SenticNet")
    else:
        path = './Sentic/senticnet/senticnet.txt'
        print("Using English SenticNet")

    sentic_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                concept = parts[0]
                try:
                    polarity = float(parts[1])
                    sentic_dict[concept] = polarity
                except ValueError:
                    continue

    return sentic_dict


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, device='cpu'):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False).to(device)
    return h0, c0


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        # Linear layers for query, key, value
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, query))]

        # 2) Apply attention on all the projected vectors in batch.
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        # 3) 'attn' shape: (nbatches, h, seq_len_q, seq_len_k)
        # For adjacency matrix, sum attention weights over heads
        attn_sum = attn.sum(dim=1)  # Sum over heads

        return attn_sum


class SenticGCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(SenticGCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_dim = opt.embed_dim + opt.post_dim + opt.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        self.polarity_emb_dim = getattr(opt, 'polarity_emb_dim', 50)
        self.in_dim += self.polarity_emb_dim

        # RNN Layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers,
                           batch_first=True, dropout=opt.rnn_dropout, bidirectional=opt.bidirect)

        if opt.bidirect:
            self.rnn_out_dim = opt.rnn_hidden * 2
        else:
            self.rnn_out_dim = opt.rnn_hidden

        # Dropouts
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # GCN layers for dependency graph
        self.W_dep = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.rnn_out_dim if layer == 0 else self.mem_dim
            self.W_dep.append(nn.Linear(input_dim, self.mem_dim))

        # GCN layers for semantic graph (from attention)
        self.W_sen = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.rnn_out_dim if layer == 0 else self.mem_dim
            self.W_sen.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        nn.init.xavier_uniform_(self.affine1)
        nn.init.xavier_uniform_(self.affine2)

        self.attention_heads = opt.attention_heads
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim*2)

    def forward(self, adj_dep, inputs, polarity_feats):
        tok, asp, pos, head, deprel, post, mask, l, _ = inputs
        batch_size, seq_len = tok.size()

        # Embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs.append(self.pos_emb(pos))
        if self.opt.post_dim > 0:
            embs.append(self.post_emb(post))
        embs.append(polarity_feats)  # polarity vector already embedded

        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # RNN layer
        self.rnn.flatten_parameters()
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, batch_size))

        # Prepare adjacency matrices
        denom_dep = adj_dep.sum(2).unsqueeze(2) + 1  # For normalization

        src_mask = (tok != 0).unsqueeze(-2)
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        # attn_tensor shape: (batch_size, seq_len, seq_len) - average over heads done inside MultiHeadAttention

        adj_sen = attn_tensor

        # Remove self-connections and add identity
        for i in range(adj_sen.size(0)):
            adj_sen[i] = adj_sen[i] - torch.diag(torch.diag(adj_sen[i]))
            adj_sen[i] = adj_sen[i] + torch.eye(adj_sen.size(1)).to(adj_sen.device)

        adj_sen = (tok != 0).float().unsqueeze(-1) * adj_sen

        denom_sen = adj_sen.sum(2).unsqueeze(2) + 1

        outputs_dep = gcn_inputs
        outputs_sen = gcn_inputs

        for layer in range(self.layers):
            # Dependency GCN
            Ax_dep = adj_dep.bmm(outputs_dep)
            AxW_dep = self.W_dep[layer](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # Semantic GCN
            Ax_sen = adj_sen.bmm(outputs_sen)
            AxW_sen = self.W_sen[layer](Ax_sen)
            AxW_sen = AxW_sen / denom_sen
            gAxW_sen = F.relu(AxW_sen)

            # Mutual biaffine interaction
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), gAxW_sen.transpose(1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_sen, self.affine2), gAxW_dep.transpose(1, 2)), dim=-1)

            gAxW_dep = torch.bmm(A1, gAxW_sen)
            gAxW_sen = torch.bmm(A2, gAxW_dep)

            outputs_dep = self.gcn_drop(gAxW_dep) if layer < self.layers - 1 else gAxW_dep
            outputs_sen = self.gcn_drop(gAxW_sen) if layer < self.layers - 1 else gAxW_sen

        return outputs_sen, outputs_dep, adj_sen

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        device = self.emb.weight.device
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect, device=device)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.cpu().long(), batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs


import torch
import torch.nn as nn

class GCNAbsaModel(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GCNAbsaModel, self).__init__()
        self.opt = opt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # word embedding
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True).to(device)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0).to(device) if opt.pos_dim > 0 else None
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0).to(device) if opt.post_dim > 0 else None

        # Load sentic dict
        self.sentic_dict = load_senticnet(opt) if hasattr(opt, 'sentic_path') and opt.sentic_path else {}

        # Tạo polarity embedding vector tensor: size (vocab_size, 1)
        polarity_tensor = torch.zeros(opt.vocab_size, 1, dtype=torch.float)
        for word, idx in opt.word2idx.items():
            polarity = self.sentic_dict.get(word.lower(), 0.0)
            polarity_tensor[idx, 0] = polarity

        # Embedding polarity, freeze vì chỉ lookup
        self.polarity_embedding = nn.Embedding.from_pretrained(polarity_tensor, freeze=True).to(device)

        self.polarity_emb_dim = getattr(opt, 'polarity_emb_dim', 50)
        self.polarity_linear = nn.Linear(1, self.polarity_emb_dim).to(device)

        embeddings = (self.emb, self.pos_emb, self.post_emb)
        self.gcn = SenticGCN(opt, embeddings, opt.hidden_dim, opt.num_layers).to(device)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj = inputs
        device = tok.device

        batch_size, seq_len = tok.size()
        maxlen = max(l.data)
        mask = mask[:, :maxlen]

        # Lấy polarity embedding theo index tok
        polarity_feats = self.polarity_embedding(tok)  # shape (batch_size, seq_len, 1)
        polarity_feats = self.polarity_linear(polarity_feats)  # embed dim -> 50

        h1, h2, adj_ag = self.gcn(adj, inputs, polarity_feats=polarity_feats)

        asp_wn = mask.sum(dim=1).unsqueeze(-1)
        mask_ = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)
        outputs1 = (h1 * mask_).sum(dim=1) / asp_wn
        outputs2 = (h2 * mask_).sum(dim=1) / asp_wn

        return outputs1, outputs2, adj_ag


class SenticGCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SenticGCNClassifier, self).__init__()
        self.opt = opt
        in_dim = opt.hidden_dim
        self.gcn_model = GCNAbsaModel(embedding_matrix=embedding_matrix, opt=opt)
        self.classifier = nn.Linear(in_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, adj_ag = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_outputs)

        # Penalty loss for orthogonality on adj_ag (semantic graph adjacency)
        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).to(adj_ag.device)
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag @ adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] = ortho[i] - torch.diag(torch.diag(ortho[i]))
            ortho[i] = ortho[i] + identity[i]

        penal = (ortho - identity).pow(2).sum()
        return logits, penal