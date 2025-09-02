
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sentic_loader import load_senticnet  # ✅ Dùng hàm đã cache

# Các hàm build matrix không đổi

def build_dependency_adj(seq_len, dep_edges):
    D = np.eye(seq_len, dtype=float)
    for h, d in dep_edges:
        if 0 <= h < seq_len and 0 <= d < seq_len:
            D[h, d] = 1
            D[d, h] = 1
    return D

def build_sentic_matrix(words, sentic_dict):
    seq_len = len(words)
    S = np.zeros((seq_len, seq_len), dtype=float)
    polarities = [sentic_dict.get(w.lower(), 0.0) for w in words]
    for i in range(seq_len):
        for j in range(seq_len):
            if polarities[i] != 0.0 and polarities[j] != 0.0:
                S[i, j] = polarities[i] + polarities[j]
    return S

def build_aspect_matrix(seq_len, aspect_indices):
    T = np.zeros((seq_len, seq_len), dtype=float)
    for i in range(seq_len):
        for j in range(seq_len):
            if i in aspect_indices or j in aspect_indices:
                T[i, j] = 1.0
    return T

def normalize_adj(A):
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat

# GCN layer
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, H, A_hat):
        support = self.linear(H)
        out = torch.bmm(A_hat, support)
        return F.relu(out)

# Attention layer
class AspectAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)

    def forward(self, H, aspect_mask):
        attn_scores = self.w(H).squeeze(-1)
        attn_scores = attn_scores.masked_fill(aspect_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        return torch.sum(H * attn_weights, dim=1)

# ✅ MODEL chính
class SenticGCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt

        # idx2word để map từ id -> word
        self.opt.idx2word = {idx: word for word, idx in self.opt.word2idx.items()}

        # ✅ Tải SenticNet một lần
        self.sentics = load_senticnet(opt)

        # Embedding
        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        input_size = embedding_matrix.shape[1] * 2

        self.bilstm = nn.LSTM(
            input_size,
            opt.rnn_hidden,
            num_layers=opt.rnn_layers,
            batch_first=True,
            dropout=opt.rnn_dropout if opt.rnn_layers > 1 else 0,
            bidirectional=opt.bidirect
        )

        lstm_output_dim = opt.rnn_hidden * 2 if opt.bidirect else opt.rnn_hidden

        self.gcn = GCNLayer(lstm_output_dim, lstm_output_dim)
        self.attention = AspectAttention(lstm_output_dim)
        self.classifier = nn.Linear(lstm_output_dim, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.input_dropout)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs
        device = tok.device
        batch_size, seq_len = tok.size()

        words_batch = []
        for i in range(batch_size):
            words = [self.opt.idx2word.get(idx, '<unk>') for idx in tok[i].tolist()]
            words_batch.append(words)

        dep_edges_list = []
        for i in range(batch_size):
            dep_edges = []
            heads = head[i].tolist()
            for dep_idx, head_idx in enumerate(heads):
                if head_idx > 0:
                    dep_edges.append((head_idx - 1, dep_idx))
            dep_edges_list.append(dep_edges)

        aspect_indices_list = []
        for i in range(batch_size):
            asp_indices = [idx for idx in asp[i].tolist() if 0 <= idx < seq_len]
            aspect_indices_list.append(asp_indices)

        # Embedding
        word_emb = self.emb(tok)
        asp_emb = self.emb(asp)
        asp_avg = torch.mean(asp_emb, dim=1, keepdim=True)
        asp_repeated = asp_avg.expand(-1, seq_len, -1)
        emb = torch.cat([word_emb, asp_repeated], dim=2)
        emb = self.dropout(emb)

        lstm_out, _ = self.bilstm(emb)

        A_batch = []
        for i in range(batch_size):
            seq_len_i = len(words_batch[i])
            D = build_dependency_adj(seq_len_i, dep_edges_list[i])
            S = build_sentic_matrix(words_batch[i], self.sentics)
            T = build_aspect_matrix(seq_len_i, aspect_indices_list[i])
            A = D * (S + T + 1)
            A_hat = normalize_adj(A)

            if seq_len_i < seq_len:
                pad_size = seq_len - seq_len_i
                A_hat = np.pad(A_hat, ((0, pad_size), (0, pad_size)), mode='constant')

            A_batch.append(torch.tensor(A_hat, dtype=torch.float, device=device))

        A_hat_batch = torch.stack(A_batch)

        gcn_out = self.gcn(lstm_out, A_hat_batch)

        aspect_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for i, indices in enumerate(aspect_indices_list):
            for idx in indices:
                if idx < seq_len:
                    aspect_mask[i, idx] = True

        attn_out = self.attention(gcn_out, aspect_mask)

        logits = self.classifier(attn_out)
        return logits, None