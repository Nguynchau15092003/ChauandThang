import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_senticnet(opt):
    """
    Load SenticNet polarity dictionary based on language option.
    """
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
                concept = parts[0].lower()
                try:
                    polarity = float(parts[1])
                    sentic_dict[concept] = polarity
                except ValueError:
                    continue
    return sentic_dict

def build_dependency_adj(seq_len, dep_edges):
    """
    Tạo ma trận kề dependency D: D[i,j]=1 nếu có quan hệ dependency hoặc i==j
    dep_edges: list các cặp (head, dep)
    """
    D = np.eye(seq_len, dtype=float)
    for h, d in dep_edges:
        if 0 <= h < seq_len and 0 <= d < seq_len:
            D[h, d] = 1
            D[d, h] = 1  # thường dependency là vô hướng (có thể dùng 2 chiều)
    return D

def build_sentic_matrix(words, sentic_dict):
    """
    Ma trận S[i,j] = SenticNet(wi) + SenticNet(wj) nếu cả hai có trong sentic_dict, ngược lại 0
    """
    seq_len = len(words)
    S = np.zeros((seq_len, seq_len), dtype=float)
    polarities = []
    for w in words:
        polarities.append(sentic_dict.get(w.lower(), 0.0))
    for i in range(seq_len):
        for j in range(seq_len):
            if polarities[i] != 0.0 and polarities[j] != 0.0:
                S[i, j] = polarities[i] + polarities[j]
    return S

def build_aspect_matrix(seq_len, aspect_indices):
    """
    T[i,j] = 1 nếu wi hoặc wj là aspect word, ngược lại 0
    """
    T = np.zeros((seq_len, seq_len), dtype=float)
    for i in range(seq_len):
        for j in range(seq_len):
            if i in aspect_indices or j in aspect_indices:
                T[i, j] = 1.0
    return T

def normalize_adj(A):
    """
    Chuẩn hóa ma trận A theo dạng A_hat = D^{-1/2} A D^{-1/2}
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, H, A_hat):
        """
        H: (B, L, input_dim) hidden states
        A_hat: (B, L, L) normalized adjacency matrix
        """
        support = self.linear(H)        # (B, L, output_dim)
        out = torch.bmm(A_hat, support)  # (B, L, output_dim)
        return F.relu(out)

class AspectAttention(nn.Module):
    def __init__(self, input_dim):
        super(AspectAttention, self).__init__()
        self.w = nn.Linear(input_dim, 1)

    def forward(self, H, aspect_mask):
        """
        H: (B, L, D)
        aspect_mask: (B, L) binary mask, 1 for aspect words, 0 else
        """
        attn_scores = self.w(H).squeeze(-1)   # (B, L)
        attn_scores = attn_scores.masked_fill(aspect_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, L)
        attn_weights = attn_weights.unsqueeze(-1)     # (B, L, 1)
        out = torch.sum(H * attn_weights, dim=1)      # (B, D)
        return out

class SenticGCClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SenticGCN, self).__init__()
        self.opt = opt
        self.sentics = load_senticnet(opt)  # Load sentic dict 1 lần khi khởi tạo

        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        input_size = embedding_matrix.shape[1] * 2  # word emb + aspect emb

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
        """
        inputs:
            tok: LongTensor (B, L) - token indices
            asp: LongTensor (B, asp_len) - aspect token indices
            words: list of list of str - raw words in sentences (batch)
            dep_edges_list: list of list of tuples (head, dep) - dependency edges for each sample in batch
            aspect_indices_list: list of list of int - indices of aspect words in each sample
        """
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs
        batch_size, seq_len = tok.size()
        device = tok.device

        word_emb = self.emb(tok)    # (B, L, D)
        asp_emb = self.emb(asp)     # (B, asp_len, D)
        asp_avg = torch.mean(asp_emb, dim=1, keepdim=True)  # (B, 1, D)
        asp_repeated = asp_avg.expand(-1, seq_len, -1)      # (B, L, D)

        emb = torch.cat([word_emb, asp_repeated], dim=2)   # (B, L, 2D)
        emb = self.dropout(emb)

        # BiLSTM
        lstm_out, _ = self.bilstm(emb)  # (B, L, hidden*2 or hidden)

        # Tạo ma trận kề batch_size * L * L
        A_batch = []
        for i in range(batch_size):
            words_i = words[i]
            dep_edges = dep_edges_list[i]
            aspect_indices = aspect_indices_list[i]

            D = build_dependency_adj(seq_len, dep_edges)
            S = build_sentic_matrix(words_i, self.sentics)
            T = build_aspect_matrix(seq_len, aspect_indices)
            A = D * (S + T + 1)
            A_hat = normalize_adj(A)

            A_batch.append(torch.tensor(A_hat, dtype=torch.float, device=device))

        A_hat_batch = torch.stack(A_batch)  # (B, L, L)

        # GCN
        gcn_out = self.gcn(lstm_out, A_hat_batch)  # (B, L, hidden)

        # Aspect mask cho attention
        aspect_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for i, indices in enumerate(aspect_indices_list):
            aspect_mask[i, indices] = True

        # Attention pooling
        attn_out = self.attention(gcn_out, aspect_mask)  # (B, hidden)

        logits = self.classifier(attn_out)  # (B, polarities_dim)

        return logits
