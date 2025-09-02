import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_senticnet(opt):
    # như bạn đã viết, load sentic_dict
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
                except:
                    continue
    return sentic_dict

def build_dependency_adj(seq_len, head):
    # head: tensor (L,), head indices cho từng token
    D = np.eye(seq_len)
    for dep_idx, head_idx in enumerate(head):
        if head_idx >= 0 and head_idx < seq_len and dep_idx < seq_len:
            D[head_idx, dep_idx] = 1
            D[dep_idx, head_idx] = 1
    return D

def build_sentic_matrix(words, sentic_dict):
    seq_len = len(words)
    S = np.zeros((seq_len, seq_len))
    polarities = [sentic_dict.get(w.lower(), 0.0) for w in words]
    for i in range(seq_len):
        for j in range(seq_len):
            if polarities[i] != 0 and polarities[j] != 0:
                S[i, j] = polarities[i] + polarities[j]
    return S

def build_aspect_matrix(seq_len, aspect_indices):
    T = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if i in aspect_indices or j in aspect_indices:
                T[i, j] = 1
    return T

def normalize_adj(A):
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(D) + 1e-8))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, H, A_hat):
        support = self.linear(H)
        out = torch.bmm(A_hat, support)
        return F.relu(out)

class AspectAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)

    def forward(self, H, aspect_mask):
        attn_scores = self.w(H).squeeze(-1)
        attn_scores = attn_scores.masked_fill(aspect_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.unsqueeze(-1)
        out = torch.sum(H * attn_weights, dim=1)
        return out

class SenticGCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt, sentic_dict=None):
        super().__init__()
        self.opt = opt
        self.sentics = sentic_dict if sentic_dict is not None else load_senticnet(opt)

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
        lstm_out_dim = opt.rnn_hidden * 2 if opt.bidirect else opt.rnn_hidden

        self.gcn = GCNLayer(lstm_out_dim, lstm_out_dim)
        self.attention = AspectAttention(lstm_out_dim)
        self.classifier = nn.Linear(lstm_out_dim, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.input_dropout)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs
        batch_size, seq_len = tok.size()
        device = tok.device

        # Convert tok and asp embeddings
        word_emb = self.emb(tok)    # (B, L, D)
        asp_emb = self.emb(asp)     # (B, asp_len, D)
        asp_avg = torch.mean(asp_emb, dim=1, keepdim=True)  # (B, 1, D)
        asp_repeated = asp_avg.expand(-1, seq_len, -1)      # (B, L, D)

        emb = torch.cat([word_emb, asp_repeated], dim=2)   # (B, L, 2D)
        emb = self.dropout(emb)

        # BiLSTM
        lstm_out, _ = self.bilstm(emb)  # (B, L, hidden*2)

        A_batch = []
        aspect_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

        for i in range(batch_size):
            # Lấy raw words từ tok (giả sử bạn có vocab để convert idx->word)
            # Nếu không có, bạn cần truyền words raw ngoài
            # Ở đây giả sử bạn có biến opt.idx2word dict idx->word
            words_i = [self.opt.idx2word[idx.item()] for idx in tok[i]]

            # Tạo ma trận dependency từ head vector cho từng câu
            head_i = head[i].cpu().numpy()
            D = build_dependency_adj(seq_len, head_i)

            # Lấy aspect indices từ asp tokens bằng cách tìm idx của asp tokens trong câu tok
            asp_tokens_i = asp[i].cpu().tolist()
            aspect_indices = []
            for idx_token in asp_tokens_i:
                if idx_token in tok[i]:
                    aspect_indices += [j for j, v in enumerate(tok[i].cpu().tolist()) if v == idx_token]
            aspect_indices = list(set(aspect_indices))

            # Tạo ma trận sentic
            S = build_sentic_matrix(words_i, self.sentics)
            # Ma trận aspect
            T = build_aspect_matrix(seq_len, aspect_indices)

            A = D * (S + T + 1)
            A_hat = normalize_adj(A)

            A_batch.append(torch.tensor(A_hat, dtype=torch.float, device=device))

            # aspect_mask cho attention
            for idx in aspect_indices:
                if 0 <= idx < seq_len:
                    aspect_mask[i, idx] = True

        A_hat_batch = torch.stack(A_batch)  # (B, L, L)

        gcn_out = self.gcn(lstm_out, A_hat_batch)  # (B, L, hidden)

        attn_out = self.attention(gcn_out, aspect_mask)  # (B, hidden)

        logits = self.classifier(attn_out)

        return logits
