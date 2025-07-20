import torch
import torch.nn as nn
import torch.nn.functional as F

class BILSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt

        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0)
        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)

        input_size = embedding_matrix.shape[1] + opt.post_dim + opt.dep_dim

        self.lstm = nn.LSTM(
            input_size,
            opt.rnn_hidden,
            num_layers=opt.rnn_layers,
            batch_first=True,
            dropout=opt.rnn_dropout if opt.rnn_layers > 1 else 0,
            bidirectional=opt.bidirect
        )

        lstm_output_dim = opt.rnn_hidden * 2 if opt.bidirect else opt.rnn_hidden
        self.dropout = nn.Dropout(opt.input_dropout)
        self.classifier = nn.Linear(lstm_output_dim, opt.polarities_dim)

        self.dep_type = DEP_type(opt.dep_dim)  # module dự đoán dependency attention

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        word_emb = self.emb(tok)                    # (batch, seq_len, emb_dim)
        post_emb = self.post_emb(post)              # (batch, seq_len, post_dim)
        dep_emb = self.dep_emb(deprel)              # (batch, seq_len, dep_dim)

        emb = torch.cat([word_emb, post_emb, dep_emb], dim=2)  # (batch, seq_len, input_size)
        emb = self.dropout(emb)

        seq_lens = l.cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, seq_lens, batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed_emb)

        # Lấy hidden cuối cùng
        if self.opt.bidirect:
            last_fw = h_n[-2]
            last_bw = h_n[-1]
            final_feat = torch.cat([last_fw, last_bw], dim=1)
        else:
            final_feat = h_n[-1]

        logits = self.classifier(final_feat)  # (batch, num_classes)

        # ---------- se_loss ----------
        overall_max_len = tok.shape[1]
        batch_size = tok.shape[0]
        # Cắt syn_dep_adj đúng kích thước
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len].long()  # long để làm index gather

        # Lấy embedding dependency cho từng token (batch, seq_len, dep_dim)
        dep_emb_tensor = self.dep_emb(deprel[:, :overall_max_len])

        # Tính ma trận attention dự đoán
        adj_pred = self.dep_type(dep_emb_tensor, syn_dep_adj, overall_max_len, batch_size)  # (batch, seq_len, seq_len)

        # Tính loss dependency
        se_loss = se_loss_batched(adj_pred, deprel[:, :overall_max_len], deprel.max().item() + 1)

        return logits, se_loss


class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super(DEP_type, self).__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, input, syn_dep_adj, overall_max_len, batch_size):
        """
        input: (batch, seq_len, att_dim)
        syn_dep_adj: (batch, seq_len, seq_len) - chỉ số head token (long)
        """
        # Tính query score attention cho từng token
        query = self.q(input).squeeze(-1)  # (batch, seq_len)

        # Softmax theo chiều seq_len
        att_adj = F.softmax(query, dim=-1)  # (batch, seq_len)

        # Mở rộng để thành ma trận attention [batch, seq_len, seq_len]
        att_adj = att_adj.unsqueeze(1).expand(-1, overall_max_len, -1)  # (batch, seq_len, seq_len)

        # Lấy attention theo cấu trúc syn_dep_adj (index)
        att_adj = torch.gather(att_adj, 2, syn_dep_adj)  # (batch, seq_len, seq_len)

        # Mask padding token (syn_dep_adj == 0)
        att_adj = att_adj.masked_fill(syn_dep_adj == 0, 0.0)

        return att_adj


def se_loss_batched(adj_pred, deprel_gold, num_relations):
    """
    adj_pred: Tensor float [batch, seq_len, seq_len] - attention probabilities
    deprel_gold: LongTensor [batch, seq_len] - label dependency (0 is padding)
    num_relations: int - số lượng nhãn dependencies (không dùng ở đây)

    Trả về: se_loss (mean cross-entropy trên token thật)
    """
    batch, seq_len, _ = adj_pred.size()

    adj_flat = adj_pred.view(-1, seq_len)    # [batch*seq_len, seq_len]
    rel_flat = deprel_gold.view(-1)          # [batch*seq_len]

    mask = (rel_flat != 0)
    adj_flat = adj_flat[mask]                 # [? , seq_len]
    rel_flat = rel_flat[mask]                 # [?]

    if rel_flat.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(adj_pred.device)

    logits = torch.log(adj_flat + 1e-9)  # tránh log(0)
    # ignore_index=0 để bỏ padding
    se_loss = F.nll_loss(logits, rel_flat, reduction='mean', ignore_index=0)

    return se_loss
