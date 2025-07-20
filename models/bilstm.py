import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.asp_emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb  # dùng cùng embedding cho aspect
        )

        input_size = embedding_matrix.shape[1] * 2 + opt.post_dim + opt.dep_dim

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

        self.dep_type = DEP_type(opt.dep_dim)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        word_emb = self.emb(tok)                  # (B, L, D)
        post_emb = self.post_emb(post)            # (B, L, Dp)
        dep_emb = self.dep_emb(deprel)            # (B, L, Dd)

        # Tính aspect embedding theo trung bình
        asp_emb = self.asp_emb(asp)               # (B, asp_len, D)
        asp_avg = torch.mean(asp_emb, dim=1, keepdim=True)  # (B, 1, D)
        asp_repeated = asp_avg.expand(-1, word_emb.size(1), -1)  # (B, L, D)

        emb = torch.cat([word_emb, asp_repeated, post_emb, dep_emb], dim=2)
        emb = self.dropout(emb)

        seq_lens = l.cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, seq_lens, batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed_emb)

        if self.opt.bidirect:
            last_fw = h_n[-2]
            last_bw = h_n[-1]
            final_feat = torch.cat([last_fw, last_bw], dim=1)
        else:
            final_feat = h_n[-1]

        logits = self.classifier(final_feat)

        # ---------- se_loss ----------
        overall_max_len = tok.shape[1]
        batch_size = tok.shape[0]
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len]
        dep_input = self.dep_emb(deprel[:, :overall_max_len])  # (B, L, Dd)
        adj_pred = self.dep_type(dep_input, syn_dep_adj, overall_max_len, batch_size)
        se_loss = se_loss_batched(adj_pred, deprel[:, :overall_max_len], deprel.max().item() + 1)

        return logits, se_loss

class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super(DEP_type, self).__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, dep_input, syn_dep_adj, overall_max_len, batch_size):
        # dep_input: (B, L, dep_dim)
        query = self.q(dep_input).squeeze(-1)       # (B, L)
        att_adj = F.softmax(query, dim=-1)          # (B, L)

        # Mở rộng để thành ma trận attention: (B, L, L)
        att_adj = att_adj.unsqueeze(1).expand(-1, overall_max_len, -1)  # (B, L, L)

        # Gán zero cho vị trí không có dependency
        att_adj = att_adj * syn_dep_adj  # (B, L, L), giữ lại những vị trí hợp lệ

        return att_adj
def se_loss_batched(adj_pred, deprel_gold, num_relations):
    """
    adj_pred: Tensor float [batch, seq_len, seq_len], là xác suất attention giữa các token.
    deprel_gold: LongTensor [batch, seq_len], label dependency (0 là padding) cho từng token.
    num_relations: int, số lượng nhãn dependencies.
    
    Trả về: se_loss (mean cross-entropy trên các token thật, không tính padding).
    """
    batch, seq_len, _ = adj_pred.size()
    
    # Mỗi token i có một hàng probability adj_pred[:, i, :] biểu thị phân bố label cho head
    # Mỗi token có đúng một nhãn gold là deprel_gold[:, i].
    # Vì deprel_gold shape [batch, seq_len], ta flatten cả 2 chiều batch và token.
    
    adj_flat = adj_pred.view(-1, seq_len)            # [batch*seq_len, seq_len]
    rel_flat = deprel_gold.view(-1)                  # [batch*seq_len]
    
    # Lọc những token thực (rel != 0) để loại bỏ padding
    mask = (rel_flat != 0)
    adj_flat = adj_flat[mask]                        # [? , seq_len]
    rel_flat = rel_flat[mask]                        # [?]
    
    # Nếu không còn token nào, trả về 0
    if rel_flat.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(adj_pred.device)
    
    # Chúng ta cần một máy phân lớp với số class = seq_len (position trong sentence)
    # Và deprel_gold chỉ là nhãn quan hệ, không phải vị trí head.
    # Vậy ý nghĩa của se_loss trong GCN gốc là:
    # - Dùng ma trận attention syn_dep_adj dự đoán nhãn deprel cho từng cặp (i, j).
    # - Mỗi vị trí i được gán nhãn deprel tương ứng head tại j thực tế.
    # Do đó, rel_flat chứa giá trị j (head index).
    # Và adj_flat chứa probability cho mỗi j.

    # cross-entropy classification: logits = log(adj_flat + eps)
    logits = torch.log(adj_flat + 1e-9)  # giữ numerical stability
    se_loss = F.nll_loss(logits, rel_flat, reduction='mean')
    return se_loss