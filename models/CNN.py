import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CNNClassifier, self).__init__()
        self.opt = opt

        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0)
        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
        self.asp_emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        self.embed_dim = embedding_matrix.shape[1]
        self.kernel_sizes = [int(k) for k in opt.kernel_sizes.split(',')]
        self.num_filters = opt.num_filters
        self.dropout = nn.Dropout(opt.input_dropout)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embed_dim * 2 + opt.post_dim + opt.dep_dim,
                out_channels=self.num_filters,
                kernel_size=k,
                padding=k // 2  # giữ nguyên độ dài
            )
            for k in self.kernel_sizes
        ])

        self.fc = nn.Linear(self.num_filters * len(self.kernel_sizes), opt.polarities_dim)
        self.dep_type = DEP_type(opt.dep_dim)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs
        batch_size, seq_len = tok.shape

        word_emb = self.embed(tok)
        asp_emb = self.asp_emb(asp)
        asp_avg = torch.mean(asp_emb, dim=1, keepdim=True)
        asp_repeated = asp_avg.expand(-1, seq_len, -1)

        post_emb = self.post_emb(post)
        dep_emb = self.dep_emb(deprel)

        x = torch.cat([word_emb, asp_repeated, post_emb, dep_emb], dim=2)
        x = self.dropout(x).permute(0, 2, 1)  # (B, D, L)

        conv_outputs = [F.relu(conv(x)) for conv in self.convs]  # (B, F, L)

        # Masked max-pooling theo `l`
        pooled_outputs = []
        for conv_out in conv_outputs:
            # Tạo mask: (B, 1, L)
            mask_len = torch.arange(seq_len, device=l.device).expand(batch_size, seq_len)
            mask = mask_len < l.unsqueeze(1)
            mask = mask.unsqueeze(1)  # (B, 1, L)

            conv_out = conv_out.masked_fill(~mask, float('-inf'))
            pooled = torch.max(conv_out, dim=2)[0]  # (B, F)
            pooled_outputs.append(pooled)

        cat = torch.cat(pooled_outputs, dim=1)  # (B, F * num_kernels)
        logits = self.fc(cat)

        # ---------- se_loss ----------
        overall_max_len = tok.shape[1]
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
        query = self.q(dep_input).squeeze(-1)  # (B, L)
        att_adj = F.softmax(query, dim=-1)     # (B, L)

        # Chuyển thành (B, L, L)
        att_adj = att_adj.unsqueeze(1).expand(-1, overall_max_len, -1)  # (B, L, L)

        if syn_dep_adj.dtype == torch.bool or syn_dep_adj.max() <= 1:
            att_adj = att_adj * syn_dep_adj
        else:
            att_adj = torch.gather(att_adj, 2, syn_dep_adj)
            att_adj[syn_dep_adj == 0] = 0.

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