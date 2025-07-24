import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        embed_dim = embedding_matrix.shape[1]

        # Word embedding
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        # Dependency relation embedding (thêm)
        self.dep_emb = nn.Embedding(opt.dep_size, embed_dim, padding_idx=0)

        self.dropout = nn.Dropout(opt.input_dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=10,
            dim_feedforward=512,
            dropout=opt.input_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention pooling
        self.attention_weights = nn.Linear(embed_dim, 1)

        # Final classification layer
        self.classifier = nn.Linear(embed_dim, opt.polarities_dim)

        # Dependency relation attention
        self.dep_type = DEP_type(embed_dim)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        emb = self.embedding(tok)  # (B, L, D)
        emb = self.dropout(emb)

        # Encoding
        encoded = self.encoder(emb)  # (B, L, D)

        # Attention pooling
        att_score = self.attention_weights(torch.tanh(encoded))  # (B, L, 1)
        att_weights = torch.softmax(att_score, dim=1)            # (B, L, 1)
        rep = torch.sum(att_weights * encoded, dim=1)            # (B, D)

        # Classification
        logits = self.classifier(rep)                            # (B, num_classes)

        # --- Structure-enhanced loss ---
        overall_max_len = tok.shape[1]
        batch_size = tok.shape[0]
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len]

        # Use dep embedding for dependency prediction
        dep_input = self.dep_emb(deprel[:, :overall_max_len])    # (B, L, D)
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