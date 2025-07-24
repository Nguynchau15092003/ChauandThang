import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
 def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        embed_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0)
        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
        self.asp_emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )
        self.asp_proj = nn.Linear(embedding_matrix.shape[1], embed_dim)

        self.dropout = nn.Dropout(opt.input_dropout)

        total_embed_dim = embed_dim * 2 + opt.post_dim + opt.dep_dim
        self.input_proj = nn.Linear(total_embed_dim, embed_dim)  # map về đúng d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=10,
            dim_feedforward=512,
            dropout=opt.input_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.attention_weights = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, opt.polarities_dim)
        self.dep_type = DEP_type(opt.dep_dim)


def forward(self, inputs):
    tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

    word_emb = self.embedding(tok)           # (B, L, D_word)
    post_emb = self.post_emb(post)           # (B, L, D_post)
    dep_emb = self.dep_emb(deprel)           # (B, L, D_dep)

    asp_emb = self.asp_emb(asp)               # (B, asp_len, D_asp)
    asp_avg = torch.mean(asp_emb, dim=1)     # (B, D_asp)
    asp_avg = self.asp_proj(asp_avg)         # (B, embed_dim)

    asp_repeated = asp_avg.unsqueeze(1).expand(-1, word_emb.size(1), -1)  # (B, L, embed_dim)

    emb = torch.cat([word_emb, asp_repeated, post_emb, dep_emb], dim=2)  # (B, L, total_embed_dim)
    emb = self.dropout(emb)
    emb = self.input_proj(emb)  # (B, L, embed_dim)

    # Tạo mask padding từ độ dài l (tensor batch_size)
    transformer_mask = create_src_key_padding_mask(l)  # (B, L), True ở padding

    # Kiểm tra có sample nào toàn padding không
    fully_masked_samples = transformer_mask.all(dim=1)  # (B,)
    if fully_masked_samples.any():
        idxs = torch.where(fully_masked_samples)[0].tolist()
        raise RuntimeError(f"Batch contains fully masked samples at indices {idxs}. Transformer cannot process empty sequences.")

    encoded = self.encoder(emb, src_key_padding_mask=transformer_mask)  # (B, L, embed_dim)

    att_score = torch.bmm(encoded, asp_avg.unsqueeze(2)).squeeze(2)  # (B, L)
    att_score = att_score.masked_fill(transformer_mask, float('-inf'))
    att_weights = torch.softmax(att_score, dim=1)  # (B, L)

    rep = torch.bmm(att_weights.unsqueeze(1), encoded).squeeze(1)  # (B, embed_dim)

    logits = self.classifier(rep)

    overall_max_len = tok.shape[1]
    syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len]
    dep_input = self.dep_emb(deprel[:, :overall_max_len])
    adj_pred = self.dep_type(dep_input, syn_dep_adj, overall_max_len, tok.size(0))
    se_loss = se_loss_batched(adj_pred, deprel[:, :overall_max_len], deprel.max().item() + 1)

    return logits, se_loss
def create_src_key_padding_mask(lengths):
    """
    lengths: tensor (batch_size,) chứa độ dài thực của mỗi câu
    Trả về mask bool (batch_size, max_len), True với vị trí padding
    """
    batch_size = lengths.size(0)
    max_len = lengths.max().item()

    # Tạo tensor vị trí: shape (max_len,)
    range_tensor = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, max_len)

    # Mask True ở những vị trí index >= length của câu (padding)
    mask = range_tensor >= lengths.unsqueeze(1)

    return mask  # dtype=torch.bool, shape (batch_size, max_len)
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