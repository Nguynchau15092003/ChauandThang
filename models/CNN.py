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

        self.embed_dim = embedding_matrix.shape[1]
        self.num_filters = opt.num_filters
        self.kernel_sizes = [int(k) for k in opt.kernel_sizes.split(',')]
        self.dropout = nn.Dropout(opt.input_dropout)

        # CNN layer
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embed_dim * 2 + opt.post_dim + opt.dep_dim,
                out_channels=self.num_filters,
                kernel_size=k
            )
            for k in self.kernel_sizes
        ])

        self.fc = nn.Linear(self.num_filters * len(self.kernel_sizes), opt.polarities_dim)
        self.dep_type = DEP_type(opt.dep_dim)  # như trong GCN model

    def forward(self, inputs):
     tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

     emb = self.embedding(tok)
     emb = self.dropout(emb)

     encoded = self.encoder(emb)  # (batch, seq_len, dim)

     att_score = self.attention_weights(torch.tanh(encoded))  # (batch, seq_len, 1)
     att_weights = torch.softmax(att_score, dim=1)
     rep = torch.sum(att_weights * encoded, dim=1)  # (batch, dim)

     logits = self.classifier(rep)  # (batch, num_classes)

     # Tính attention dựa trên embedding nhãn dependency giống CNN
     dep_embeds = self.dep_emb.weight.unsqueeze(0).expand(encoded.size(0), -1, -1)  # (batch, num_dep_labels, dep_dim)
     # Hoặc trực tiếp lấy dep_emb.weight và truyền vào dep_type nếu thiết kế như CNN
     att_adj = self.dep_type_module(dep_embeds, syn_dep_adj, encoded.size(1), encoded.size(0))

     se_loss = se_loss_batched(att_adj, deprel[:, :syn_dep_adj.shape[1]], deprel.max().item() + 1)

     return logits, se_loss

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