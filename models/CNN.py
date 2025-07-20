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
