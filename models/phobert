import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class PhoBERTAspectClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.asp_emb = nn.Embedding(opt.asp_size, self.bert.config.hidden_size, padding_idx=0)

        hidden_size = self.bert.config.hidden_size

        # Nếu bạn muốn kết hợp CLS + aspect embedding trung bình
        self.classifier = nn.Linear(hidden_size * 2, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.input_dropout)

        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
        self.dep_type = DEP_type(opt.dep_dim)  # giữ phần dependency loss nếu cần

    def forward(self, inputs):
        # inputs = tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        # tok shape: (B, L), mã token đã được tokenizer chuẩn PhoBERT rồi
        attention_mask = (tok != 1).long()  # PhoBERT pad token = 1

        outputs = self.bert(input_ids=tok, attention_mask=attention_mask)
        # Lấy CLS vector (B, hidden)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Aspect embedding trung bình
        asp_emb = self.asp_emb(asp)  # (B, asp_len, hidden)
        asp_avg = torch.mean(asp_emb, dim=1)  # (B, hidden)

        # Kết hợp CLS và aspect embedding
        combined = torch.cat([cls_emb, asp_avg], dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        # --------- phần loss về dependency -----------
        overall_max_len = tok.shape[1]
        batch_size = tok.shape[0]
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len]
        dep_input = self.dep_emb(deprel[:, :overall_max_len])
        adj_pred = self.dep_type(dep_input, syn_dep_adj, overall_max_len, batch_size)
        se_loss = se_loss_batched(adj_pred, deprel[:, :overall_max_len], deprel.max().item() + 1)

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