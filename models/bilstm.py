import torch
import torch.nn as nn
import torch.nn.functional as F

class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super(DEP_type, self).__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, input, syn_dep_adj, overall_max_len, batch_size):
        """
        input: (batch, seq_len, att_dim)
        syn_dep_adj: (batch, seq_len, seq_len) - chỉ số head token (long)
        """
        query = self.q(input).squeeze(-1)  # (batch, seq_len)
        att_adj = F.softmax(query, dim=-1)  # (batch, seq_len)
        att_adj = att_adj.unsqueeze(1).expand(-1, overall_max_len, -1)  # (batch, seq_len, seq_len)
        att_adj = torch.gather(att_adj, 2, syn_dep_adj)  # (batch, seq_len, seq_len)
        att_adj = att_adj.masked_fill(syn_dep_adj == 0, 0.0)
        return att_adj
class BILSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt

        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        self.asp_emb = nn.Embedding(opt.asp_size, opt.asp_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0)
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0)
        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
        self.head_emb = nn.Embedding(opt.max_seq_len, opt.head_dim, padding_idx=0)

        input_size = (embedding_matrix.shape[1] +
                      opt.asp_dim +
                      opt.pos_dim +
                      opt.post_dim +
                      opt.dep_dim +
                      opt.head_dim)

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

        word_emb = self.emb(tok)
        asp_emb = self.asp_emb(asp)
        pos_emb = self.pos_emb(pos)
        post_emb = self.post_emb(post)
        dep_emb = self.dep_emb(deprel)
        head_emb = self.head_emb(head)

        emb = torch.cat([word_emb, asp_emb, pos_emb, post_emb, dep_emb, head_emb], dim=2)
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

        # Xử lý syn_dep_adj
        overall_max_len = tok.shape[1]
        batch_size = tok.shape[0]
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len].long()

        dep_emb_tensor = self.dep_emb(deprel[:, :overall_max_len])

        adj_pred = self.dep_type(dep_emb_tensor, syn_dep_adj, overall_max_len, batch_size)

        # Sử dụng mask để tính se_loss trên token thật
        se_loss = se_loss_batched(adj_pred, deprel[:, :overall_max_len], deprel.max().item() + 1, mask=mask[:, :overall_max_len])

        # Nếu muốn, bạn có thể dùng short_mask để weight loss hoặc attention

        return logits, se_loss

def se_loss_batched(adj_pred, deprel_gold, num_relations, mask=None):
    batch, seq_len, _ = adj_pred.size()

    adj_flat = adj_pred.view(-1, seq_len)
    rel_flat = deprel_gold.view(-1)

    if mask is not None:
        mask_flat = mask.view(-1)
        adj_flat = adj_flat[mask_flat.bool()]
        rel_flat = rel_flat[mask_flat.bool()]
    else:
        mask_flat = (rel_flat != 0)
        adj_flat = adj_flat[mask_flat]
        rel_flat = rel_flat[mask_flat]

    if rel_flat.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(adj_pred.device)

    logits = torch.log(adj_flat + 1e-9)
    se_loss = F.nll_loss(logits, rel_flat, reduction='mean', ignore_index=0)

    return se_loss

