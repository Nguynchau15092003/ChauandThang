import torch
import torch.nn as nn
import torch.nn.functional as F


class ATAELSTMClassifier(nn.Module):
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
            freeze=opt.freeze_emb
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

        self.attention = AspectAttention(
            hidden_dim=lstm_output_dim,
            aspect_dim=embedding_matrix.shape[1]
        )

        self.dropout = nn.Dropout(opt.input_dropout)
        self.classifier = nn.Linear(lstm_output_dim, opt.polarities_dim)

        self.dep_type = DEP_type(opt.dep_dim)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        word_emb = self.emb(tok)                  # (B, L, D)
        post_emb = self.post_emb(post)            # (B, L, Dp)
        dep_emb = self.dep_emb(deprel)            # (B, L, Dd)

        # Aspect embedding (average over aspect tokens)
        asp_emb = self.asp_emb(asp)               # (B, asp_len, D)
        asp_avg = torch.mean(asp_emb, dim=1, keepdim=True)  # (B, 1, D)
        asp_repeated = asp_avg.expand(-1, word_emb.size(1), -1)  # (B, L, D)

        emb = torch.cat([word_emb, asp_repeated, post_emb, dep_emb], dim=2)
        emb = self.dropout(emb)

        seq_lens = l.cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, seq_lens, batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed_emb)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B, L, H)

        # Apply aspect-based attention
        context, attn_weights = self.attention(lstm_out, asp_avg)  # (B, H)
        final_feat = context

        logits = self.classifier(final_feat)

        # ---------- se_loss ----------
        overall_max_len = tok.shape[1]
        batch_size = tok.shape[0]
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len]
        dep_input = self.dep_emb(deprel[:, :overall_max_len])  # (B, L, Dd)
        adj_pred = self.dep_type(dep_input, syn_dep_adj, overall_max_len, batch_size)
        se_loss = se_loss_batched(adj_pred, deprel[:, :overall_max_len], deprel.max().item() + 1)

        return logits, se_loss


class AspectAttention(nn.Module):
    def __init__(self, hidden_dim, aspect_dim):
        super(AspectAttention, self).__init__()
        self.linear = nn.Linear(hidden_dim + aspect_dim, 1)

    def forward(self, lstm_out, aspect):
        """
        lstm_out: (B, L, H)
        aspect: (B, 1, D)
        """
        L = lstm_out.size(1)
        aspect = aspect.expand(-1, L, -1)  # (B, L, D)
        combined = torch.cat([lstm_out, aspect], dim=2)  # (B, L, H+D)
        attn_scores = self.linear(combined).squeeze(-1)  # (B, L)
        attn_weights = F.softmax(attn_scores, dim=1)     # (B, L)

        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        return context, attn_weights


class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super(DEP_type, self).__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, dep_input, syn_dep_adj, overall_max_len, batch_size):
        query = self.q(dep_input).squeeze(-1)       # (B, L)
        att_adj = F.softmax(query, dim=-1)          # (B, L)
        att_adj = att_adj.unsqueeze(1).expand(-1, overall_max_len, -1)  # (B, L, L)
        att_adj = att_adj * syn_dep_adj  # Mask with syntactic dependencies
        return att_adj


def se_loss_batched(adj_pred, deprel_gold, num_relations):
    """
    adj_pred: Tensor float [batch, seq_len, seq_len], attention between tokens.
    deprel_gold: LongTensor [batch, seq_len], gold dependency labels (0 = padding).
    num_relations: int, number of dependency labels.

    Returns: Scalar loss (mean cross-entropy on real tokens).
    """
    batch, seq_len, _ = adj_pred.size()
    adj_flat = adj_pred.view(-1, seq_len)            # [batch*seq_len, seq_len]
    rel_flat = deprel_gold.view(-1)                  # [batch*seq_len]

    mask = (rel_flat != 0)
    adj_flat = adj_flat[mask]                        # [? , seq_len]
    rel_flat = rel_flat[mask]                        # [?]

    if rel_flat.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(adj_pred.device)

    logits = torch.log(adj_flat + 1e-9)  # For numerical stability
    se_loss = F.nll_loss(logits, rel_flat, reduction='mean')
    return se_loss
