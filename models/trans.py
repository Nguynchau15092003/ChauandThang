import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TransformerClassifier, self).__init__()
        self.opt = opt

        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=opt.freeze_emb
        )

        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0)
        self.dep_emb = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)

        self.embed_dim = embedding_matrix.shape[1]
        self.hidden_dim = opt.transformer_hidden_dim

        # Combine all embeddings: word + aspect + position + dependency
        self.input_dim = self.embed_dim * 2 + opt.post_dim + opt.dep_dim
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.dropout = nn.Dropout(opt.input_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=opt.n_heads,
            dim_feedforward=opt.ffn_dim,
            dropout=opt.transformer_dropout,
            batch_first=True  # Allows input shape (B, L, D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=opt.num_transformer_layers
        )

        self.fc = nn.Linear(self.hidden_dim, opt.polarities_dim)
        self.dep_type = DEP_type(opt.dep_dim)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        word_embed = self.embed(tok)
        aspect_embed = self.embed(asp)
        aspect_avg = torch.mean(aspect_embed, dim=1, keepdim=True)
        aspect_repeated = aspect_avg.expand(-1, word_embed.size(1), -1)

        post_embed = self.post_emb(post)
        dep_embed = self.dep_emb(deprel)

        x = torch.cat([word_embed, aspect_repeated, post_embed, dep_embed], dim=2)
        x = self.dropout(self.input_proj(x))  # (B, L, D_model)

        # Transformer expects mask where True = pad
        attention_mask = (tok == 0)
        x = self.transformer(x, src_key_padding_mask=attention_mask)

        # Mean pooling over non-padding tokens
        mask = (~attention_mask).unsqueeze(2).float()
        x_mean = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)

        logits = self.fc(x_mean)

        # Structured loss
        overall_max_len = tok.shape[1]
        batch_size = tok.shape[0]
        syn_dep_adj = syn_dep_adj[:, :overall_max_len, :overall_max_len]
        adj_pred = self.dep_type(self.dep_emb.weight, syn_dep_adj, overall_max_len, batch_size)
        se_loss = se_loss_batched(adj_pred, head[:, :syn_dep_adj.shape[1]])


        return logits, se_loss


class DEP_type(nn.Module):
    def __init__(self, att_dim):
        super(DEP_type, self).__init__()
        self.q = nn.Linear(att_dim, 1)

    def forward(self, dep_emb_weight, syn_dep_adj, overall_max_len, batch_size):
        query = self.q(dep_emb_weight).T  # (1, dep_size)
        attention_scores = F.softmax(query, dim=-1)  # (1, dep_size)
        attention_scores = attention_scores.unsqueeze(0).repeat(batch_size, overall_max_len, 1)  # (B, L, dep_size)
        att_adj = torch.gather(attention_scores, 2, syn_dep_adj)
        att_adj[syn_dep_adj == 0] = 0.0
        return att_adj

def se_loss_batched(adj_pred, head_gold):
    batch, seq_len, _ = adj_pred.size()

    adj_flat = adj_pred.view(-1, seq_len)
    head_flat = head_gold.view(-1)

    mask = (head_flat != 0)
    adj_flat = adj_flat[mask]
    head_flat = head_flat[mask]

    if head_flat.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(adj_pred.device)

    # Optional: normalize adj_flat rows to sum=1 again, just in case
    adj_flat = adj_flat / adj_flat.sum(dim=1, keepdim=True).clamp(min=1e-9)

    logits = torch.log(adj_flat + 1e-9)
    se_loss = F.nll_loss(logits, head_flat, reduction='mean')
    return se_loss