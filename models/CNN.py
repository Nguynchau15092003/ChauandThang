import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClasiffier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CNNClasiffier, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float), freeze=opt.freeze_emb
        )

        self.embed_dim = embedding_matrix.shape[1]
        self.num_filters = opt.num_filters  # số lượng filters mỗi kernel size
        self.kernel_sizes = opt.kernel_sizes  # ví dụ [3,4,5]
        self.dropout = nn.Dropout(opt.input_dropout)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embed_dim * 2,  # word + aspect
                out_channels=self.num_filters,
                kernel_size=k
            )
            for k in self.kernel_sizes
        ])

        self.fc = nn.Linear(self.num_filters * len(self.kernel_sizes), opt.polarities_dim)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        # word embedding
        word_embed = self.embed(tok)      # (batch, seq_len, embed_dim)
        aspect_embed = self.embed(asp)    # (batch, asp_len, embed_dim)
        aspect_avg = torch.mean(aspect_embed, dim=1, keepdim=True)  # (batch, 1, embed_dim)

        aspect_repeated = aspect_avg.expand(-1, word_embed.size(1), -1)  # (batch, seq_len, embed_dim)
        x = torch.cat([word_embed, aspect_repeated], dim=2)             # (batch, seq_len, 2*embed_dim)
        x = self.dropout(x)

        # CNN expects input of shape: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, 2*embed_dim, seq_len)

        conv_outputs = [
            F.relu(conv(x)) for conv in self.convs  # each: (batch, num_filters, seq_len - k + 1)
        ]

        # Max-pooling over time (dim=2)
        pooled_outputs = [
            F.max_pool1d(out, kernel_size=out.size(2)).squeeze(2)
            for out in conv_outputs
        ]

        # Concatenate all pooled features
        cat = torch.cat(pooled_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))

        logits = self.fc(cat)
        return logits, None
