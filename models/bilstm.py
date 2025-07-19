import torch
import torch.nn as nn
import torch.nn.functional as F

class BILSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt

        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float), freeze=opt.freeze_emb
        )

        input_size = embedding_matrix.shape[1]
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

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs

        emb = self.emb(tok)  # (batch_size, seq_len, embed_dim)
        emb = self.dropout(emb)

        seq_lens = l.cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, seq_lens, batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed_emb)

        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        if self.opt.bidirect:
            # Concatenate last forward and backward hidden states
            last_fw = h_n[-2]  # (batch, hidden)
            last_bw = h_n[-1]  # (batch, hidden)
            final_feat = torch.cat([last_fw, last_bw], dim=1)  # (batch, 2*hidden)
        else:
            final_feat = h_n[-1]  # (batch, hidden)

        logits = self.classifier(final_feat)  # (batch, polarity_dim)
        return logits, None
