import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        # Embedding layer từ pretrained embeddings, freeze=True để không update
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        
        input_size = embedding_matrix.shape[1]  # kích thước embedding word
        self.lstm = nn.LSTM(input_size, opt.hidden_dim, num_layers=opt.rnn_layers,
                            batch_first=True, dropout=opt.rnn_dropout, bidirectional=opt.bidirect)
        
        lstm_output_dim = opt.hidden_dim * 2 if opt.bidirect else opt.hidden_dim
        
        self.classifier = nn.Linear(lstm_output_dim, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.input_dropout)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, short_mask, syn_dep_adj = inputs
        
        # Embedding lookup
        emb = self.emb(tok)  # (batch_size, seq_len, embed_dim)
        emb = self.dropout(emb)

        # Chuẩn bị packed sequence để bỏ qua padding
        seq_lens = l.cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, seq_lens, batch_first=True, enforce_sorted=False)

        # LSTM
        packed_out, (h_n, c_n) = self.lstm(packed_emb)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Lấy representation sequence qua mean pooling (bỏ phần padding)
        mask = mask.float()
        mask = mask[:, :out.size(1)]  # đảm bảo shape phù hợp
        mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        out = out * mask
        seq_len = mask.sum(dim=1)  # (batch_size, 1)
        seq_len = seq_len.clamp(min=1e-9)  # tránh chia cho 0

        pooled = out.sum(dim=1) / seq_len  # mean pooling

        # Classifier
        logits = self.classifier(pooled)

        # Không có se_loss (loss khác) trong bản này
        se_loss = None

        return logits, se_loss