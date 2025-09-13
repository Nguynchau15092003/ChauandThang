import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DualGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bert = bert
        self.hidden_dim = self.bert.config.hidden_size  # hidden size của BERT

        # GCN model: sử dụng vector embedding từ BERT, không dùng embedding_matrix
        self.gcn_model = GCNAbsaBertModel(opt)

        self.classifier = nn.Linear(self.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        """
        inputs = (
            text_bert_indices, bert_segments_ids, attention_mask,
            head, deprel, post, pos, aspect_mask, seq_lens, syn_dep_adj
        )
        """
        text_bert_indices, bert_segments_ids, attention_mask, head, deprel, post, pos, aspect_mask, seq_lens, syn_dep_adj = inputs

        # Lấy output BERT
        bert_outputs = self.bert(
            input_ids=text_bert_indices,
            token_type_ids=bert_segments_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # GCN model
        outputs1, outputs2, adj_ag, adj_dep = self.gcn_model(sequence_output, attention_mask, head, deprel, post, pos, aspect_mask, seq_lens, syn_dep_adj)

        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_outputs)

        # Tính penalty giống DualGCNClassifier gốc
        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).to(adj_ag.device)
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag @ adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).to(ortho.device)

        penal = None
        if self.opt.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).to(adj_ag.device)
            penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).to(adj_ag.device)
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2
        
        elif self.opt.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_ag.size(0)).to(adj_ag.device)
            penal = self.opt.alpha * penal

        elif self.opt.losstype == 'differentiatedloss':
            penal = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).to(adj_ag.device)
            penal = self.opt.beta * penal
        
        return logits, penal


class GCNAbsaBertModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.hidden_dim = opt.hidden_dim  # Thường set = BERT hidden size hoặc fix theo opt

        # Nếu bạn muốn dùng pos_emb hoặc post_emb, bạn có thể khai báo ở đây
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None

        # GCN layer (giống GCN model cũ, nhưng input_dim = BERT hidden size + pos_dim + post_dim)
        input_dim = self.hidden_dim
        if self.opt.pos_dim > 0:
            input_dim += opt.pos_dim
        if self.opt.post_dim > 0:
            input_dim += opt.post_dim

        self.gcn = GCN(opt, embeddings=None, mem_dim=self.hidden_dim, num_layers=opt.num_layers)

    def forward(self, sequence_output, attention_mask, head, deprel, post, pos, aspect_mask, seq_lens, syn_dep_adj):
        """
        sequence_output: (batch_size, seq_len, hidden_dim) -- output từ BERT
        attention_mask: (batch_size, seq_len)
        các input khác như trên
        """
        batch_size, seq_len, _ = sequence_output.size()

        # Chuẩn bị embedding cho pos, post nếu có
        embs = [sequence_output]
        if self.opt.pos_dim > 0 and pos is not None:
            embs.append(self.pos_emb(pos))
        if self.opt.post_dim > 0 and post is not None:
            embs.append(self.post_emb(post))
        embs = torch.cat(embs, dim=2)

        # Dùng lại phần encode RNN của GCN? Nếu không dùng RNN nữa, có thể bỏ hoặc thay thế
        # Ở đây giả sử bỏ RNN vì BERT đã encode rồi
        gcn_inputs = embs  # trực tiếp đưa vào GCN

        # Xử lý adjacency matrix dependency
        if self.opt.parseadj:
            adj_dep = syn_dep_adj.float()
        else:
            # Tạo adjacency từ head nếu cần (giống code gốc)
            # Bạn cần tự code lại hàm head_to_tree, tree_to_adj nếu dùng
            adj_dep = syn_dep_adj.float()

        # Tạo mask aspect cho GCN pooling
        mask = aspect_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        asp_wn = mask.sum(dim=1)  # số từ aspect (batch, 1)

        # Chạy GCN
        h1, h2, adj_ag = self.gcn(adj_dep, (gcn_inputs, mask, seq_lens))

        # avg pooling aspect features
        outputs1 = (h1 * mask).sum(dim=1) / asp_wn.clamp(min=1e-10)
        outputs2 = (h2 * mask).sum(dim=1) / asp_wn.clamp(min=1e-10)

        return outputs1, outputs2, adj_ag, adj_dep


class GCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim

        # Vì input đã là BERT output hoặc kết hợp embedding khác, self.in_dim = mem_dim
        self.in_dim = mem_dim  

        # Dropouts
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # Linear layers cho mỗi layer GCN
        self.W = nn.ModuleList([nn.Linear(self.in_dim if i == 0 else self.mem_dim, self.mem_dim) for i in range(self.layers)])
        self.weight_list = nn.ModuleList([nn.Linear(self.in_dim if j == 0 else self.mem_dim, self.mem_dim) for j in range(self.layers)])

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

        # Khởi tạo parameter affine
        nn.init.xavier_uniform_(self.affine1)
        nn.init.xavier_uniform_(self.affine2)

    def forward(self, adj, inputs):
        """
        adj: adjacency matrix (batch, seq_len, seq_len)
        inputs: tuple (gcn_inputs, mask, seq_lens)
            gcn_inputs: (batch, seq_len, mem_dim)
            mask: aspect mask (batch, seq_len, 1)
            seq_lens: độ dài câu (batch)
        """
        gcn_inputs, mask, seq_lens = inputs

        denom_dep = adj.sum(2).unsqueeze(2) + 1
        outputs_dep = gcn_inputs
        adj_ag = adj  # nếu bạn muốn dùng adjacency attention khác thì thay đổi

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_ag = gcn_inputs

        for l in range(self.layers):
            # SynGCN
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # SemGCN
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # Mutual Biaffine
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), gAxW_ag.transpose(1,2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), gAxW_dep.transpose(1,2)), dim=-1)

            gAxW_dep = torch.bmm(A1, gAxW_ag)
            gAxW_ag = torch.bmm(A2, gAxW_dep)

            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, adj_ag


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn