import os
import sys
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
import networkx as nx
from prepare_vocab import VocabHelp


def create_mask(max_length, short):
    mask = np.full((11, max_length, max_length), -99999, dtype='float32')
    short_length = len(short)
    assert len(short) == len(short[0])
    mask[0, :short_length, :short_length] = 0
    for min_dis in range(1, 11):
        for i in range(short_length):
            for j in range(short_length):
                if short[i][j] >= 1 and short[i][j] <= min_dis:
                    mask[min_dis][i][j] = 0
        for i in range(short_length):
            mask[min_dis][i][i] = 0
    return mask


def create_mask_bert(max_length, context_len, column_short):
    mask = np.full((11, max_length, max_length), -99999, dtype='float32')
    mask[:, 0, :context_len] = 0
    mask[0, 1:context_len + 1, 1:context_len + 1] = 0
    for min_dis in range(1, 11):
        for i in range(context_len):
            for j in range(context_len):
                if column_short[i][j] >= 1 and column_short[i][j] <= min_dis:
                    mask[min_dis][i + 1][j + 1] = 0
    return mask


def dep_adj_expansion(syn_adj, max_len, weight):
    for node_id in range(max_len):
        syn_adj.append([node_id, node_id, weight])
    syn_nx = nx.Graph()
    syn_nx.add_nodes_from(range(max_len))
    syn_nx.add_weighted_edges_from(syn_adj)
    syn_adj_ = nx.adjacency_matrix(syn_nx).A
    return syn_adj_


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label
                pos = list(d['pos'])         # pos_tag
                head = list(d['head'])       # head
                deprel = list(d['deprel'])   # deprel
                short = list(d['short'])
                syn_dep_adj = list(d['syn_dep_adj'])
                # position
                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                    + [0 for _ in range(aspect['from'], aspect['to'])] \
                    + [i - aspect['to'] +
                        1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                        + [1 for _ in range(aspect['from'], aspect['to'])] \
                        + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask,
                          'aspect_post': aspect_post, 'text_list': text_list, 'short': short, 'syn_dep_adj': syn_dep_adj}
                all_data.append(sample)

    return all_data


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(
            fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''

    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    ''' transform text to indices '''

    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char

    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = []
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                tokens = Tokenizer.split_text(text_raw)
                corpus.extend(token for token in tokens if token not in corpus)
        corpus = sorted(corpus)
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length,
                                      padding=padding, truncating=truncating)

    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''

    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            text = tokenizer.text_to_sequence(obj['text'])
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10
            post = [post_vocab.stoi.get(t, post_vocab.unk_index)
                    for t in obj['post']]
            post = tokenizer.pad_sequence(
                post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index)
                   for t in obj['pos']]
            pos = tokenizer.pad_sequence(
                pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index)
                      for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(
                deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            mask = tokenizer.pad_sequence(
                obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')

            # left_len = np.sum(left_indices != 0)
            # aspect_len = np.sum(aspect_indices != 0)
            # aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

            adj = np.ones(opt.max_length) * opt.pad_id
            if opt.parseadj:
                from absa_parser import headparser
                # * adj
                headp, syntree = headparser.parse_heads(obj['text'])
                adj = softmax(headp[0])
                adj = np.delete(adj, 0, axis=0)
                adj = np.delete(adj, 0, axis=1)
                adj -= np.diag(np.diag(adj))
                if not opt.direct:
                    adj = adj + adj.T
                adj = adj + np.eye(adj.shape[0])
                adj = np.pad(adj, (0, opt.max_length -
                             adj.shape[0]), 'constant')

            if opt.parsehead:
                from absa_parser import headparser
                headp, syntree = headparser.parse_heads(obj['text'])
                syntree2head = [[leaf.father for leaf in tree.leaves()]
                                for tree in syntree]
                head = tokenizer.pad_sequence(
                    syntree2head[0], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            else:
                head = tokenizer.pad_sequence(
                    obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            length = obj['length']
            polarity = polarity_dict[obj['label']]
            # short 根据 obj['short']
            short_mask = create_mask(opt.max_length, obj['short'])
            short_mask = short_mask[:opt.attention_heads, :, :]
            syn_dep_adj = obj['syn_dep_adj']
            root_id = dep_vocab.stoi.get('root', dep_vocab.unk_index)
            syn_dep_adj = dep_adj_expansion(
                syn_dep_adj, opt.max_length, root_id)

            data.append({
                'text': text,
                'aspect': aspect,
                'post': post,
                'pos': pos,
                'deprel': deprel,
                'head': head,
                'adj': adj,
                'mask': mask,
                'length': length,
                'polarity': polarity,
                'short_mask': short_mask,
                'syn_dep_adj': syn_dep_adj
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>':  # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(
                        tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>':  # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()

        return word_vec


def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = '/content/glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, embed_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        dep_vocab = VocabHelp.load_vocab(
            opt.vocab_dir + '/vocab_dep.vocab')
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index)
                      for t in obj['deprel']]
            deprel = Tokenizer.pad_sequence(
                deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tokens.append(t)
                    left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i + offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()

            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')

            context_asp_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                bert_tokens) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [
                0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * \
                (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + \
                (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(
                context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(
                context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')

            row_short = obj['short']

            for i in range(context_len - 1):
                if tok2ori_map[i + 1] == tok2ori_map[i]:
                    a = row_short[i]
                    row_short = np.insert(row_short, i, values=a, axis=0)

            column_short = row_short
            for j in range(context_len - 1):
                if tok2ori_map[j + 1] == tok2ori_map[j]:
                    a = column_short[:, j]
                    column_short = np.insert(column_short, j, values=a, axis=1)

            short_mask = create_mask_bert(
                opt.max_length, context_len, column_short)
            short_mask = short_mask[:opt.attention_heads, :, :]
            syn_dep_adj = obj['syn_dep_adj']
            root_id = dep_vocab.stoi.get('root', dep_vocab.unk_index)
            syn_dep_adj = dep_adj_expansion(
                syn_dep_adj, opt.max_length, root_id)

            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'deprel': deprel,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
                'short_mask': short_mask,
                'syn_dep_adj': syn_dep_adj
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
