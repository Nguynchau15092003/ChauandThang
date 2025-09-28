import torch
import argparse
import json
import os
import numpy as np
import stanza

from transformers import AutoModel
from models.dualbert import DualGCNBertClassifier
from utils.data_utils import ABSAGCNData, Tokenizer4BertGCN
from prepare_vocab import VocabHelp

# Download và khởi tạo pipeline Stanza 1 lần duy nhất
stanza.download('vi')
nlp = stanza.Pipeline('vi', processors='tokenize,pos,lemma,depparse', use_gpu=False)


def create_short_matrix(heads):
    n = len(heads)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        dist[i][i] = 0
    for i, h in enumerate(heads):
        if h > 0:
            dist[i][h - 1] = 1
            dist[h - 1][i] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    max_dist = n * 10
    dist[dist == np.inf] = max_dist
    dist = dist.astype(int)
    return dist.tolist()


def create_syn_dep_adj(heads):
    edges = []
    for i, h in enumerate(heads):
        if h > 0:
            edges.append([i, h - 1, 1])
            edges.append([h - 1, i, 1])
    return edges


def find_aspect_positions(tokens, aspect_tokens):
    aspect_len = len(aspect_tokens)
    for i in range(len(tokens) - aspect_len + 1):
        if tokens[i:i + aspect_len] == aspect_tokens:
            return i, i + aspect_len
    raise ValueError(f"Aspect term '{' '.join(aspect_tokens)}' not found in tokenized sentence: {tokens}")


def create_temp_json_multi_aspects(text, aspect_list):
    """
    Tạo sample JSON với nhiều aspect cho cùng 1 câu.
    Trả về list các dict aspect có thông tin vị trí để sử dụng sau.
    """
    doc = nlp(text)
    tokens = []
    pos = []
    head = []
    deprel = []

    for sent in doc.sentences:
        for word in sent.words:
            tokens.append(word.text.lower())
            pos.append(word.upos)
            head.append(word.head)
            deprel.append(word.deprel)

    short = create_short_matrix(head)
    syn_dep_adj = create_syn_dep_adj(head)

    aspects_data = []
    for asp in aspect_list:
        asp_doc = nlp(asp)
        aspect_tokens = [word.text.lower() for sent in asp_doc.sentences for word in sent.words]
        try:
            from_idx, to_idx = find_aspect_positions(tokens, aspect_tokens)
        except ValueError as e:
            print(f"[WARNING] {e}, bỏ aspect này.")
            continue
        aspects_data.append({
            "term": aspect_tokens,
            "from": from_idx,
            "to": to_idx,
            "polarity": 5  # placeholder
        })

    sample = [
        {
            "token": tokens,
            "pos": pos,
            "head": head,
            "deprel": deprel,
            "aspects": aspects_data,
            "short": short,
            "syn_dep_adj": syn_dep_adj
        }
    ]

    with open("temp_sample.json", "w", encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=4)

    return aspects_data  # Trả về danh sách aspects có vị trí
def load_model(opt):
    dep_vocab = VocabHelp.load_vocab(os.path.join(opt.vocab_dir, 'vocab_dep.vocab'))
    opt.dep_size = len(dep_vocab)

    tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
    bert = AutoModel.from_pretrained(opt.pretrained_bert_name)

    model = DualGCNBertClassifier(bert, opt).to(opt.device)
    model.load_state_dict(torch.load(opt.checkpoint, map_location=opt.device))
    model.eval()
    return model, tokenizer


def predict_multi_aspects(text, aspect_list, opt, model, tokenizer):
    results = []
    polarity_dict_inv = {0: "surprise", 1: "optimism", 2: "joy", 3: "disgust", 4: "sadness"}

    for i, asp in enumerate(aspect_list):
        # Tạo file JSON riêng cho aspect i
        aspect_infos = create_temp_json_multi_aspects(text, [asp])  # chỉ 1 aspect tại 1 thời điểm
        if len(aspect_infos) == 0:
            print(f"[WARNING] Aspect '{asp}' không tìm thấy trong câu, bỏ qua.")
            continue

        dataset = ABSAGCNData("temp_sample.json", tokenizer, opt=opt, is_training=False)
        batch = dataset[0]  # sample duy nhất

        input_cols = [
            'text_bert_indices', 'bert_segments_ids', 'attention_mask', 'deprel',
            'asp_start', 'asp_end', 'src_mask', 'aspect_mask', 'short_mask', 'syn_dep_adj'
        ]

        inputs = []
        for col in input_cols:
            val = batch[col]
            if torch.is_tensor(val):
                inputs.append(val.unsqueeze(0).to(opt.device))  # batch size 1
            else:
                if isinstance(val, (list, tuple)):
                    inputs.append(torch.tensor([val[0]], device=opt.device))
                else:
                    inputs.append(torch.tensor([val], device=opt.device))

        with torch.no_grad():
            outputs, _ = model(inputs)
            pred = torch.argmax(outputs, dim=-1).item()

        polarity = polarity_dict_inv.get(pred, "neutral")
        aspect_info = aspect_infos[0]

        results.append({
            "term": aspect_info['term'],
            "from": aspect_info['from'],
            "to": aspect_info['to'],
            "polarity": polarity
        })

        os.remove("temp_sample.json")  # xóa file tạm mỗi lần

    return results



def generate_structure_json(text, aspects_result):
    """
    Tạo cấu trúc JSON kết quả theo yêu cầu.
    """
    return {
        "text": text,
        "aspects": aspects_result
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='dualbert', type=str)
    parser.add_argument('--dataset', default='Movie_vietnamese', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--kernel_sizes', default='3,4,5', type=str)
    parser.add_argument('--num_filters', default=100, type=int)
    parser.add_argument('--freeze_emb', type=bool, default=False)
    parser.add_argument('--learning_rate', default=5e-5 , type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30)
    parser.add_argument('--pos_dim', type=int, default=30)
    parser.add_argument('--dep_dim', type=int, default=30)
    parser.add_argument('--sentic', default='eng', type=str, choices=['eng', 'vi'],
                    help='Chọn bộ senticnet (eng hoặc vi)')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--polarities_dim', default=5, type=int)
    parser.add_argument('--input_dropout', type=float, default=0.3)
    parser.add_argument('--gcn_dropout', type=float, default=0.1)
    parser.add_argument('--lower', default=True)
    parser.add_argument('--direct', default=False)
    parser.add_argument('--loop', default=True)
    parser.add_argument('--bidirect', default=True)
    parser.add_argument('--rnn_hidden', type=int, default=50)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--rnn_dropout', type=float, default=0.1)
    parser.add_argument('--attention_heads', default=4, type=int)
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--transformer_hidden_dim', default=256, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--ffn_dim', default=512, type=int)
    parser.add_argument('--num_transformer_layers', default=4, type=int)
    parser.add_argument('--transformer_dropout', default=0.2, type=float)
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true')
    parser.add_argument('--parsehead', default=False, action='store_true')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default=None, type=str)
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)
    parser.add_argument('--pretrained_bert_name', default='vinai/phobert-base', type=str)
    parser.add_argument("--orthogonal_weight", default=1.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--differential_weight", default=1.0, type=float)
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3)
    parser.add_argument('--linear_dropout', type=float, default=0.2)
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=5e-5, type=float)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--gamma', default=0.0, type=float)
    parser.add_argument('--text', type=str, required=True, help='Input sentence')
    parser.add_argument('--aspects', type=str, required=True, help='Aspect term')
    parser.add_argument('--run_device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model .pt checkpoint')
    opt = parser.parse_args()
    opt.device = torch.device(opt.run_device)
    opt.vocab_dir = f"./dataset/{opt.dataset}"

 # Tách aspect thành list
    aspect_list = [a.strip() for a in opt.aspects.split(",")]

    model, tokenizer = load_model(opt)
    results = predict_multi_aspects(opt.text, aspect_list, opt, model, tokenizer)

    result_structure = generate_structure_json(opt.text, results)

    print(json.dumps(result_structure, ensure_ascii=False, indent=2))
    with open("output_structure.json", "w", encoding="utf-8") as f:
        json.dump(result_structure, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
