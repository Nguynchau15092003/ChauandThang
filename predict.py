import torch
import argparse
import json
from transformers import BertModel
from models.dualbert import DualGCNBertClassifier
from utils.data_utils import ABSAGCNData, Tokenizer4BertGCN
from prepare_vocab import VocabHelp
import os

def load_model(opt):
    dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')
    opt.dep_size = len(dep_vocab)
    
    tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)

    model = DualGCNBertClassifier(bert, opt).to(opt.device)
    model.load_state_dict(torch.load(opt.checkpoint, map_location=opt.device))
    model.eval()
    return model, tokenizer

def predict(text, aspect, opt, model, tokenizer):
    from_idx = text.lower().index(aspect.lower())
    to_idx = from_idx + len(aspect)

    # Tạo sample giả giống như trong train file
    sample = {
        "sentence": text,
        "aspect": aspect,
        "from": from_idx,
        "to": to_idx,
        "polarity": 0  # dummy
    }

    # Tạo 1 file json tạm thời để dùng ABSAGCNData
    with open("temp_sample.json", "w", encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False)

    dataset = ABSAGCNData("temp_sample.json", tokenizer, opt=opt)
    batch = dataset[0]

    input_cols = [
        'text_bert_indices', 'bert_segments_ids', 'attention_mask', 'deprel',
        'asp_start', 'asp_end', 'src_mask', 'aspect_mask', 'short_mask', 'syn_dep_adj'
    ]

    inputs = [batch[col].unsqueeze(0).to(opt.device) for col in input_cols]

    with torch.no_grad():
        outputs, _ = model(inputs)
        pred = torch.argmax(outputs, dim=-1).item()

    # Xoá file tạm
    os.remove("temp_sample.json")
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Input sentence')
    parser.add_argument('--aspect', type=str, required=True, help='Aspect term')
    parser.add_argument('--model_name', default='dualbert', type=str)
    parser.add_argument('--dataset', default='Movie_vietnamese', type=str)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model .pt checkpoint')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--pretrained_bert_name', default='vinai/phobert-base', type=str)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--bert_dropout', default=0.3, type=float)
    parser.add_argument('--linear_dropout', type=float, default=0.2)
    parser.add_argument('--polarities_dim', default=5, type=int)  # Match training
    opt = parser.parse_args()

    opt.vocab_dir = f"./dataset/{opt.dataset}"
    opt.device = torch.device(opt.device)

    model, tokenizer = load_model(opt)
    pred = predict(opt.text, opt.aspect, opt, model, tokenizer)

    print(f"[PREDICTED LABEL]: {pred}")

if __name__ == '__main__':
    main()