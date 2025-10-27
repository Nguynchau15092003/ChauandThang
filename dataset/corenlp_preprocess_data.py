import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import DefaultDict
from utils.data_preprocessor import syn_dep_adj_generation
from prepare_vocab import VocabHelp
from utils.data_preprocessor import short_adj_generation

# with open("./Laptops_corenlp/train.json", 'r') as f:
#     dep_vocab = VocabHelp.load_vocab(
#         './Laptops_corenlp/vocab_dep.vocab')
#     all_data = []
#     data = json.load(f)
#     for d in data:
#         d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
#         d['syn_dep_adj'] = syn_dep_adj_generation(
#             d['head'], d['deprel'], dep_vocab)
#     wf = open('./Laptops_corenlp/train_preprocessed.json', 'w')
#     wf.write(json.dumps(data, indent=4))
#     wf.close()

# with open("./Laptops_corenlp/test.json", 'r') as f:
#     dep_vocab = VocabHelp.load_vocab(
#         './Laptops_corenlp/vocab_dep.vocab')
#     all_data = []
#     data = json.load(f)
#     for d in data:
#         d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
#         d['syn_dep_adj'] = syn_dep_adj_generation(
#             d['head'], d['deprel'], dep_vocab)
#     wf = open('./Laptops_corenlp/test_preprocessed.json', 'w')
#     wf.write(json.dumps(data, indent=4))
#     wf.close()

# with open("./Restaurants_corenlp/train.json", 'r') as f:
#     dep_vocab = VocabHelp.load_vocab(
#         './Restaurants_corenlp/vocab_dep.vocab')
#     all_data = []
#     data = json.load(f)
#     for d in data:
#         d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
#         d['syn_dep_adj'] = syn_dep_adj_generation(
#             d['head'], d['deprel'], dep_vocab)
#     wf = open('./Restaurants_corenlp/train_preprocessed.json', 'w')
#     wf.write(json.dumps(data, indent=4))
#     wf.close()

# with open("./Restaurants_corenlp/test.json", 'r') as f:
#     dep_vocab = VocabHelp.load_vocab(
#         './Restaurants_corenlp/vocab_dep.vocab')
#     all_data = []
#     data = json.load(f)
#     for d in data:
#         d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
#         d['syn_dep_adj'] = syn_dep_adj_generation(
#             d['head'], d['deprel'], dep_vocab)
#     wf = open('./Restaurants_corenlp/test_preprocessed.json', 'w')
#     wf.write(json.dumps(data, indent=4))
#     wf.close()
with open("./Movie_vietnamese/train.json", 'r', encoding='utf-8') as f:
    dep_vocab = VocabHelp.load_vocab('./Movie_vietnamese/vocab_dep.vocab')
    data = json.load(f)

    filtered_data = []
    for d in data:
        if len(d['token']) < 100:  # Lọc những câu có token length < 100
            d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
            d['syn_dep_adj'] = syn_dep_adj_generation(d['head'], d['deprel'], dep_vocab)
            filtered_data.append(d)

    with open('./Movie_vietnamese/train_preprocessed.json', 'w', encoding='utf-8') as wf:
        json.dump(filtered_data, wf, indent=4, ensure_ascii=False)

with open("./Movie_vietnamese/test.json", 'r', encoding='utf-8') as f:
    dep_vocab = VocabHelp.load_vocab('./Movie_vietnamese/vocab_dep.vocab')
    data = json.load(f)

    filtered_data = []
    for d in data:
        if len(d['token']) < 100:  # Lọc những câu có token length < 100
            d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
            d['syn_dep_adj'] = syn_dep_adj_generation(d['head'], d['deprel'], dep_vocab)
            filtered_data.append(d)

    with open('./Movie_vietnamese/test_preprocessed.json', 'w', encoding='utf-8') as wf:
        json.dump(filtered_data, wf, indent=4, ensure_ascii=False)




