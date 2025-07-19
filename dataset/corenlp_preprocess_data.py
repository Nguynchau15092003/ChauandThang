import json
from typing import DefaultDict
from utils.data_preprocessor import syn_dep_adj_generation
from prepare_vocab import VocabHelp
from utils.data_preprocessor import short_adj_generation

with open("../dataset/Laptops_corenlp/train.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Laptops_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Laptops_corenlp/train_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Laptops_corenlp/test.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Laptops_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Laptops_corenlp/test_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Restaurants_corenlp/train.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Restaurants_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Restaurants_corenlp/train_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Restaurants_corenlp/test.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Restaurants_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Restaurants_corenlp/test_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Tweets_corenlp/train.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Tweets_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Tweets_corenlp/train_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Tweets_corenlp/test.json", 'r') as f:
    dep_vocab = VocabHelp.load_vocab(
        '../dataset/Tweets_corenlp/vocab_dep.vocab')
    all_data = []
    data = json.load(f)
    for d in data:
        d['short'] = short_adj_generation(d['head'], max_tree_dis=10)
        d['syn_dep_adj'] = syn_dep_adj_generation(
            d['head'], d['deprel'], dep_vocab)
    wf = open('../dataset/Tweets_corenlp/test_preprocessed.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()
