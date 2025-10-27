"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter


class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            if tok in counter:
                del counter[tok]

        # Sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: (-tup[1], tup[0]))

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi: word to index
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        return self.stoi == other.stoi and self.itos == other.itos

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        for w in v.itos:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', required=True, help='Directory containing train.json and test.json')
    parser.add_argument('--vocab_dir', required=True, help='Directory to save vocab files')
    parser.add_argument('--lower', action='store_true', help='Lowercase all words if set')
    return parser.parse_args()


def load_tokens(filename):
    with open(filename, encoding='utf-8') as infile:  # ✅ đảm bảo đọc tiếng Việt
        data = json.load(infile)
        tokens = []
        pos = []
        dep = []
        max_len = 0
        for d in data:
            tokens.extend(d['token'])
            pos.extend(d['pos'])
            dep.extend(d['deprel'])
            max_len = max(len(d['token']), max_len)
    print(f"{len(tokens)} tokens from {len(data)} examples loaded from {filename}.")
    return tokens, pos, dep, max_len


def main():
    args = parse_args()

    # Input files
    train_file = f"{args.data_dir}/train.json"
    test_file = f"{args.data_dir}/test.json"

    # Output files
    vocab_tok_file = f"{args.vocab_dir}/vocab_tok.vocab"
    vocab_post_file = f"{args.vocab_dir}/vocab_post.vocab"
    vocab_pos_file = f"{args.vocab_dir}/vocab_pos.vocab"
    vocab_dep_file = f"{args.vocab_dir}/vocab_dep.vocab"
    vocab_pol_file = f"{args.vocab_dir}/vocab_pol.vocab"

    # Load data
    print("🔄 Loading data files...")
    train_tokens, train_pos, train_dep, train_max_len = load_tokens(train_file)
    test_tokens, test_pos, test_dep, test_max_len = load_tokens(test_file)

    if args.lower:
        print("🔡 Lowercasing tokens...")
        train_tokens = [t.lower() for t in train_tokens]
        test_tokens = [t.lower() for t in test_tokens]

    # Counters
    print("🔢 Building counters...")
    token_counter = Counter(train_tokens + test_tokens)
    pos_counter = Counter(train_pos + test_pos)
    dep_counter = Counter(train_dep + test_dep)
    max_len = max(train_max_len, test_max_len)
    post_counter = Counter(list(range(-max_len, max_len)))
    pol_counter = Counter( [
    "surprise",
    "optimism",
    "joy",
    "disgust",
    "sadness"
])
    # pol_counter = Counter( [
    #     "neutral",
    #     "negative",
    #     "positive"
    # ])

    # Build vocabs
    print("🛠️ Building vocabularies...")
    token_vocab = VocabHelp(token_counter, specials=['<pad>', '<unk>'])
    pos_vocab = VocabHelp(pos_counter, specials=['<pad>', '<unk>'])
    dep_vocab = VocabHelp(dep_counter, specials=['<pad>', '<unk>'])
    post_vocab = VocabHelp(post_counter, specials=['<pad>', '<unk>'])
    pol_vocab = VocabHelp(pol_counter, specials=[])

    print(f"✅ Vocab sizes -> token: {len(token_vocab)}, pos: {len(pos_vocab)}, "
          f"dep: {len(dep_vocab)}, post: {len(post_vocab)}, pol: {len(pol_vocab)}")

    # Save vocabs
    print("💾 Saving vocab files...")
    token_vocab.save_vocab(vocab_tok_file)
    pos_vocab.save_vocab(vocab_pos_file)
    dep_vocab.save_vocab(vocab_dep_file)
    post_vocab.save_vocab(vocab_post_file)
    pol_vocab.save_vocab(vocab_pol_file)
    print("✅ All done.")


if __name__ == '__main__':
    main()
