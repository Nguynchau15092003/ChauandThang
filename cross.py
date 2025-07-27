import os
import json
import argparse
import sys
import copy
import torch
import numpy as np
from sklearn.model_selection import KFold
from prepare_vocab import VocabHelp   

from train import Instructor, setup_seed, get_parser
from train import (
    model_classes, dataset_files, input_colses,
    initializers, optimizers, MIN_ACC
)

def kfold_cross_validation(opt, k=5):
    full_data_path = opt.dataset_file['train']

    with open(full_data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    kf = KFold(n_splits=k, shuffle=True, random_state=opt.seed)
    all_data = np.array(all_data)
    accs, f1s,precision,recall = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n=== Fold {fold + 1}/{k} ===")

        train_data = all_data[train_idx].tolist()
        val_data = all_data[val_idx].tolist()

        fold_train_path = f"./dataset/{opt.dataset}/fold_train_{fold}.json"
        fold_val_path = f"./dataset/{opt.dataset}/fold_val_{fold}.json"

        with open(fold_train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
        with open(fold_val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2)

        # Clone `opt` cho fold hiện tại
        opt_fold = copy.deepcopy(opt)
        opt_fold.dataset_file = {
            'train': fold_train_path,
            'test': fold_val_path
        }

        setup_seed(opt_fold.seed)
        instructor = Instructor(opt_fold)
        instructor.run()

        eval_result = instructor._evaluate()
        accs.append(eval_result['test_acc'])
        f1s.append(eval_result['f1'])
        precision.append(eval_result['precision'])
        recall.append(eval_result['recall'])
        CV=np.std(accs)/np.mean(accs)
 
        # Nếu muốn xoá file tạm sau mỗi fold, bỏ comment:
        # os.remove(fold_train_path)
        # os.remove(fold_val_path)

    print("\n=== K-Fold Cross Validation Result ===")
    print(f"Accuracy: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}, min={np.min(accs):.4f}, max={np.max(accs):.4f}")
    print(f"F1 Score: mean={np.mean(f1s):.4f}, std={np.std(f1s):.4f}, min={np.min(f1s):.4f}, max={np.max(f1s):.4f}")
    print(f"precision Score: mean={np.mean(precision):.4f}, std={np.std(precision):.4f}, min={np.min(precision):.4f}, max={np.max(precision):.4f}")
    print(f"Recall Score: mean={np.mean(recall):.4f}, std={np.std(recall):.4f}, min={np.min(recall):.4f}, max={np.max(recall):.4f}")
    print(f"CV (Coefficient of Variation for Accuracy): {CV:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', default=5, type=int, help="Number of folds")
    parser.add_argument('--train_args', nargs=argparse.REMAINDER, help="Args for training script (train.py)")
    args = parser.parse_args()

    # Sửa sys.argv để parser của train.py có thể hiểu được
    sys.argv = ['train.py'] + (args.train_args if args.train_args else [])
    opt = get_parser().parse_args()

    # Set các biến cần thiết từ train.py
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.vocab_dir = f'./dataset/{opt.dataset}'
    opt.min_acc = MIN_ACC[opt.model_name][opt.dataset]

    if 'bert' not in opt.model_name:
        opt.rnn_hidden = opt.hidden_dim

    if opt.device is None:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt.device = torch.device(opt.device)
    opt.eval = False  # Chạy train chứ không phải eval

    kfold_cross_validation(opt, k=args.kfold)
