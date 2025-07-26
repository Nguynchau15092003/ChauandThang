import os
import json
import shutil
import random
import argparse
import numpy as np
from sklearn.model_selection import KFold
from train import Instructor, setup_seed, main as train_main

from train import logger  # dùng logger nếu muốn log giống `train.py`

def kfold_cross_validation(opt, k=5):
    # Load toàn bộ data (giả định json list dạng như train_preprocessed.json)
    full_data_path = opt.dataset_file['train']
    with open(full_data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    kf = KFold(n_splits=k, shuffle=True, random_state=opt.seed)
    all_data = np.array(all_data)
    accs, f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n=== Fold {fold + 1}/{k} ===")

        train_data = all_data[train_idx].tolist()
        val_data = all_data[val_idx].tolist()

        # Tạo file tạm cho fold
        fold_train_path = f"./dataset/{opt.dataset}/fold_train_{fold}.json"
        fold_val_path = f"./dataset/{opt.dataset}/fold_val_{fold}.json"
        with open(fold_train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
        with open(fold_val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2)

        # Cập nhật lại đường dẫn trong opt
        opt_fold = argparse.Namespace(**vars(opt))  # deepcopy đơn giản
        opt_fold.dataset_file = {
            'train': fold_train_path,
            'test': fold_val_path
        }

        # Huấn luyện
        setup_seed(opt_fold.seed)
        instructor = Instructor(opt_fold)
        instructor.run()

        # Ghi lại kết quả fold
        eval_result = instructor._evaluate()
        accs.append(eval_result['test_acc'])
        f1s.append(eval_result['f1'])

        # Dọn file tạm nếu cần
        # os.remove(fold_train_path)
        # os.remove(fold_val_path)

    print("\n=== K-Fold Cross Validation Result ===")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', default=5, type=int, help="Number of folds")
    parser.add_argument('--train_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Parse args dùng chung với train.py
    sys_argv = ['train.py'] + (args.train_args if args.train_args else [])
    import sys
    sys.argv = sys_argv
    from train import main as train_main, argparse as train_argparse

    opt = train_argparse.parse_args()
    opt.eval = False  # đảm bảo không chỉ chạy test
    kfold_cross_validation(opt, k=args.kfold)
