import os
import re
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from sklearn import metrics
from models.masgcn import MASGCNClassifier
from models.masgcn_bert import MASGCNBertClassifier
from models.bilstm import BILSTMClassifier
from models.CNN import CNNClassifier
from models.trans import TransformerClassifier
from models.phobert import PhoBERTClassifier
from models.tnet import TNETClassifier
from models.dualgcn import DualGCNClassifier
from models.senticgcn import SenticGCNClassifier
from models.dualbert import DualGCNBertClassifier
from models.atae_lstm import ATAELSTMClassifier
from utils.data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp
from torch.optim.lr_scheduler import StepLR, LinearLR
import torch.backends.cudnn
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from torch.nn.functional import one_hot

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
os.makedirs('./log', exist_ok=True)
model_classes = {
        'masgcn': MASGCNClassifier,
        'masgcnbert': MASGCNBertClassifier,
        'bilstm': BILSTMClassifier,
        'cnn': CNNClassifier,
        'phobert': PhoBERTClassifier,
        'trans': TransformerClassifier,
        'atae': ATAELSTMClassifier,
        'tnet': TNETClassifier,
        'dualgcn': DualGCNClassifier,
        'senticgcn': SenticGCNClassifier,
        'dualbert': DualGCNBertClassifier
    }

dataset_files = {
        
        'Restaurants_corenlp': {
            'train': './dataset/Restaurants_corenlp/train_preprocessed.json',
            'test': './dataset/Restaurants_corenlp/test_preprocessed.json',
        },
        'Laptops_corenlp': {
            'train': './dataset/Laptops_corenlp/train_preprocessed.json',
            'test': './dataset/Laptops_corenlp/test_preprocessed.json'
        },
        'Movie_vietnamese': {
            'train': './dataset/Movie_vietnamese/train_preprocessed.json',
            'test': './dataset/Movie_vietnamese/test_preprocessed.json'
        },
    }

input_colses = {
        'cnn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask', 'syn_dep_adj'],
        'trans': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask', 'syn_dep_adj'],
        'bilstm': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask', 'syn_dep_adj'],
        'atae': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask', 'syn_dep_adj'],
        'dualgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj'],
        'senticgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask', 'syn_dep_adj'],
        'tnet': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask', 'syn_dep_adj'],
        'phobert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'deprel', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask', 'short_mask', 'syn_dep_adj'],
        'dualbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'deprel', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask', 'short_mask', 'syn_dep_adj'],
        'masgcn': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask', 'syn_dep_adj'],
        'masgcnbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'deprel', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask', 'short_mask', 'syn_dep_adj']
    }
initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
MIN_ACC = {
        'cnn':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'trans':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'atae':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'tnet':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'dualgcn':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'senticgcn':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'bilstm':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'phobert':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'dualbert':{'Laptops_corenlp': 0.50, 'Restaurants_corenlp': 0.50, 'Movie_vietnamese': 0.50},
        'masgcn':{'Laptops_corenlp': 0. , 'Restaurants_corenlp': 0.83, 'Movie_vietnamese': 0.75},
        'masgcnbert': {'Laptops_corenlp': 0.81, 'Restaurants_corenlp': 0.86, 'Movie_vietnamese': 0.77}
    }
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Instructor:
    ''' Model training and evaluation '''
    def get_bert_optimizer(self, model):
     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
     model_param = list(model.named_parameters())
     optimizer_grouped_parameters = []

     if self.opt.diff_lr:
        logger.info("Using differential learning rate for BERT and other layers.")
        bert_params = [(n, p) for n, p in model_param if 'bert' in n]
        other_params = [(n, p) for n, p in model_param if 'bert' not in n]

        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay,
                'lr': self.opt.bert_lr,
            },
            {
                'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.opt.bert_lr,
            },
            {
                'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay,
                'lr': self.opt.learning_rate,
            },
            {
                'params': [p for n, p in other_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.opt.learning_rate,
            },
        ]
     else:
        logger.info("Using same learning rate for all parameters.")
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in model_param if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay,
                'lr': self.opt.bert_lr,
            },
            {
                'params': [p for n, p in model_param if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.opt.bert_lr,
            },
        ]

     return AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')
            opt.dep_size = len(dep_vocab)
            tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_length=opt.max_length,
                data_file='{}/tokenizer.dat'.format(opt.vocab_dir))
            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab,
                embed_dim=opt.embed_dim,
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')


            logger.info("token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
                len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)))

            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)
            opt.dep_size = len(dep_vocab)
            opt.vocab_size = len(token_vocab)
            opt.word2idx = token_vocab.stoi

            vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

        self.train_dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {}, n_nontrainable_params: {}'.format(
            n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {}: {}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _evaluate(self, show_results=False):
     self.model.eval()
     n_test_correct, n_test_total = 0, 0
     targets_all, outputs_all = [], []

     with torch.no_grad():
        for sample_batched in self.test_dataloader:
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            targets = sample_batched['polarity'].to(self.opt.device)
            outputs, _ = self.model(inputs)

            n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_test_total += len(outputs)
            targets_all.append(targets)
            outputs_all.append(outputs)

     targets_all = torch.cat(targets_all, dim=0)
     outputs_all = torch.cat(outputs_all, dim=0)
     preds = torch.argmax(outputs_all, -1).cpu()
     labels = targets_all.cpu()

    # ✅ Phát hiện nhãn động
     unique_labels = sorted(np.unique(labels.numpy()))

     test_acc = n_test_correct / n_test_total
     f1 = metrics.f1_score(labels, preds, labels=unique_labels, average='macro', zero_division=0)
     precision = precision_score(labels, preds, labels=unique_labels, average='macro', zero_division=0)
     recall = recall_score(labels, preds, labels=unique_labels, average='macro', zero_division=0)

     if show_results:
        logger.info(f"Unique labels detected: {unique_labels}")

     report = metrics.classification_report(labels, preds, labels=unique_labels, digits=4) if show_results else None
     confusion = metrics.confusion_matrix(labels, preds, labels=unique_labels) if show_results else None

     return {
        'report': report,
        'confusion': confusion,
        'test_acc': test_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


    def _train(self, criterion, optimizer, max_test_acc_overall=0, lr_schedule=None):
        max_test_acc = 0
        max_f1 = 0
        model_path = ''
        global_step = 0

        for epoch in range(self.opt.num_epoch):
            logger.info('=' * 60)
            logger.info('Epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0

            for sample_batched in self.train_dataloader:
                global_step += 1
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, penal = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                num_classes = self.opt.polarities_dim
                if (targets < 0).any() or (targets >= num_classes).any():
                   invalid = targets[(targets < 0) | (targets >= num_classes)]
                   logger.error(f"[Label Error] Invalid target labels found: {invalid.tolist()} | Expected: 0 ≤ label < {num_classes}")
                   raise ValueError("Invalid label detected. Please check dataset preprocessing.")
                

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    eval_result = self._evaluate()
                    test_acc = eval_result['test_acc']
                    f1 = eval_result['f1']
                    precision = eval_result['precision']
                    recall = eval_result['recall']

                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(
                                self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> Saved: {}'.format(model_path))

                    if f1 > max_f1:
                        max_f1 = f1

                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
                        loss.item(), train_acc, test_acc, f1, precision, recall))

            if lr_schedule:
                lr_schedule.step()

        return max_test_acc, max_f1, model_path

    def _test(self):
        self.model = self.best_model
        eval_result = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score Report:")
        logger.info(eval_result['report'])
        logger.info("Confusion Matrix:")
        logger.info(eval_result['confusion'])
        logger.info('max_test_acc: {:.4f}, max_f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
            eval_result['test_acc'], eval_result['f1'], eval_result['precision'], eval_result['recall']))

    def run(self):
        if self.opt.eval:
            self.best_model = self.model
            pattern = re.compile(rf'{self.opt.model_name}_{self.opt.dataset}_acc_([0-9.]+)_.*')
            file_list = os.listdir('./state_dict')
            matched_file = [(f, m.group(1)) for f in file_list if (m := pattern.match(f))]

            if not matched_file:
                raise FileNotFoundError("No checkpoint file found matching model/dataset pattern.")
            max_acc_file, _ = max(matched_file, key=lambda x: float(x[1]))
            # Load best model weights
            self.best_model.load_state_dict(torch.load(f'./state_dict/{max_acc_file}', map_location=self.opt.device))
            self._test()
            return

        # Start training
        criterion = nn.CrossEntropyLoss()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            lr_schedule = LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=20)
        else:
            optimizer = self.get_bert_optimizer(self.model)
            lr_schedule = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=15)

        max_test_acc_overall, max_f1_overall = 0, 0

        if 'bert' not in self.opt.model_name:
            self._reset_params()

        max_test_acc, max_f1, model_path = self._train(
            criterion, optimizer, max_test_acc_overall, lr_schedule)

        logger.info('Training Completed.')
        logger.info('Best test accuracy: {:.4f}, F1: {:.4f}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc_overall, max_test_acc)
        max_f1_overall = max(max_f1_overall, max_f1)

        if max_test_acc_overall > self.opt.min_acc:
            torch.save(self.best_model.state_dict(), model_path)
            logger.info('>> Final model saved: {}'.format(model_path))

        logger.info('#' * 60)
        logger.info('max_test_acc_overall: {:.4f}'.format(max_test_acc_overall))
        logger.info('max_f1_overall: {:.4f}'.format(max_f1_overall))

        # Final test
        self._test()



def main():
    

    parser = get_parser()
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.vocab_dir = f'./dataset/{opt.dataset}'
    if 'bert' not in opt.model_name:
        opt.rnn_hidden = opt.hidden_dim
        opt.min_acc = MIN_ACC[opt.model_name][opt.dataset]
    else:
        opt.min_acc = MIN_ACC[opt.model_name][opt.dataset]
    
    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'cpu') if opt.device is None else torch.device(opt.device)

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./log'):
     os.makedirs('./log', mode=0o777)

   # Use valid filename format (no colons)
    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, timestamp)

    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    ins = Instructor(opt)
    ins.run()

def get_parser():
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
    return parser
if __name__ == '__main__':
    main()
