import os
import sys
import numpy as np
import random
import argparse
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import constant, helper
from data.loader import DataLoader
from model.trainer import MyTrainer
from utils.vocab import Vocab

import jieba

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
parser.add_argument('--vocab_dir', type=str, default='dataset', help='Vocab directory')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--top5', default=True, help='Only use Top5 for training.')
################################################################################################ inserted this line
parser.add_argument('--top_asp', type=float, default=1, help='Only use top asp for training.')
################################################################################################

parser.add_argument('--rnn_hidden', type=int, default=150, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_bidirect', default=True, help='Do use bidirect RNN.')
parser.add_argument('--input_dropout', type=float, default=0.8, help='input dropout rate.')
parser.add_argument('--rnn_dropout', type=float, default=0., help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.003, help='Applies to sgd and adagrad.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=400, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=24, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=10, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# print opt
helper.print_config(opt)

helper.ensure_dir(opt['save_dir'], verbose=True)

vocab_file = opt['save_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file)
opt['vocab_size'] = vocab.size
opt = helper.load_config(opt['save_dir'] + '/config.json', verbose=True)

if not torch.cuda.is_available():
    opt['cuda'] = False

# load data
print("Loading data from {} with batch size {} ...".format(opt['data_dir'], opt['batch_size']))



dev_batch = DataLoader(eval(open(opt['data_dir']+'/test.list', 'r', encoding='utf-8').read()), opt['batch_size'], opt, vocab)


print('Building model...')
trainer = MyTrainer(opt)
current_lr = opt['lr']

model_file = os.path.join(opt['save_dir'], 'best_model.pt')
trainer.load(model_file)

# eval on dev
print("Evaluating on dev set...")
test_loss, test_step = 0., 0
right_num, logits_num, label_num = 0., 0., 0.
for i, (batch, _) in enumerate(dev_batch):
    loss, tmp_r, tmp_lo, tmp_la = trainer.evaluate(batch)
    test_loss += loss
    right_num += tmp_r
    logits_num += tmp_lo
    label_num += tmp_la
    test_step += 1
P = right_num / logits_num if logits_num else 0
R = right_num / label_num if label_num else 0
F1 = 2 * P * R / (P+R) if P+R else 0

print("test_loss: {}, P: {}, R: {}, F1: {}".format(test_loss/test_step, P, R, F1))

def predict(sentences):
    test_data = list()
    for sent in sentences:
        tokens = jieba.lcut(sent, cut_all=False)
        test_data.append({'text': tokens, 'aspects': [constant.ID_TO_ASP[0]], 'polarities': [constant.ID_TO_LABEL[0]]})
    test_batch = DataLoader(test_data, opt['batch_size'], opt, vocab)
    print("Predicting on test set...")
    labels = list()
    for i, (batch, indices) in enumerate(test_batch):
        predicts = trainer.predict(batch)
        labels += [predicts[k] for k in indices]
    results = list()
    for i, label in enumerate(labels):
        aspects = [x1 for x1, x2 in label]
        polarities = [x2 for x1, x2 in label]
        results.append({'text': test_data[i]['text'], 'aspects': aspects, 'polarities': polarities})
    return results

if __name__ == '__main__':
    sentences = eval(open('test_sentences.txt', 'r', encoding='utf-8').read())
    results = predict(sentences)
    open('results.txt', 'w', encoding='utf-8').write(str(results))
    
    
