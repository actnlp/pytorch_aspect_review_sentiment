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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset', help='Dataset directory')
parser.add_argument('--vocab_dir', type=str, default='dataset', help='Vocab directory')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--top_asp', type=float, default=1, help='Only use top asp for training.')

parser.add_argument('--rnn_hidden', type=int, default=150, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=3, help='Number of RNN layers.')
parser.add_argument('--rnn_bidirect', default=True, help='Do use bidirect RNN.')
parser.add_argument('--input_dropout', type=float, default=0.6, help='input dropout rate.')
parser.add_argument('--rnn_dropout', type=float, default=0., help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.003, help='Applies to sgd and adagrad.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=300, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=24, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
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

# load vocab id2word -> list, word2id -> dict
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
asp_emb_matrix = np.load(opt['vocab_dir'] + '/asp_embedding.npy')
considered = int(len(constant.ASP_TO_ID)*opt['top_asp'])
asp_emb_matrix = asp_emb_matrix[0:considered]

assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']
assert asp_emb_matrix.shape[1] == opt['emb_dim']

# model save dir
helper.ensure_dir(opt['save_dir'], verbose=True)

# save config
helper.save_config(opt, opt['save_dir'] + '/config.json', verbose=True)
vocab.save(opt['save_dir'] + '/vocab.pkl')
file_logger = helper.FileLogger(opt['save_dir'] + '/' + opt['log'], header="# epoch\ttrain_loss\ttest_loss\tP\tR\tF1")

# load data
print("Loading data from {} with batch size {} ...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.list', opt['batch_size'], opt, vocab)
dev_batch = DataLoader(opt['data_dir'] + '/test.list', opt['batch_size'], opt, vocab)

print('Building model...')
trainer = MyTrainer(opt, emb_matrix, asp_emb_matrix)
current_lr = opt['lr']

# start training
train_loss_his, test_loss_his, P_his, R_his, F1_his = [], [], [], [], []
for epoch in range(1, args.num_epoch+1):
    train_loss, train_step = 0., 0
    for i, batch in enumerate(train_batch):
        loss = trainer.update(batch)
        train_loss += loss
        train_step += 1
        if train_step % args.log_step == 0:
            print("train_loss: {}".format(train_loss/train_step))

    # eval on dev
    print("Evaluating on dev set...")
    test_loss, test_step = 0., 0
    right_num, logits_num, label_num = 0., 0., 0.
    for i, batch in enumerate(dev_batch):
        loss, tmp_r, tmp_lo, tmp_la = trainer.predict(batch)
        test_loss += loss
        right_num += tmp_r
        logits_num += tmp_lo
        label_num += tmp_la
        test_step += 1
    if logits_num == 0:
        P = 0
    else:
        P = right_num / logits_num
    if label_num == 0:
        R = 0
    else:
        R = right_num / label_num
    if P+R == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P+R)
    print("trian_loss: {}, test_loss: {}, P: {}, R: {}, F1: {}".format( \
        train_loss/train_step, test_loss/test_step, P, R, F1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format( \
        epoch, train_loss/train_step, test_loss/test_step, P, R, F1))

    train_loss_his.append(train_loss/train_step)
    test_loss_his.append(test_loss/test_step)
    P_his.append(P)
    R_his.append(R)

    # save
    #model_file = opt['save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
    #trainer.save(model_file)
    # save best model
    if epoch == 1 or F1 > max(F1_his):
        #copyfile(model_file, opt['save_dir'] + '/best_model.pt')
        print("new best model saved.")
        print("")
        file_logger.log("new best model saved at epoch {}: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}"\
            .format(epoch, train_loss/train_step, test_loss/test_step, P, R, F1))
    F1_his.append(F1)

print("Training ended with {} epochs.".format(epoch))
bt_train_loss = min(train_loss_his)
bt_F1 = max(F1_his)
bt_test_loss = min(test_loss_his)
print("best train_loss: {}, best test_loss: {}, best results: P: {}, R:{}, F1:{}".format( \
    bt_train_loss, bt_test_loss, P_his[F1_his.index(bt_F1)], R_his[F1_his.index(bt_F1)], bt_F1))
    #bt_train_loss, bt_test_loss, bt_P, R_his[P_his.index(bt_P)], F1_his[P_his.index(bt_P)]))
of = open('tmp.txt','a')
of.write(str(bt_train_loss)+","+str(bt_test_loss)+","+str(P_his[F1_his.index(bt_F1)])+str(R_his[F1_his.index(bt_F1)])+str(bt_F1)+'\n')
#of.write(str(bt_train_loss)+","+str(bt_test_loss)+","+str(bt_P)+str(R_his[P_his.index(bt_P)])+str(F1_his[P_his.index(bt_P)])+'\n')
of.close()
