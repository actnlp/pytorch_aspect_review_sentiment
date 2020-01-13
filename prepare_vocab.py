"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
import random
from collections import Counter
import jieba

from utils import constant, helper

random.seed(1234)
np.random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab.')
    parser.add_argument('--data_dir', default='dataset', help='dataset directory.')
    parser.add_argument('--vocab_dir', default='dataset', help='Output vocab directory.')
    parser.add_argument('--w2v_dir', default='pretrained_emb', help='pretrained emb directory.')
    parser.add_argument('--wv_file', default='sgns.baidubaike.bigram-char', help='W2V file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='word vector dimension.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.list'
    test_file = args.data_dir + '/test.list'
    wv_file = args.w2v_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    test_tokens = load_tokens(test_file)

    # load glove
    print("loading word vector...")
    glove_vocab = load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens+test_tokens+constant.ASP_TOKEN, glove_vocab)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("building asp embeddings...")
    w2id = {w: i for i, w in enumerate(v)}
    ASP_TO_ID = constant.ASP_TO_ID
    asp_emb = np.random.uniform(-1, 1, (len(ASP_TO_ID), wv_dim))
    for key in ASP_TO_ID.keys():
        ts = list(jieba.cut(key))
        tmp = np.zeros(wv_dim)
        for t in ts:
            if t not in w2id:
                tmp += embedding[w2id['<UNK>']]
                print(t)
            else:
                tmp += embedding[w2id[t]]
        asp_emb[ASP_TO_ID[key]] = tmp / len(ts)
    print("embedding size: {} x {}".format(*asp_emb.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    np.save(args.vocab_dir+'/asp_embedding.npy', asp_emb)
    print("all done.")

def load_glove_vocab(file, wv_dim):
    """
    Load all words from glove.
    """
    vocab = set()
    with open(file, encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:1])
            vocab.add(token)
    return vocab

def load_tokens(filename):
    data = eval(open(filename, 'r').read())
    tokens = []
    for d in data:
        ts = d['text']
        tokens += list(ts)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, glove_vocab):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # sort words according to its freq
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[constant.PAD_ID] = 0 # pad vector
    w2id = {w: i for i, w in enumerate(vocab)}
    with open(wv_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            word_end = len(elems) - wv_dim
            token = ''.join(elems[0:word_end])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[word_end:len(elems)]]
    return emb

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

if __name__ == '__main__':
    main()


