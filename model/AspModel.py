import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import constant, torch_utils

class AspModel(nn.Module):
    def __init__(self, opt, emb_matrix=None, asp_emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.asp_emb_matrix = asp_emb_matrix
        considered = int(len(constant.ASP_TO_ID)*opt['top_asp'])

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.asp_emb = nn.Embedding(considered, opt['emb_dim'])
        # bilstm 
        self.rnn_model = LSTMRelationModel(opt)
        # NN
        self.l1 = nn.Linear(opt['rnn_hidden']*2, opt['emb_dim'])
        
        self.input_dropout = nn.Dropout(opt['input_dropout'])
        self.classifier = nn.Linear(opt['emb_dim'], len(constant.LABEL_TO_ID))

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.asp_emb_matrix is None:
            self.asp_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.asp_emb_matrix = torch.from_numpy(self.asp_emb_matrix)
            self.asp_emb.weight.data.copy_(self.asp_emb_matrix)
        # top N embeddings is finetuned
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        # unpack inputs 
        tokens, mask_s = inputs

        # ipnut embs
        tokens_embs = self.emb(tokens)
        lens = mask_s.sum(dim=1)

        # drop out
        rnn_inputs = self.input_dropout(tokens_embs)

        # RNNs forward
        rnn_outputs = self.rnn_model((rnn_inputs, lens)) # [batch_size, len, rnn_hidden*2]

        # NN forward
        rnn_outputs = F.relu(self.l1(rnn_outputs))       # [batch_size, len, emb_dim]

        # attention
        att_q = self.asp_emb.weight.unsqueeze(0).repeat(rnn_outputs.size(0), 1, 1).transpose(1, 2)
        att_m = rnn_outputs.bmm(att_q)                          # [batch_size, len, asp_num]
        mask_s = mask_s.unsqueeze(-1).repeat(1, 1, att_m.size(2))
        att_m = torch.where(mask_s==0, torch.zeros_like(att_m)-10e10, att_m)
        att_m = F.softmax(att_m, dim=1).transpose(1, 2)         # [batch_size, asp_num, len]
        c_inputs = att_m.bmm(rnn_outputs)                       # [batch_size, asp_num, rnn_hidden*2]
        
        # logits 
        logits = self.classifier(c_inputs)
        
        return logits
        

# BiLSTM model 
class LSTMRelationModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.in_dim = opt['emb_dim']
        self.rnn = nn.LSTM(self.in_dim, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                               dropout=opt['rnn_dropout']*int(opt['rnn_layers']>1), bidirectional=opt['rnn_bidirect'])
        self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # drop the last layer

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'], self.opt['rnn_bidirect'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        # unpack inputs
        inputs, lens = inputs[0], inputs[1]
        return self.rnn_drop(self.encode_with_rnn(inputs, lens, inputs.size()[0]))

# Initialize zero state
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
