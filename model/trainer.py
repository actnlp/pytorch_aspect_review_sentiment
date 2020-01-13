"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.AspModel import AspModel
from utils import torch_utils, constant
from utils.scorer import sta

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

# 0: tokens, 1: mask_s, 2: label
def unpack_batch(batch, cuda):
    inputs, label = batch[0:2], batch[2]
    if cuda:
        inputs = [Variable(i.cuda()) for i in inputs]
        label = Variable(label.cuda())
    else:
        inputs = [Variable(i) for i in inputs]
        label = Variable(label)
    return inputs, label

# 0: tokens, 2: mask_s, 3: label
class MyTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None, asp_emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = AspModel(opt, emb_matrix, asp_emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        if opt['cuda']:
            self.model.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits.view(-1, len(constant.LABEL_TO_ID)), label.view(-1), reduction='sum') / label.size(0)
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        # loss value
        loss_val = loss.item()
        return loss_val

    def predict(self, batch):
        inputs, label = unpack_batch(batch, self.opt['cuda'])
        # forward
        self.model.eval()
        logits = self.model(inputs)
        # loss 
        loss = F.cross_entropy(logits.view(-1, len(constant.LABEL_TO_ID)), label.view(-1), reduction='sum') / label.size(0)
        # predict result
        right_num, logits_num, label_num = sta(logits, label)
        loss_val = loss.item()
        return loss_val, right_num, logits_num, label_num
