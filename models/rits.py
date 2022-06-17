import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import numpy as np

SEQ_LEN = 252
feature_len = 13

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

def get_not_nan(y):
    return np.argwhere(np.isnan(y) == False) 

def loss_mse(input, target, reduce=True, size_average=True):
    if not reduce:
        loss_fn = nn.MSELoss(reduction='none')
    elif size_average:
        loss_fn = nn.MSELoss(reduction='mean')
    else:
        loss_fn = nn.MSELoss(reduction='sum')
    loss = loss_fn(input, target)
    if reduce:
        return loss.item()
    else:
        return loss

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class RITSModel(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight, label_weight, feature_len=feature_len):
        super(RITSModel, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.feature_len = feature_len

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.feature_len * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = self.feature_len, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.feature_len, output_size = self.feature_len, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.feature_len)
        self.feat_reg = FeatureRegression(self.feature_len)

        self.weight_combine = nn.Linear(self.feature_len * 2, self.feature_len)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, SEQ_LEN)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        # y_h = self.out(h)
        # y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        # y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)
        # not_nan = get_not_nan(labels)
        # y_p = y_h.view(-1, 1)
        # y_loss = loss_mse(y_p[not_nan[0], not_nan[1]], labels[not_nan[0], not_nan[1]])

        # y_h = F.sigmoid(y_h)
        #+ y_loss * self.label_weight
        # 'predictions': y_h
        # x_loss * self.impute_weight
        return {'loss': x_loss, 'predictions': None,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
