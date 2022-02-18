import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
from models.brits_i import BRITSModel
import argparse
import data_loader
import pandas as pd

from sklearn import metrics


# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', type=int, default=1000)
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--model', type=str)
# parser.add_argument('--hid_size', type=int)
# parser.add_argument('--impute_weight', type=float)
# parser.add_argument('--label_weight', type=float)
# args = parser.parse_args()

batch_size = 16
n_epochs = 50

# BRITS_I
RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.3
LABEL_WEIGHT = 1

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data_iter = data_loader.get_loader(batch_size=batch_size)

    for epoch in range(n_epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

        evaluate(model, data_iter)


def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    save_impute = []
    save_label = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    print('AUC {}'.format(metrics.roc_auc_score(labels, preds)))

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print('MAE', np.abs(evals - imputations).mean())

    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())

    save_impute = np.concatenate(save_impute, axis=0)
    save_label = np.concatenate(save_label, axis=0)

    np.save('./result/data', save_impute)
    np.save('./result/label', save_label)



model = BRITSModel(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)#getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Total params is {}'.format(total_params))

if torch.cuda.is_available():
    model = model.cuda()

train(model)
evaluate(model)



