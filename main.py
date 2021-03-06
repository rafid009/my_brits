import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
from models.brits import BRITSModel as BRITS
from models.brits_i import BRITSModel as BRITS_I
import argparse
import data_loader
import pandas as pd
from tqdm import tqdm
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
# BRITS -> 4000
# BRITS_I -> 6000
n_epochs = 1000

# BRITS_I
RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.5
LABEL_WEIGHT = 1

model_name = 'BRITS'
model_path_name = 'BRITS'
model_path = 'model_'+model_path_name+'_LT.model'

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data_iter = data_loader.get_loader(batch_size=batch_size, filename='./json/json_LT')

    
    for epoch in range(n_epochs):
        model.train()
        with tqdm(data_iter, unit='batch') as tepoch:
            run_loss = 0.0
            tepoch.set_description(f"Epoch {epoch+1}/{n_epochs} [T]")
            for idx, data in enumerate(data_iter):
                data = utils.to_var(data)
                ret = model.run_on_batch(data, optimizer, epoch)

                run_loss += ret['loss'].item()
                tepoch.set_postfix(train_loss=(run_loss / (idx + 1.0)))
                tepoch.update(1)
                # print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

            mre, mse = evaluate(model, data_iter)
            tepoch.set_postfix(MSE=mse, MRE=mre)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), model_path)


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
        # save_label.append(ret['labels'].data.cpu().numpy())

        # pred = ret['predictions'].data.cpu().numpy()
        # label = ret['labels'].data.cpu().numpy()
        # is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()


    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    # mae = np.abs(evals - imputations).mean()
    # print('MAE: ', mae)
    mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    # print('MRE: ', mre)

    mse = ((evals - imputations) ** 2).mean()
    # print('MSE: ', mse)
    save_impute = np.concatenate(save_impute, axis=0)
    # save_label = np.concatenate(save_label, axis=0)
    if not os.path.isdir('./result/'):
        os.makedirs('./result/')
    np.save('./result/data_LT', save_impute)
    # np.save('./result/label', save_label)
    return mre, mse


if model_name == 'BRITS':
    model = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)
else:
    model = BRITS_I(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

if torch.cuda.is_available():
    model = model.cuda()

train(model)



