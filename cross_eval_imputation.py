import copy
from genericpath import isdir
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from process_data import *
# from input_process import *
import json
import os
import torch
from models.brits import BRITSModel as BRITS
import utils
import data_loader
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time
from naomi.model import *
from naomi.helpers import run_test
from tqdm import tqdm
from transformer.src.transformer import run_transformer, add_season_id_and_save
import warnings
import matplotlib
from pypots.data import mcar, masked_fill
from saits.custom_saits import SAITS
from pypots.utils.metrics import cal_mae, cal_mse
import pickle
from linear_imputation import impute, Imputer
import torch.optim as optim
from input_process import prepare_brits_input
warnings.filterwarnings("ignore")
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import sys
np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features = [
    'MEAN_AT',
    'MIN_AT',
    'AVG_AT',
    'MAX_AT',
    'MIN_REL_HUMIDITY',
    'AVG_REL_HUMIDITY',
    'MAX_REL_HUMIDITY',
    'MIN_DEWPT',
    'AVG_DEWPT',
    'MAX_DEWPT',
    'P_INCHES', # precipitation
    'WS_MPH', # wind speed. if no sensor then value will be na
    'MAX_WS_MPH', 
    'LW_UNITY', # leaf wetness sensor
    'SR_WM2', # solar radiation # different from zengxian
    'MIN_ST8', # diff from zengxian
    'ST8', # soil temperature # diff from zengxian
    'MAX_ST8', # diff from zengxian
    #'MSLP_HPA', # barrometric pressure # diff from zengxian
    'ETO', # evaporation of soil water lost to atmosphere
    'ETR', # ???
    'LTE50'
]

seasons_to_idx = {
    '1988-1989': 0,
    '1989-1990': 1,
    '1990-1991': 2,
    '1991-1992': 3,
    '1992-1993': 4,
    '1993-1994': 5,
    '1994-1995': 6,
    '1995-1996': 7,
    '1996-1997': 8,
    '1997-1998': 9,
    '1998-1999': 10,
    '1999-2000': 11,
    '2000-2001': 12,
    '2001-2002': 13,
    '2002-2003': 14,
    '2003-2004': 15,
    '2004-2005': 16,
    '2005-2006': 17,
    '2006-2007': 18,
    '2007-2008': 19,
    '2008-2009': 20,
    '2009-2010': 21,
    '2010-2011': 22,
    '2011-2012': 23,
    '2012-2013': 24,
    '2013-2014': 25,
    '2014-2015': 26,
    '2015-2016': 27,
    '2016-2017': 28,
    '2017-2018': 29,
    '2018-2019': 30,
    '2019-2020': 31,
    '2020-2021': 32,
    '2021-2022': 33,
}

idx_to_seasons = [
    '1988-1989',
    '1989-1990',
    '1990-1991',
    '1991-1992',
    '1992-1993',
    '1993-1994',
    '1994-1995',
    '1995-1996',
    '1996-1997',
    '1997-1998',
    '1998-1999',
    '1999-2000',
    '2000-2001',
    '2001-2002',
    '2002-2003',
    '2003-2004',
    '2004-2005',
    '2005-2006',
    '2006-2007',
    '2007-2008',
    '2008-2009',
    '2009-2010',
    '2010-2011',
    '2011-2012',
    '2012-2013',
    '2013-2014',
    '2014-2015',
    '2015-2016',
    '2016-2017',
    '2017-2018',
    '2018-2019',
    '2019-2020',
    '2020-2021',
    '2021-2022'
]


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(masks.shape[0]):
        if h == 0:
            deltas.append(np.ones(len(features)))
        else:
            deltas.append(np.ones(len(features)) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).to_numpy()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec

def get_minimum_missing_season(df, feature, season_array):
    min_missing_season = None
    min_missing_size = 999999
    season_idx = -1
    for i, season in enumerate(season_array):
        x = (df[feature].loc[season]).to_numpy()
        # print(f"x in min miss season: {x.shape}")
        missing = np.isnan(x).sum()
        if min_missing_size > missing:
            min_missing_size = missing
            min_missing_season = season
            season_idx = i
    return min_missing_season, season_idx

def unnormalize(X, mean, std, feature_idx=-1):
    if feature_idx == -1:
        return (X * std) + mean
    else:
        return (X * std[feature_idx]) + mean[feature_idx]

def parse_id(fs, x, y, mean, std, feature_impute_idx, length, trial_num=-1, dependent_features=None, real_test=True, random_start=False, random=False, pad=-1):
    if random:
        idx1 = np.where(~np.isnan(x[:,feature_impute_idx]))[0]
        indices = np.random.choice(idx1[:-pad], int(len(idx1[:-pad]) * 0.2), replace=False)
        indices = indices * len(features) + feature_impute_idx

        if dependent_features is not None and len(dependent_features) != 0:
            inv_indices = (indices - feature_impute_idx)//len(features)
            for i in inv_indices:
                x[i, dependent_features] = np.nan 
    elif real_test:
        idx1 = np.where(~np.isnan(x[:,feature_impute_idx]))[0]
        idx1 = idx1 * len(features) + feature_impute_idx

        indices = idx1.tolist()
        if random_start:
            start_idx = np.random.choice(indices, 1).item()
        elif trial_num == -1:
            if len(indices) != 0:
                start_idx = indices[len(indices)//4]#np.random.choice(indices, 1)
            else:
                return []
        else:
            start_idx = indices[trial_num] #np.random.choice(indices, 1)
        start = indices.index(start_idx)
        if start + length <= len(indices): 
            end = start + length
        else:
            end = len(indices)
        indices = np.array(indices)[start:end]

        if dependent_features is not None and len(dependent_features) != 0:
            inv_indices = (indices - feature_impute_idx)//len(features)
            for i in inv_indices:
                x[i, dependent_features] = np.nan 
    else:
        indices = None
    evals = x

    evals = (evals - mean) / std
    # print('eval: ', evals)
    # print('eval shape: ', evals.shape)
    shp = evals.shape
    evals = evals.reshape(-1)

    values = evals.copy()
    if indices is not None:
        values[indices] = np.nan

    masks = ~np.isnan(values)

    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)
    label = y.tolist() #out.loc[int(id_)]
    # print(f'rec y: {list(y)}')
    rec = {'label': label}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return indices, values



def train(model, n_epochs, batch_size, model_path, data_file='./json/json_LT'):
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    data_iter = data_loader.get_loader(batch_size=batch_size, filename=data_file)
    pre_mse = 9999999
    count_diverge = 0
    for epoch in range(n_epochs):
        model.train()
        if count_diverge > 3:
            break
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

            mse = evaluate(model, data_iter)
            if pre_mse < mse:
                count_diverge += 1
            else:
                count_diverge = 0
            tepoch.set_postfix(train_loss=(run_loss / (idx + 1.0)), val_loss=mse)
            tepoch.update(1)
        if (epoch + 1) % 100 == 0 and count_diverge == 0:
            torch.save(model.state_dict(), model_path)
    end = time.time()
    print(f"time taken for training: {end-start}s")
    return model

def evaluate(model, val_iter):
    model.eval()
    evals = []
    imputations = []

    save_impute = []
    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()


    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    mse = ((evals - imputations) ** 2).mean()
    # print('MSE: ', mse)
    save_impute = np.concatenate(save_impute, axis=0)
    if not os.path.isdir('./result/'):
        os.makedirs('./result/')
    np.save('./result/data_LT', save_impute)
    return mse


def train_imputation_model(season_df, season_array, max_length, mean, std, model_name, suffix, model_dir, k=-1):
    print(f"=========== {model_name} Training Ends ===========")
    model_path = f'{model_dir}/model_{model_name}_{suffix}.model'
    if model_name == "BRITS":
        prepare_brits_input(season_df, season_array, max_length, features, mean, std, model_dir)#, complete_seasons)
        batch_size = 16
        n_epochs = 3000
        RNN_HID_SIZE = 64
        IMPUTE_WEIGHT = 0.5
        LABEL_WEIGHT = 1
        model = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT, feature_len=len(features))
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            model = model.cuda()
        train(model, n_epochs, batch_size, model_path, data_file='./json/json_without_LT')
    elif model_name == "SAITS":
        # k = 5
        model_path = f'{model_dir}/model_{model_name}_{suffix}_{k if k != -1 else 0}.model'
        X, Y = split_XY(season_df, max_length, season_array, features)

        for i in range(X.shape[0]):
            X[i] = (X[i] - mean)/std

        # filename = f'{model_dir}/model_{model_name}_{suffix}.model'
        # X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
        # X = masked_fill(X, 1 - missing_mask, np.nan)
        # Model training. This is PyPOTS showtime.
        if k == -1:
            saits = SAITS(n_steps=252, n_features=len(features), n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=300, k=k)
        else:
            saits = SAITS(n_steps=252, n_features=len(features), n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=300, k=k, original=True)
        saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
        pickle.dump(saits, open(model_path, 'wb'))
        # imputation = saits.impute(X, k=k)  # impute the originally-missing values and artificially-missing values
        # mse = cal_mse(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
        # print(f"SAITS Validation MSE: {mse}")
    elif model_name == "MICE":
        normalized_season_df = season_df[features].copy()
        normalized_season_df = (normalized_season_df - mean) /std
        mice_impute = IterativeImputer(random_state=0, max_iter=30)
        mice_impute.fit(normalized_season_df[features].to_numpy())
        # filename = f'{model_dir}/model_{model_name}_{suffix}.model'#synth_{n_random}.model'
        pickle.dump(mice_impute, open(model_path, 'wb'))
    elif model_name == "MVTS":
        params = {
            'config_filepath': None, 
            'output_dir': './transformer/output',
            'data_dir': './transformer/data_dir/',
            'load_model': None,
            'resume': False,
            'change_output': False,
            'save_all': False,
            'experiment_name': f'model_{model_name}_{suffix}', 
            'comment': 'pretraining through imputation', 
            'no_timestamp': False, 
            'records_file': 'Imputation_records.csv', 
            'console': False, 
            'print_interval': 1, 
            'gpu': '0', 
            'n_proc': 1, 
            'num_workers': 0, 
            'seed': None, 
            'limit_size': None, 
            'test_only': None, 
            'data_class': 'agaid', 
            'labels': None, 
            'test_from': None, 
            'test_ratio': 0, 
            'val_ratio': 0.1, 
            'pattern': f'Merlot_{suffix}', 
            'val_pattern': None, 
            'test_pattern': None, 
            'normalization': 'standardization', 
            'norm_from': None, 
            'subsample_factor': None, 
            'task': 'imputation', 
            'masking_ratio': 0.15, 
            'mean_mask_length': 20.0, 
            'mask_mode': 'separate', 
            'mask_distribution': 'geometric', 
            'exclude_feats': None, 
            'mask_feats': [0, 1], 
            'start_hint': 0.0, 
            'end_hint': 0.0, 
            'harden': True, 
            'epochs': 1500, 
            'val_interval': 2, 
            'optimizer': 'Adam', 
            'lr': 0.0009, 
            'lr_step': [1000000], 
            'lr_factor': [0.1], 
            'batch_size': 16, 
            'l2_reg': 0, 
            'global_reg': False, 
            'key_metric': 'loss', 
            'freeze': False, 
            'model': 'transformer', 
            'max_seq_len': 252,#366, 
            'data_window_len': None, 
            'd_model': 128, 
            'dim_feedforward': 256, 
            'num_heads': 8, 
            'num_layers': 3, 
            'dropout': 0.1, 
            'pos_encoding': 'learnable', 
            'activation': 'relu', 
            'normalization_layer': 'BatchNorm'
        }

        data_folder = './transformer/data_dir'
        add_season_id_and_save(data_folder, season_df, season_array, f'ColdHardiness_Grape_Merlot_{suffix}.csv')
        run_transformer(params)


    print(f"=========== {model_name} Training Ends ===========")
    return model_path

def parse_id_imputation(fs, x, y, mean, std, feature_impute_idx, length, trial_num=-1, dependent_features=None, real_test=True, random_start=False, random=False, pad=-1):
    if random:
        idx1 = np.where(~np.isnan(x[:,feature_impute_idx]))[0]
        indices = np.random.choice(idx1[:-pad], int(len(idx1[:-pad]) * 0.2), replace=False)
        indices = indices * len(features) + feature_impute_idx

        if dependent_features is not None and len(dependent_features) != 0:
            inv_indices = (indices - feature_impute_idx)//len(features)
            # if length > 1:
            #     print('inv: ', inv_indices)
            for i in inv_indices:
                x[i, dependent_features] = np.nan
    elif real_test:
        idx1 = np.where(~np.isnan(x[:,feature_impute_idx]))[0]

        idx1 = idx1 * len(features) + feature_impute_idx
        
        # randomly eliminate 10% values as the imputation ground-truth
        # print('not null: ',np.where(~np.isnan(evals)))
        indices = idx1.tolist()
        if random_start:
            start_idx = np.random.choice(indices, 1).item()
        elif trial_num == -1:
            if len(indices) != 0:
                start_idx = indices[len(indices)//4]#np.random.choice(indices, 1)
            else:
                return []
        else:
            start_idx = indices[trial_num] #np.random.choice(indices, 1)
        start = indices.index(start_idx)
        if start + length <= len(indices): 
            end = start + length
        else:
            end = len(indices)
        indices = np.array(indices)[start:end]

        # global real_values
        # real_values = evals[indices]
        if dependent_features is not None and len(dependent_features) != 0:
            inv_indices = (indices - feature_impute_idx)//len(features)
            # if length > 1:
            #     print('inv: ', inv_indices)
            for i in inv_indices:
                x[i, dependent_features] = np.nan 
    else:
        indices = None
    evals = x

    evals = (evals - mean) / std
    # print('eval: ', evals)
    # print('eval shape: ', evals.shape)
    shp = evals.shape
    evals = evals.reshape(-1)

    values = evals.copy()
    if indices is not None:
        values[indices] = np.nan

    masks = ~np.isnan(values)

    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)
    label = y.tolist() #out.loc[int(id_)]
    # print(f'rec y: {list(y)}')
    rec = {'label': label}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return indices, values, evals

def forward_parse_id_day(fs, x, y, mean, std, feature_impute_idx, existing_LT, trial_num=-1, all=False, same=True):

    idx_temp = np.where(~np.isnan(x[:,feature_impute_idx]))[0]
    # print(f"idx1: {idx_temp}")
    # print(f"existing LT: {existing_LT}")
    idx1 = idx_temp * len(features) + feature_impute_idx

    indices = idx1.tolist()

    if trial_num != -1:
        start_idx = indices[(trial_num + existing_LT + 1)] #np.random.choice(indices, 1)
    else:
        start_idx = indices[(existing_LT + 1)]
    # print(f"indices: {indices}")
    start = indices.index(start_idx)
    end = len(indices)
    # print(f"start: {start}, end: {end}")
    indices = np.array(indices)[start:end]

    # global real_values
    # real_values = evals[indices]
    if all:
        # print(f"x: {x.shape}")
        x_copy = x.copy()
        # inv_indices = (indices - feature_impute_idx)//len(features)
        # if length > 1:
        # print('inv: ', inv_indices)
        features_to_nan = [features.index(f) for f in features if features.index(f) != feature_impute_idx and f != 'SEASON_JDAY']
        # print(f'features: {features_to_nan}\nstart: {start_idx}')
        # for i in inv_indices:
        if trial_num != 0 and trial_num != -1:
            x_copy[:idx_temp[trial_num], feature_impute_idx] = np.nan
        features_to_nan
        if trial_num != -1:
            if same:
                x_copy[(idx_temp[trial_num + existing_LT + 1] + 1):, features_to_nan] = np.nan
            else:
                x_copy[idx_temp[trial_num + existing_LT + 1]:, features_to_nan] = np.nan
        else:
            if same:
                x_copy[(idx_temp[existing_LT + 1] + 1):, features_to_nan] = np.nan
            else:
                x_copy[idx_temp[existing_LT + 1], features_to_nan] = np.nan

        evals = x_copy
    else:
        evals = x

    evals = (evals - mean) / std

    shp = evals.shape
    evals = evals.reshape(-1)

    values = evals.copy()
    if indices is not None:
        values[indices] = np.nan
    
    masks = ~np.isnan(values)

    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)
    # print(f"values: {values[~np.isnan(values[:, feature_impute_idx]), feature_impute_idx]}")
    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)
    label = y.tolist() #out.loc[int(id_)]
    # print(f'rec y: {list(y)}')
    rec = {'label': label}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return indices, values, evals


def evaluate_imputation(results, season, season_df, season_array, max_length, models, mean, std, suffix, k):
    out_folder = 'cross_val_imputation_outs'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    out_file = open(f'{out_folder}/cv_{suffix}_{k if k != -1 else 0}.txt', 'w')
    filename = 'json/json_eval_2_LT'
    given_features = [
        'MEAN_AT', # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
        'MIN_AT',
        'AVG_AT', # average temp is AgWeather Network
        'MAX_AT',
        'MIN_REL_HUMIDITY',
        'AVG_REL_HUMIDITY',
        'MAX_REL_HUMIDITY',
        'MIN_DEWPT',
        'AVG_DEWPT',
        'MAX_DEWPT',
        'P_INCHES', # precipitation
        'WS_MPH', # wind speed. if no sensor then value will be na
        'MAX_WS_MPH', 
        'LW_UNITY', # leaf wetness sensor
        'SR_WM2', # solar radiation # different from zengxian
        'MIN_ST8', # diff from zengxian
        'ST8', # soil temperature # diff from zengxian
        'MAX_ST8', # diff from zengxian
        #'MSLP_HPA', # barrometric pressure # diff from zengxian
        'ETO', # evaporation of soil water lost to atmosphere
        'ETR',
        'LTE50' # ???
    ]
    print(f"Season: {season}")
    out_file.write(f"Season: {season}\n")
    results[season] = {}
    for feature in given_features:
        feature_idx = features.index(feature)
        X, Y, pads = split_XY(season_df, max_length, season_array, features, is_pad=True)
        # original_missing_indices = np.where(np.isnan(X[:, :, feature_idx]))[0]
        
        iter = 100#len(season_array[season_idx]) - (l-1) - len(original_missing_indices) - pads[season_idx]
        total_count = 0
        model_mse = {
            # 'BRITS': 0,
            'SAITS': 0,
            # 'MICE': 0,
            # 'MVTS': 0,
            # 'LINEAR': 0
        }
        for i in range(iter):
            fs = open(filename, "w")
            if feature.split('_')[-1] not in feature_dependency.keys():
                dependent_feature_ids = []
            else:
                dependent_feature_ids = [features.index(f) for f in feature_dependency[feature.split('_')[-1]] if (f != feature) and (f in features)]
            
            missing_indices, Xeval, eval = parse_id_imputation(fs, X[0], Y[0], mean, std, feature_idx, -1, i, dependent_feature_ids, random=True, pad=pads[0])
            # print(f"missing idx: {missing_indices}")
            # print(f"missing idx: len: {len(missing_indices)}")
            fs.close()
            if len(missing_indices) == 0:
                break
            # print(f"i: {i}\nmissing indices: {missing_indices}")
            # val_iter = data_loader.get_loader(batch_size=1, filename=filename)

            row_indices = missing_indices // len(features)
            Xeval = np.reshape(Xeval, (1, Xeval.shape[0], Xeval.shape[1]))               
            imputation_saits = models['SAITS'].impute(Xeval, k=k, is_test=True)
            imputation_saits = np.squeeze(imputation_saits)
            imputed_saits = imputation_saits[row_indices, feature_idx]
            real_values = eval[row_indices, feature_idx]

            # for idx, data in enumerate(val_iter):
                # data = utils.to_var(data)
                # row_indices = missing_indices // len(features)
                # # print(f"rows: {row_indices}")
                # ret = models['BRITS'].run_on_batch(data, None)
                # eval_ = ret['evals'].data.cpu().numpy()
                # eval_ = np.squeeze(eval_)
                # imputation_brits = ret['imputations'].data.cpu().numpy()
                # imputation_brits = np.squeeze(imputation_brits)
                # imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)
                # print(f"imputed brits: {imputed_brits}")
                # Xeval = np.reshape(Xeval, (1, Xeval.shape[0], Xeval.shape[1]))
                
                # imputation_saits = models['SAITS'].impute(Xeval)
                # # print(f"SAITS imputation: {imputation_saits}")
                # imputation_saits = np.squeeze(imputation_saits)
                # imputed_saits = imputation_saits[row_indices, feature_idx]
                # print(f"imputed saits: {imputed_saits}")

                # ret_eval = copy.deepcopy(eval_)
                # ret_eval[row_indices, feature_idx] = np.nan
                # imputation_mice = models['MICE'].transform(ret_eval)
                # imputed_mice = imputation_mice[row_indices, feature_idx]
                # print(f"imputed mice: {imputed_mice}")
                
                # ret_eval = copy.deepcopy(eval_)
                # ret_eval = unnormalize(ret_eval, mean, std, feature_idx)
                # ret_eval[row_indices, feature_idx] = np.nan
                # test_df = pd.DataFrame(ret_eval, columns=features)
                # add_season_id_and_save('./transformer/data_dir', test_df, filename='ColdHardiness_Grape_Merlot_test_2.csv')

                # params = {
                #     'config_filepath': None, 
                #     'output_dir': './transformer/output/', 
                #     'data_dir': './transformer/data_dir/', 
                #     'load_model': f'./transformer/output/mvts-{suffix}/checkpoints/model_best.pth', 
                #     'resume': False, 
                #     'change_output': False, 
                #     'save_all': False, 
                #     'experiment_name': 'MVTS_test',
                #     'comment': 'imputation test', 
                #     'no_timestamp': False, 
                #     'records_file': 'Imputation_records.csv', 
                #     'console': False, 
                #     'print_interval': 1, 
                #     'gpu': '0', 
                #     'n_proc': 1, 
                #     'num_workers': 0, 
                #     'seed': None, 
                #     'limit_size': None, 
                #     'test_only': 'testset', 
                #     'data_class': 'agaid', 
                #     'labels': None, 
                #     'test_from': './transformer/test_indices.txt', 
                #     'test_ratio': 0, 
                #     'val_ratio': 0, 
                #     'pattern': None, 
                #     'val_pattern': None, 
                #     'test_pattern': 'Merlot_test', 
                #     'normalization': 'standardization', 
                #     'norm_from': None, 
                #     'subsample_factor': None, 
                #     'task': 'imputation', 
                #     'masking_ratio': 0.15, 
                #     'mean_mask_length': 10.0, 
                #     'mask_mode': 'separate', 
                #     'mask_distribution': 'geometric', 
                #     'exclude_feats': None, 
                #     'mask_feats': [0, 1], 
                #     'start_hint': 0.0, 
                #     'end_hint': 0.0, 
                #     'harden': True, 
                #     'epochs': 1000, 
                #     'val_interval': 2, 
                #     'optimizer': 'Adam', 
                #     'lr': 0.0009, 
                #     'lr_step': [1000000], 
                #     'lr_factor': [0.1], 
                #     'batch_size': 16, 
                #     'l2_reg': 0, 
                #     'global_reg': False, 
                #     'key_metric': 'loss', 
                #     'freeze': False, 
                #     'model': 'transformer', 
                #     'max_seq_len': 252, 
                #     'data_window_len': None, 
                #     'd_model': 128, 
                #     'dim_feedforward': 256, 
                #     'num_heads': 8, 
                #     'num_layers': 3, 
                #     'dropout': 0.1, 
                #     'pos_encoding': 'learnable', 
                #     'activation': 'relu', 
                #     'normalization_layer': 'BatchNorm'
                # }

                # transformer_preds = run_transformer(params)
                # print(f'trasformer preds: {transformer_preds}')
                
                # imputation_transformer = np.squeeze(transformer_preds)
                # imputed_transformer = imputation_transformer[row_indices, feature_idx].cpu().detach().numpy()
                # print(f"imputed mvts: {imputed_transformer}")

                # ret_eval[row_indices, feature_idx] = np.nan

                # ret_eval_df = pd.DataFrame(ret_eval, columns=features)
                # # imputed_linear = ret_eval_df.interpolate(method='linear', limit_direction='both')
                # imputed_linear = impute(ret_eval_df).to_numpy()
                # imputed_linear = imputed_linear[row_indices, feature_idx]
                # print(f"imputd linear: {imputed_linear}")

                # real_values = eval[row_indices, feature_idx]
                # print(f"real: {real_values}")
            print(f"\n\nFor festure = {feature} trial: {i}\nreal: {real_values}\nimputed: {imputed_saits}\n\n")
            out_file.write(f"\n\nFor festure = {feature} trial: {i}\nreal: {real_values}\nimputed: {imputed_saits}\n\n")
            # model_mse['BRITS'] += np.sqrt((real_values - imputed_brits) ** 2).mean()
            model_mse['SAITS'] += np.sqrt((real_values - imputed_saits) ** 2).mean()
            # model_mse['MICE'] += np.sqrt((real_values - imputed_mice) ** 2).mean()
            # model_mse['MVTS'] += np.sqrt((real_values - imputed_transformer) ** 2).mean()
            # model_mse['LINEAR'] += np.sqrt((real_values - imputed_linear) ** 2).mean()
            # print(f"real: {real_values}\nlinear mse: {linear_mse}")
            total_count += 1
        if total_count == 0:
            out_file.write(f"For feature: {feature} 0 missing\n")
            print(f"\tFor feature: {feature} 0 missing\n")
            continue
        # print(f"\tFor feature: {feature}\n\t\tBRITS: {model_mse['BRITS']/total_count}\n\t\tSAITS: {model_mse['SAITS']/total_count}\n\t\tMICE: {model_mse['MICE']/total_count}\n\t\tMVTS: {model_mse['MVTS']/total_count}\n\t\tLINEAR: {model_mse['LINEAR']/total_count}")
        print(f"\tFor feature: {feature}\n\t\tSAITS: {model_mse['SAITS']/total_count}\n")
        out_file.write(f"For feature: {feature}\n\tSAITS: {model_mse['SAITS']/total_count}\n")
        results[season][feature] = model_mse['SAITS']/total_count
    out_file.close()

def forward_prediction_LT_day(results, models, given_season, season_df, max_length, season_array, mean, std, suffix, slide=False, same=True, data_folder=None, diff_folder=None, k=-1):
    out_folder = 'cross_val_LT_preds'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    out_file = open(f'{out_folder}/cv_{suffix}_{k if k != -1 else 0}.txt', 'w')
    filename = 'json/json_eval_forward_LT'
    feature_idx = features.index('LTE50')
    X, Y, pads = split_XY(season_df, max_length, season_array, features, is_pad=True)

    non_missing_indices = np.where(~np.isnan(X[0, :, feature_idx]))[0]

    draws_1 = {
        # 'BRITS': [],
        'SAITS': [],
        # 'MICE': [],
        # 'MVTS': []
    }

    draws_2 = {
        # 'BRITS': [],
        'SAITS': [],
        # 'MICE': [],
        # 'MVTS': []
    }
    # draws_1_brits = []
    # draws_1_saits = []
    # draws_2_brits = []
    # draws_2_saits = []
    # draw_data = {}
    # draws_2 = []
    x_axis = []

    season_mse_1 = {
        # 'BRITS': [],
        'SAITS': [],
        # 'MICE': [],
        # 'MVTS': []
    }
    season_mse_2 = {
        # 'BRITS': [],
        'SAITS': [],
        # 'MICE': [],
        # 'MVTS': []
    }

    season_mse_3 = {
        # 'BRITS': [],
        'SAITS': [],
        # 'MICE': [],
        # 'MVTS': []
    }
    # season_mse_brits = []
    # season_mse_saits = []
    # season_mse_2_brits = []
    # season_mse_2_saits = []
    print(f"Season: {given_season}")
    out_file.write(f"Season: {given_season}\n")
    for i in range(len(non_missing_indices)-3):
        # print(f"i = {i}")
        # mse_1_brits = 0
        # mse_1_saits = 0
        # mse_2_brits = 0
        # mse_2_saits = 0

        mse_1 = {
            # 'BRITS': 0,
            'SAITS': 0,
            # 'MICE': 0,
            # 'MVTS': 0
        }
        mse_2 = {
            # 'BRITS': 0,
            'SAITS': 0,
            # 'MICE': 0,
            # 'MVTS': 0
        }

        mse_3 = {
            # 'BRITS': 0,
            'SAITS': 0,
            # 'MICE': 0,
            # 'MVTS': 0
        }
        # mse_2 = 0
        trial_count = 0
        if slide:
            num_trials = len(non_missing_indices) - i - 3 - pads[0]
        else:
            num_trials = 1
        for trial in tqdm(range(num_trials)):
            fs = open(filename, 'w')
            
            missing_indices, Xeval, evals = forward_parse_id_day(fs, X[0], Y[0], mean, std, feature_idx, i, trial_num=trial, all=True, same=same)
            
            fs.close()
            if len(missing_indices) == 0:
                continue

            val_iter = data_loader.get_loader(batch_size=1, filename=filename)
            
            for idx, data in enumerate(val_iter):
                data = utils.to_var(data)
                row_indices = missing_indices // len(features)
                with torch.no_grad():
                    # BRITS
                    # ret = models['BRITS'].run_on_batch(data, None)
                    # eval_ = ret['evals'].data.cpu().numpy()
                    # eval_ = np.squeeze(eval_)
                    # imputation_brits = ret['imputations'].data.cpu().numpy()
                    # imputation_brits = np.squeeze(imputation_brits)
                    # imputed_brits = imputation_brits[row_indices, feature_idx]

                    # SAITS
                    Xeval = np.reshape(Xeval, (1, Xeval.shape[0], Xeval.shape[1]))
                    imputation_saits = models['SAITS'].impute(Xeval, k=k, is_test=True)
                    imputation_saits = np.squeeze(imputation_saits)
                    imputed_saits = imputation_saits[row_indices, feature_idx]

                    # MICE
                    # ret_eval = copy.deepcopy(eval_)
                    # ret_eval[row_indices, feature_idx] = np.nan
                    # imputation_mice = models['MICE'].transform(ret_eval)
                    # imputed_mice = imputation_mice[row_indices, feature_idx]
                    
                    # ret_eval = copy.deepcopy(eval_)
                    # ret_eval = unnormalize(ret_eval, mean, std, feature_idx)
                    # ret_eval[row_indices, feature_idx] = np.nan
                    # test_df = pd.DataFrame(ret_eval, columns=features)
                    # # print(f"test df: {test_df.columns}")
                    # add_season_id_and_save('./transformer/data_dir', test_df, filename=f'ColdHardiness_Grape_Merlot_test_2.csv')

                    # params = {
                    #     'config_filepath': None, 
                    #     'output_dir': './transformer/output/', 
                    #     'data_dir': './transformer/data_dir/', 
                    #     'load_model': f'./transformer/output/model_mvts_{suffix}/checkpoints/model_best.pth', 
                    #     'resume': False, 
                    #     'change_output': False, 
                    #     'save_all': False, 
                    #     'experiment_name': 'MVTS_test_2',
                    #     'comment': 'imputation test', 
                    #     'no_timestamp': False, 
                    #     'records_file': 'Imputation_records.csv', 
                    #     'console': False, 
                    #     'print_interval': 1, 
                    #     'gpu': '-1', 
                    #     'n_proc': 1, 
                    #     'num_workers': 0, 
                    #     'seed': None, 
                    #     'limit_size': None, 
                    #     'test_only': 'testset', 
                    #     'data_class': 'agaid', 
                    #     'labels': None, 
                    #     'test_from': './transformer/test_indices.txt', 
                    #     'test_ratio': 0, 
                    #     'val_ratio': 0, 
                    #     'pattern': None, 
                    #     'val_pattern': None, 
                    #     'test_pattern': 'Merlot_test', 
                    #     'normalization': 'standardization', 
                    #     'norm_from': None, 
                    #     'subsample_factor': None, 
                    #     'task': 'imputation', 
                    #     'masking_ratio': 0.15, 
                    #     'mean_mask_length': 10.0, 
                    #     'mask_mode': 'separate', 
                    #     'mask_distribution': 'geometric', 
                    #     'exclude_feats': None, 
                    #     'mask_feats': [0, 1], 
                    #     'start_hint': 0.0, 
                    #     'end_hint': 0.0, 
                    #     'harden': True, 
                    #     'epochs': 1000, 
                    #     'val_interval': 2, 
                    #     'optimizer': 'Adam', 
                    #     'lr': 0.0009, 
                    #     'lr_step': [1000000], 
                    #     'lr_factor': [0.1], 
                    #     'batch_size': 16, 
                    #     'l2_reg': 0, 
                    #     'global_reg': False, 
                    #     'key_metric': 'loss', 
                    #     'freeze': False, 
                    #     'model': 'transformer', 
                    #     'max_seq_len': 252, 
                    #     'data_window_len': None, 
                    #     'd_model': 128, 
                    #     'dim_feedforward': 256, 
                    #     'num_heads': 8, 
                    #     'num_layers': 3, 
                    #     'dropout': 0.1, 
                    #     'pos_encoding': 'learnable', 
                    #     'activation': 'relu', 
                    #     'normalization_layer': 'BatchNorm'
                    # }

                    # transformer_preds = run_transformer(params)
                    # # # print(f'trasformer preds: {transformer_preds.shape}')
                    
                    # imputation_transformer = np.squeeze(transformer_preds)
                    # imputed_transformer = imputation_transformer[row_indices, feature_idx].cpu().detach().numpy()


                    # REAL
                    real_values = evals[row_indices, feature_idx]

                    # Same day MSE
                    # mse_1['BRITS'] += ((real_values[0] - imputed_brits[0]) ** 2)
                    # season_mse_1['BRITS'].append(((real_values[0] - imputed_brits[0]) ** 2))

                    mse_1['SAITS'] += ((real_values[0] - imputed_saits[0]) ** 2)
                    season_mse_1['SAITS'].append(((real_values[0] - imputed_saits[0]) ** 2))

                    # mse_1['MICE'] += ((real_values[0] - imputed_mice[0]) ** 2)
                    # season_mse_1['MICE'].append(((real_values[0] - imputed_mice[0]) ** 2))

                    # mse_1['MVTS'] += ((real_values[0] - imputed_transformer[0]) ** 2)
                    # season_mse_1['MVTS'].append(((real_values[0] - imputed_transformer[0]) ** 2))

                    # Next day MSE
                    # mse_2['BRITS'] += ((real_values[1] - imputed_brits[1]) ** 2)
                    # season_mse_2['BRITS'].append(((real_values[1] - imputed_brits[1]) ** 2))

                    mse_2['SAITS'] += ((real_values[1] - imputed_saits[1]) ** 2)
                    season_mse_2['SAITS'].append(((real_values[1] - imputed_saits[1]) ** 2))

                    # mse_2['MICE'] += ((real_values[1] - imputed_mice[1]) ** 2)
                    # season_mse_2['MICE'].append(((real_values[1] - imputed_mice[1]) ** 2))

                    # mse_2['MVTS'] += ((real_values[1] - imputed_transformer[1]) ** 2)
                    # season_mse_2['MVTS'].append(((real_values[1] - imputed_transformer[1]) ** 2))

                    # mse_3['BRITS'] += ((real_values[2] - imputed_brits[2]) ** 2)
                    # season_mse_3['BRITS'].append(((real_values[2] - imputed_brits[2]) ** 2))

                    mse_3['SAITS'] += ((real_values[2] - imputed_saits[2]) ** 2)
                    season_mse_3['SAITS'].append(((real_values[2] - imputed_saits[2]) ** 2))

                    # mse_3['MICE'] += ((real_values[2] - imputed_mice[2]) ** 2)
                    # season_mse_3['MICE'].append(((real_values[2] - imputed_mice[2]) ** 2))

                    # mse_3['MVTS'] += ((real_values[2] - imputed_transformer[2]) ** 2)
                    # season_mse_2['MVTS'].append(((real_values[2] - imputed_transformer[2]) ** 2))
                    # mse_2 += ((real_values[1] - imputed_brits[1]) ** 2)
                    trial_count += 1

                    # real_value = evals[:, feature_idx]
                
                    # draw_data['real'] = unnormalize(real_value, mean, std, feature_idx)
                    # draw_data['real'][original_missing_indices] = 0

                    # missing_values = copy.deepcopy(real_value)
                    # missing_values[row_indices] = 0
                    # draw_data['missing'] = unnormalize(missing_values, mean, std, feature_idx)
                    # draw_data['missing'][original_missing_indices] = 0
                    # draw_data['missing'][row_indices] = 0

                    # draw_data['BRITS'] = unnormalize(imputation_brits[:, feature_idx], mean, std, feature_idx)
                    # draw_data['SAITS'] = unnormalize(imputation_saits[:, feature_idx], mean, std, feature_idx)
                    # draw_data['MICE'] = unnormalize(imputation_mice[:, feature_idx], mean, std, feature_idx)
                    # draw_data['MVTS'] = unnormalize(imputation_transformer[:, feature_idx], mean, std, feature_idx).numpy()

                    # draws['real'][row_indices] = 0
                    # if data_folder is not None:
                    #     draw_data_plot(draw_data, features[feature_idx], given_season, folder=data_folder, existing=i)
                    # if diff_folder is not None:
                    #     graph_bar_diff_multi(diff_folder, draw_data['real'][row_indices], draw_data, f'Difference From Gorund Truth for LTE50 in {given_season} existing = {i+1}', np.arange(len(row_indices)), 'Days where LTE50 value is available', 'Degrees', given_season, 'LTE50', missing=row_indices, existing=i)

        if trial_count <= 0:
            continue
        # mse_1['BRITS'] /= trial_count
        mse_1['SAITS'] /= trial_count
        # mse_1['MICE'] /= trial_count
        # mse_1['MVTS'] /= trial_count

        # mse_2['BRITS'] /= trial_count
        mse_2['SAITS'] /= trial_count
        # mse_2['MICE'] /= trial_count
        # mse_2['MVTS'] /= trial_count

        # mse_3['BRITS'] /= trial_count
        mse_3['SAITS'] /= trial_count
        # mse_3['MICE'] /= trial_count
        # mse_3['MVTS'] /= trial_count

        # mse_2 /= trial_count
        # draws_1['BRITS'].append(mse_1['BRITS'])
        # draws_1['SAITS'].append(mse_1['SAITS'])
        # draws_1['MICE'].append(mse_1['MICE'])
        # draws_1['MVTS'].append(mse_1['MVTS'])

        # draws_2['BRITS'].append(mse_2['BRITS'])
        # draws_2['SAITS'].append(mse_2['SAITS'])
        # draws_2['MICE'].append(mse_2['MICE'])
        # draws_2['MVTS'].append(mse_2['MVTS'])
        # draws_2.append(mse_2)
        # x_axis.append(i+1)
    # print(f"For season = {given_season} same day prediction results:\n\tBRITS mse = {np.array(season_mse_1['BRITS']).mean()}\n\tSAITS mse = {np.array(season_mse_1['SAITS']).mean()}\n\tMICE mse = {np.array(season_mse_1['MICE']).mean()}\n\tMVTS mse = {np.array(season_mse_1['MVTS']).mean()}")
    # print(f"For season = {given_season} next day prediction results:\n\tBRITS mse = {np.array(season_mse_2['BRITS']).mean()}\n\tSAITS mse = {np.array(season_mse_2['SAITS']).mean()}\n\tMICE mse = {np.array(season_mse_2['MICE']).mean()}\n\tMVTS mse = {np.array(season_mse_2['MVTS']).mean()}")
    # print(f"For season = {given_season} next 2 day prediction results:\n\tBRITS mse = {np.array(season_mse_3['BRITS']).mean()}\n\tSAITS mse = {np.array(season_mse_3['SAITS']).mean()}\n\tMICE mse = {np.array(season_mse_3['MICE']).mean()}\n\tMVTS mse = {np.array(season_mse_3['MVTS']).mean()}")
    same_mse = np.array(season_mse_1['SAITS']).mean()
    next_1_mse = np.array(season_mse_2['SAITS']).mean()
    next_2_mse = np.array(season_mse_3['SAITS']).mean()
    print(f"For season = {given_season} same day prediction results:\n\tSAITS mse = {same_mse}\n")
    print(f"For season = {given_season} next day prediction results:\n\tSAITS mse = {next_1_mse}\n")
    print(f"For season = {given_season} next 2 day prediction results:\n\tSAITS mse = {next_2_mse}\n")
    out_file.write(f"For season = {given_season} same day prediction results:\n\tSAITS mse = {same_mse}\n")
    out_file.write(f"For season = {given_season} next day prediction results:\n\tSAITS mse = {next_1_mse}\n")
    out_file.write(f"For season = {given_season} next 2 day prediction results:\n\tSAITS mse = {next_2_mse}\n")
    ferguson_mse, ferguson_preds = get_FG(season_df, 'LTE50', 'PREDICTED_LTE50', season_array[0])
    print(f"For season = {given_season} Ferguson mse = {ferguson_mse}\n")
    out_file.write(f"For season = {given_season} Ferguson mse = {ferguson_mse}\n")
    out_file.close()
    results[given_season] = {'same': same_mse, 'next_1': next_1_mse, 'next_2': next_2_mse}

# def evaluate_imputation_model(model_dir, model_name, suffix, test_season_df, test_seasons, mean, std, max_length):
#     model_path = f"{model_dir}/model_{model_name}_{suffix}.model"
#     if model_name == "BRITS":
#         RNN_HID_SIZE = 64
#         IMPUTE_WEIGHT = 0.5
#         LABEL_WEIGHT = 1
#         model = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT, feature_len=len(features))
#         if os.path.exists(model_path):
#             model.load_state_dict(torch.load(model_path))
#         if torch.cuda.is_available():
#             model = model.cuda()
        
        
#     elif model_name == "SAITS":
#         pass
#     elif model_name == "MICE":
#         pass
#     elif model_name == "MVTS":
#         pass
#     else:
#         pass

def cross_eval_imputation(df_file, model_dir):

    # models = ['BRITS', 'SAITS', 'MICE', 'MVTS']
    # model_names = ['SAITS']
    models = {
        'SAITS': None
        }

    df = pd.read_csv(df_file)
    modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#False, is_year=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)
    ks = [-1, 3, 4, 5, 6]
    # ks = [5]
    for k in ks:
        result_LT_day = {}
        result_imputation = {}
        for i in range(len(season_array)):
            test_seasons = season_array[i]
            test_season_name = idx_to_seasons[i]
            suffix = test_season_name
            if i == 0:
                train_seasons = season_array[(i + 1):]
            elif i == len(season_array) - 1:
                train_seasons = season_array[:i]
            else:
                train_seasons = copy.deepcopy(season_array[:i])
                rest = copy.deepcopy(season_array[(i + 1):])
                train_seasons.extend(rest)

            train_season_df = season_df.drop(test_seasons, axis=0)
            mean, std = get_mean_std(train_season_df, features)

            test_season_df = season_df.loc[test_seasons]
            
            for model in models.keys():
                model_path = train_imputation_model(train_season_df, train_seasons, max_length, mean, std, model, suffix, model_dir, k)
                model_loaded = pickle.load(open(model_path, 'rb'))
                models[model] = model_loaded
                evaluate_imputation(result_imputation, test_season_name, test_season_df, [test_seasons], max_length, models, mean, std, suffix, k)
                forward_prediction_LT_day(result_LT_day, models, test_season_name, test_season_df, max_length, [test_seasons], mean, std, suffix, k)
        df_imputation = pd.DataFrame(result_imputation)
        df_LT_day = pd.DataFrame(result_LT_day)
        dir = 'cross_val_csvs'
        if not os.path.isdir(dir):
            os.makedirs(dir)
        df_imputation.T.to_csv(f'{dir}/result_imputation_{k if k != -1 else 0}.csv')
        df_LT_day.T.to_csv(f"{dir}/result_LT_day_{k if k != -1 else 0}.csv")

if __name__ == "__main__":
    df_file = f'ColdHardiness_Grape_Merlot_2.csv'
    model_dir = "cv_models"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    cross_eval_imputation(df_file, model_dir)