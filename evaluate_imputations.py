import copy
from threading import get_ident
from turtle import color
from cvxpy import real
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
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae, cal_mse
import pickle

warnings.filterwarnings("ignore")
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import sys
np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_random = 0.2

features = [
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
    'ETR', # ???
    'LTE50'
    # 'SEASON_JDAY'
]


seasons = {
# '1988-1989': 0,
# '1989-1990': 1,
# '1990-1991': 2,
# '1991-1992': 3,
# '1992-1993': 4,
# '1993-1994': 5,
# '1994-1995': 6,
# '1995-1996': 7,
# '1996-1997': 8,
# '1997-1998': 9,
# '1998-1999': 10,
# '1999-2000': 11,
# '2000-2001': 12,
# '2001-2002': 13,
# '2002-2003': 14,
# '2003-2004': 15,
# '2004-2005': 16,
# '2005-2006': 17,
# '2006-2007': 18,
# '2007-2008': 19,
# '2008-2009': 20,
# '2009-2010': 21,
# '2010-2011': 22,
# '2011-2012': 23,
# '2012-2013': 24,
# '2013-2014': 25,
# '2014-2015': 26,
# '2015-2016': 27,
# '2016-2017': 28,
# '2017-2018': 29,
# '2018-2019': 30,
# '2019-2020': 31,
'2020-2021': 32,
'2021-2022': 33,
}

RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.5
LABEL_WEIGHT = 1


# def add_season_id(data_folder, season_df):
#     season_df['season_id'] = 0
#     for season_id in range(len(season_array)):
#         for idx in season_array[season_id]:
#             train_season_df.loc[idx, 'season_id'] = season_id
#     season_df.to_csv(f'{data_folder}/ColdHardiness_Grape_Merlot_test.csv', index=False)

folder = './json/'
file = 'json_eval_2_LT'
filename = folder + file
if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.isdir('plots'):
    os.makedirs('plots')
if not os.path.isdir('subplots'):
    os.makedirs('subplots')


mean = []
std = []

# features_impute = [features.index('MEAN_AT'), features.index('AVG_REL_HUMIDITY')]
############## Data Load and Preprocess ##############
complete_seasons = [4, 5, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

# if n_random == 0:
#     df = pd.read_csv(f'ColdHardiness_Grape_Merlot_new_synthetic.csv')
# else:
df = pd.read_csv(f'ColdHardiness_Grape_Merlot_2.csv')
modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#False, is_year=True)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)

# train_season_complete = []#[season_array[i] for i in complete_seasons[:-2]]
# for s in complete_seasons[:-2]:
#     s_copy = copy.deepcopy(season_array[s])
#     train_season_complete.extend(s_copy)

# train_season_df = season_df.loc[train_season_complete]

train_season_df = season_df.drop(season_array[-1], axis=0)
train_season_df = train_season_df.drop(season_array[-2], axis=0)

mean, std = get_mean_std(train_season_df, features)


############## Load Models ##############


# normalized_season_df = train_season_df[features].copy()
# normalized_season_df = (normalized_season_df - mean) /std
# mice_impute = IterativeImputer(random_state=0, max_iter=20)
# mice_impute.fit(normalized_season_df[features].to_numpy())

model_dir = "./model_abstract"

############## Load BRITS ##############
model_brits = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT, feature_len=len(features))
model_brits_path = f"{model_dir}/model_BRITS_LT_orig.model"#{n_random}.model"
if os.path.exists(model_brits_path):
    model_brits.load_state_dict(torch.load(model_brits_path))

if torch.cuda.is_available():
    model_brits = model_brits.cuda()
model_brits.eval()

############## Load SAITS ##############
saits_file = f"{model_dir}/model_saits_orig.model"#{n_random}.model"
model_saits = pickle.load(open(saits_file, 'rb'))


############## Load MICE ##############
mice_file = f"{model_dir}/model_mice_orig.model"#{n_random}.model"
model_mice = pickle.load(open(mice_file, 'rb'))

############## Load MVTS ##############
params = {
    'config_filepath': None, 
    'output_dir': './transformer/output/', 
    'data_dir': './transformer/data_dir/', 
    'load_model': './transformer/output/mvts-synth-0.2/checkpoints/model_best.pth', 
    'resume': False, 
    'change_output': False, 
    'save_all': False, 
    'experiment_name': 'MVTS test',
    'comment': 'imputation test', 
    'no_timestamp': False, 
    'records_file': 'Imputation_records.csv', 
    'console': False, 
    'print_interval': 1, 
    'gpu': '0', 
    'n_proc': 1, 
    'num_workers': 0, 
    'seed': None, 
    'limit_size': None, 
    'test_only': 'testset', 
    'data_class': 'agaid', 
    'labels': None, 
    'test_from': './transformer/test_indices.txt', 
    'test_ratio': 0, 
    'val_ratio': 0, 
    'pattern': 'Merlot_test', 
    'val_pattern': None, 
    'test_pattern': None, 
    'normalization': 'standardization', 
    'norm_from': None, 
    'subsample_factor': None, 
    'task': 'imputation', 
    'masking_ratio': 0.15, 
    'mean_mask_length': 10.0, 
    'mask_mode': 'separate', 
    'mask_distribution': 'geometric', 
    'exclude_feats': None, 
    'mask_feats': [0, 1], 
    'start_hint': 0.0, 
    'end_hint': 0.0, 
    'harden': True, 
    'epochs': 1000, 
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
    'max_seq_len': 252, 
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

############## Load MEAN ##############
# model_mean = np.load(f"{model_dir}/mean.npy")


############## Draw Functions ##############
def graph_bar_diff_multi(diff_folder, GT_values, result_dict, title, x, xlabel, ylabel, season, feature, drop_linear=False, missing=None, existing=-1):
    plot_dict = {}
    plt.figure(figsize=(32,20))
    for key, value in result_dict.items():
        # print(f"key: {key}")
        plot_dict[key] = np.abs(GT_values) - np.abs(value[missing])
    # ind = np.arange(prediction.shape[0])
    # x = np.array(x)
    width = 0.3
    pos = 0
    remove_keys = ['real', 'missing']

    colors = ['tab:orange', 'tab:blue', 'tab:cyan', 'tab:purple']
    # if drop_linear:
    #   remove_keys.append('LinearInterp')
    i = 0
    for key, value in plot_dict.items():
        if key not in remove_keys:
            # print(f"x = {len(x)}, value = {len(value)}")
            plt.bar(x + pos, value, width, label = key, color=colors[i])
            i += 1
            pos += width

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=25)
    plt.xticks([r + width for r in range(len(x))], [str(i) for i in x],fontsize=20)
    plt.yticks(fontsize=20)
    # plt.axis([0, 80, -2, 3])

    plt.legend(loc='best', fontsize=25)
    plt.tight_layout(pad=5)
    folder = f"{diff_folder}/{season}"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/result-diff-{feature}-{season}-{existing}.png', dpi=300)
    plt.close()
    # plt.show()



def draw_data_trend(df_t, df_c, f, interp_name, i):
    plt.figure(figsize=(32,9))
    plt.title(f)
    ax = plt.subplot(121)
    ax.set_title(f+' original')
    plt.plot(np.arange(df_t.shape[0]), df_t)
    ax = plt.subplot(122)
    ax.set_title(f+' imputed by '+interp_name)
    plt.plot(np.arange(df_c.shape[0]), df_c)
    plt.savefig(f"subplots/{f}-{interp_name}-{i}.png", dpi=300)
    plt.close()

def draw_data_plot(results, f, season, folder='subplots', is_original=False, existing=-1):
    
    plt.figure(figsize=(40,28))
    plt.title(f"For feature = {f} in Season {season} existing LT = {existing}", fontsize=30)
    if is_original:
        ax = plt.subplot(211)
        ax.set_title('Feature = '+f+' Season = '+season+' original data', fontsize=27)
        plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)

        ax = plt.subplot(212)
        # ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by MICE', fontsize=20)
        # plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'], 'tab:blue')
        # ax.set_xlabel('Days', fontsize=16)
        # ax.set_ylabel('Values', fontsize=16)

        # ax = plt.subplot(313)
        ax.set_title('Feature = '+f+' Season = '+season+' imputed by BRITS', fontsize=27)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)
    else:
        ax = plt.subplot(611)
        ax.set_title('Feature = '+f+' Season = '+season+' original data', fontsize=27)
        plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)

        ax = plt.subplot(612)
        ax.set_title('Feature = '+f+' Season = '+season+' missing data', fontsize=27)
        plt.plot(np.arange(results['missing'].shape[0]), results['missing'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)

        ax = plt.subplot(613)
        ax.set_title('Feature = '+f+' Season = '+season+' imputed by SAITS', fontsize=20)
        plt.plot(np.arange(results['SAITS'].shape[0]), results['SAITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)

        ax = plt.subplot(614)
        ax.set_title('Feature = '+f+' Season = '+season+' imputed by BRITS', fontsize=27)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)

        ax = plt.subplot(615)
        ax.set_title('Feature = '+f+' Season = '+season+' imputed by MICE', fontsize=27)
        plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)

        ax = plt.subplot(616)
        ax.set_title('Feature = '+f+' Season = '+season+' imputed by MVTS', fontsize=27)
        plt.plot(np.arange(results['MVTS'].shape[0]), results['MVTS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)

    plt.tight_layout(pad=5)
    folder = f"{folder}/{season}/{f}"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if existing == -1:
        plt.savefig(f"{folder}/{f}-imputations-season-{season}.png", dpi=300)
    else:
        plt.savefig(f"{folder}/{f}-imputations-season-{season}-{existing}.png", dpi=300)
    plt.close()

############## Model Input Process Functions ##############
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

def parse_id(fs, x, y, feature_impute_idx, length, trial_num=-1, dependent_features=None, real_test=True, random_start=False, random=False, pad=-1):
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
        # print(f"idx1: {idx1}")
        idx1 = idx1 * len(features) + feature_impute_idx
        # print(f"idx1-1: {idx1}")
        # exit()

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
    return indices, values

# given_feature = 'AVG_REL_HUMIDITY'
L = [i for i in range(1, 31, 1)]
# L = [1, 5, 10, 20]#, 70, 100, 150, 200]
iter = 30


start_time = time.time()


# given_features = features

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
# if n_random == 0:
#     test_df = pd.read_csv(f'ColdHardiness_Grape_Merlot_new_synthetic.csv')
# else:
#     test_df = pd.read_csv(f'ColdHardiness_Grape_Merlot_new_synthetic_{n_random}.csv')
test_df = pd.read_csv(f'ColdHardiness_Grape_Merlot_2.csv')
test_modified_df, test_dormant_seasons = preprocess_missing_values(test_df, features, is_dormant=True)#, is_year=True)
# print(f"dormant seasons: {len(test_dormant_seasons)}\n {test_dormant_seasons}")
season_df, season_array, max_length = get_seasons_data(test_modified_df, test_dormant_seasons, features, is_dormant=True)#, is_year=True)

# print(f"season array: {season_array[1]}")
plot_mse_folder = 'overlapping_mse/'

def evaluate_imputation(mse_folder):
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
    
    for season in seasons.keys():
        print(f"For season: {season}")
        for feature in given_features:
            season_idx = seasons[season]
            feature_idx = features.index(feature)
            X, Y, pads = split_XY(season_df, max_length, season_array, features, is_pad=True)
            original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
            
            iter = 1#len(season_array[season_idx]) - (l-1) - len(original_missing_indices) - pads[season_idx]
            total_count = 0
            model_mse = {
                'BRITS': 0,
                'SAITS': 0,
                'MICE': 0,
                'MVTS': 0,
                'LINEAR': 0
            }
            for i in range(iter):
                fs = open(filename, "w")
                if feature.split('_')[-1] not in feature_dependency.keys():
                    dependent_feature_ids = []
                else:
                    dependent_feature_ids = [features.index(f) for f in feature_dependency[feature.split('_')[-1]] if (f != feature) and (f in features)]
                
                missing_indices, Xeval = parse_id(fs, X[season_idx], Y[season_idx], feature_idx, -1, i, dependent_feature_ids, random=True, pad=pads[0])
                print(f"missing idx: {missing_indices}")
                print(f"missing idx: len: {len(missing_indices)}")
                fs.close()

                # print(f"i: {i}\nmissing indices: {missing_indices}")
                val_iter = data_loader.get_loader(batch_size=1, filename=filename)

                for idx, data in enumerate(val_iter):
                    data = utils.to_var(data)
                    row_indices = missing_indices // len(features)
                    print(f"rows: {row_indices}")
                    ret = model_brits.run_on_batch(data, None)
                    eval_ = ret['evals'].data.cpu().numpy()
                    eval_ = np.squeeze(eval_)
                    imputation_brits = ret['imputations'].data.cpu().numpy()
                    imputation_brits = np.squeeze(imputation_brits)
                    imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)
                    prinbt(f"imputed brits: {imputed_brits}")
                    Xeval = np.reshape(Xeval, (1, Xeval.shape[0], Xeval.shape[1]))
                    # X_intact, Xe, missing_mask, indicating_mask = mcar(Xeval, 0.1) # hold out 10% observed values as ground truth
                    
                    # Xe = masked_fill(Xe, 1 - missing_mask, np.nan)
                    imputation_saits = model_saits.impute(Xeval)
                    # print(f"SAITS imputation: {imputation_saits}")
                    imputation_saits = np.squeeze(imputation_saits)
                    imputed_saits = imputation_saits[row_indices, feature_idx]
                    print(f"imputed saits: {imputed_saits}")

                    ret_eval = copy.deepcopy(eval_)
                    ret_eval[row_indices, feature_idx] = np.nan
                    imputation_mice = model_mice.transform(ret_eval)
                    imputed_mice = imputation_mice[row_indices, feature_idx]
                    print(f"imputed mice: {imputed_mice}")
                    
                    ret_eval = copy.deepcopy(eval_)
                    ret_eval = unnormalize(ret_eval, mean, std, feature_idx)
                    ret_eval[row_indices, feature_idx] = np.nan
                    test_df = pd.DataFrame(ret_eval, columns=features)
                    add_season_id_and_save('./transformer/data_dir', test_df, filename='ColdHardiness_Grape_Merlot_test_2.csv')

                    params = {
                        'config_filepath': None, 
                        'output_dir': './transformer/output/', 
                        'data_dir': './transformer/data_dir/', 
                        'load_model': './transformer/output/mvts-orig/checkpoints/model_best.pth', 
                        'resume': False, 
                        'change_output': False, 
                        'save_all': False, 
                        'experiment_name': 'MVTS_test',
                        'comment': 'imputation test', 
                        'no_timestamp': False, 
                        'records_file': 'Imputation_records.csv', 
                        'console': False, 
                        'print_interval': 1, 
                        'gpu': '0', 
                        'n_proc': 1, 
                        'num_workers': 0, 
                        'seed': None, 
                        'limit_size': None, 
                        'test_only': 'testset', 
                        'data_class': 'agaid', 
                        'labels': None, 
                        'test_from': './transformer/test_indices.txt', 
                        'test_ratio': 0, 
                        'val_ratio': 0, 
                        'pattern': None, 
                        'val_pattern': None, 
                        'test_pattern': 'Merlot_test', 
                        'normalization': 'standardization', 
                        'norm_from': None, 
                        'subsample_factor': None, 
                        'task': 'imputation', 
                        'masking_ratio': 0.15, 
                        'mean_mask_length': 10.0, 
                        'mask_mode': 'separate', 
                        'mask_distribution': 'geometric', 
                        'exclude_feats': None, 
                        'mask_feats': [0, 1], 
                        'start_hint': 0.0, 
                        'end_hint': 0.0, 
                        'harden': True, 
                        'epochs': 1000, 
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
                        'max_seq_len': 252, 
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

                    transformer_preds = run_transformer(params)
                    # print(f'trasformer preds: {transformer_preds}')
                    
                    imputation_transformer = np.squeeze(transformer_preds)
                    imputed_transformer = imputation_transformer[row_indices, feature_idx].cpu().detach().numpy()
                    print(f"imputed mvts: {imputed_transformer}")

                    ret_eval[row_indices, feature_idx] = np.nan
                    ret_eval_df = pd.DataFrame(ret_eval, columns=features)
                    imputed_linear = ret_eval_df.interpolate(method='linear', limit_direction='both')
                    imputed_linear = imputed_linear.to_numpy()
                    imputed_linear = imputed_linear[row_indices, feature_idx]
                    print(f"imputd linear: {imputed_linear}")

                    real_values = eval_[row_indices, feature_idx]
                    print(f"real: {real_values}")

                model_mse['BRITS'] += ((real_values - imputed_brits) ** 2).mean()
                model_mse['SAITS'] += ((real_values - imputed_saits) ** 2).mean()
                model_mse['MICE'] += ((real_values - imputed_mice) ** 2).mean()
                model_mse['MVTS'] += ((real_values - imputed_transformer) ** 2).mean()
                model_mse['LINEAR'] += ((real_values - imputed_linear) ** 2).mean()
                # print(f"real: {real_values}\nlinear mse: {linear_mse}")
                total_count += 1
            print(f"\tFor feature: {feature}\n\t\tBRITS: {model_mse['BRITS']/total_count}\n\t\tSAITS: {model_mse['SAITS']/total_count}\n\t\tMICE: {model_mse['MICE']/total_count}\n\t\tMVTS: {model_mse['MVTS']/total_count}\n\t\tLINEAR: {model_mse['LINEAR']/total_count}")

def do_evaluation(mse_folder, eval_type, eval_season='2020-2021'):
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

    # given_features = [
        # 'MEAN_AT', # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
        # 'MIN_AT', # a
        # 'AVG_AT', # average temp is AgWeather Network
        # 'MAX_AT',  # a
        # 'MIN_REL_HUMIDITY', # a
        # 'AVG_REL_HUMIDITY', # a
        # 'MAX_REL_HUMIDITY', # a
        # 'MIN_DEWPT', # a
        # 'AVG_DEWPT', # a
        # 'MAX_DEWPT', # a
        # 'P_INCHES', # precipitation # a
        # 'WS_MPH',
        # 'LTE50']
    # L = [i for i in range(1, 31)]
    L = [1, 5, 10, 15, 20, 25, 30]
    for given_feature in given_features:
        result_mse_plots = {
        'BRITS': [],
        'SAITS': [],
        'MICE': [],
        'MVTS': [],
        'MEAN': [],
        'MEDIAN': [],
        'Linear': []
        }
        results = {
            'BRITS': {},
            'SAITS': {},
            'MICE': {},
            'MVTS': {},
            'MEAN': {},
            'MEDIAN': {},
            'Linear': {}
        }
        l_needed = []
        for l in L:
            # season_idx = seasons[eval_season]
            season_idx = seasons[eval_season]
            feature_idx = features.index(given_feature)
            X, Y, pads = split_XY(season_df, max_length, season_array, features, is_pad=True)
            original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
            
            iter = len(season_array[season_idx]) - (l-1) - len(original_missing_indices) - pads[season_idx]
            print(f"For feature = {given_feature} and length = {l}")
            # print(f"original miss: {original_missing_indices.shape[0]}\n{original_missing_indices}\nseason: {len(season_array[season_idx])}")
            total_count = 0
            brits_mse = 0
            saits_mse = 0
            mice_mse = 0
            transformer_mse = 0
            mean_mse = 0
            linear_mse = 0
            neg = 0
            for i in tqdm(range(iter)):
                # i.set_description(f"For {given_feature} & L = {l}")
                real_values = []
                imputed_brits = []
                imputed_saits = []
                
                fs = open(filename, 'w')

                if given_feature.split('_')[-1] not in feature_dependency.keys():
                    dependent_feature_ids = []
                else:
                    dependent_feature_ids = [features.index(f) for f in feature_dependency[given_feature.split('_')[-1]] if (f != given_feature) and (f in features)]
                if eval_type == 'random':
                    missing_indices = parse_id(fs, X[season_idx], Y[season_idx], feature_idx, l, i, dependent_feature_ids, random_start=True)
                else:
                    missing_indices, Xeval = parse_id(fs, X[season_idx], Y[season_idx], feature_idx, l, i, dependent_feature_ids)
                fs.close()
                if len(missing_indices) == 0:
                    # print(f"0 missing added")
                    continue
                # print(f"i: {i}\nmissing indices: {missing_indices}")
                val_iter = data_loader.get_loader(batch_size=1, filename=filename)

                for idx, data in enumerate(val_iter):
                    data = utils.to_var(data)
                    row_indices = missing_indices // len(features)
                    # print(f"rows: {row_indices}")
                    ret = model_brits.run_on_batch(data, None)
                    eval_ = ret['evals'].data.cpu().numpy()
                    eval_ = np.squeeze(eval_)
                    imputation_brits = ret['imputations'].data.cpu().numpy()
                    imputation_brits = np.squeeze(imputation_brits)
                    imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)
                    
                    Xeval = np.reshape(Xeval, (1, Xeval.shape[0], Xeval.shape[1]))
                    X_intact, Xe, missing_mask, indicating_mask = mcar(Xeval, 0.1) # hold out 10% observed values as ground truth
                    
                    Xe = masked_fill(Xe, 1 - missing_mask, np.nan)
                    imputation_saits = model_saits.impute(Xeval)
                    # print(f"SAITS imputation: {imputation_saits}")
                    imputation_saits = np.squeeze(imputation_saits)
                    imputed_saits = imputation_saits[row_indices, feature_idx]

                    ret_eval = copy.deepcopy(eval_)
                    ret_eval[row_indices, feature_idx] = np.nan
                    imputation_mice = model_mice.transform(ret_eval)
                    imputed_mice = imputation_mice[row_indices, feature_idx]
                    
                    ret_eval = copy.deepcopy(eval_)
                    ret_eval = unnormalize(ret_eval, mean, std, feature_idx)
                    ret_eval[row_indices, feature_idx] = np.nan
                    test_df = pd.DataFrame(ret_eval, columns=features)
                    add_season_id_and_save('./transformer/data_dir', test_df, filename='ColdHardiness_Grape_Merlot_test.csv')

                    params = {
                        'config_filepath': None, 
                        'output_dir': './transformer/output/', 
                        'data_dir': './transformer/data_dir/', 
                        'load_model': './transformer/output/mvts-model/checkpoints/model_best.pth', 
                        'resume': False, 
                        'change_output': False, 
                        'save_all': False, 
                        'experiment_name': 'MVTS_test',
                        'comment': 'imputation test', 
                        'no_timestamp': False, 
                        'records_file': 'Imputation_records.csv', 
                        'console': False, 
                        'print_interval': 1, 
                        'gpu': '0', 
                        'n_proc': 1, 
                        'num_workers': 0, 
                        'seed': None, 
                        'limit_size': None, 
                        'test_only': 'testset', 
                        'data_class': 'agaid', 
                        'labels': None, 
                        'test_from': './transformer/test_indices.txt', 
                        'test_ratio': 0, 
                        'val_ratio': 0, 
                        'pattern': None, 
                        'val_pattern': None, 
                        'test_pattern': 'Merlot_test', 
                        'normalization': 'standardization', 
                        'norm_from': None, 
                        'subsample_factor': None, 
                        'task': 'imputation', 
                        'masking_ratio': 0.15, 
                        'mean_mask_length': 10.0, 
                        'mask_mode': 'separate', 
                        'mask_distribution': 'geometric', 
                        'exclude_feats': None, 
                        'mask_feats': [0, 1], 
                        'start_hint': 0.0, 
                        'end_hint': 0.0, 
                        'harden': True, 
                        'epochs': 1000, 
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
                        'max_seq_len': 252, 
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

                    transformer_preds = run_transformer(params)
                    # # print(f'trasformer preds: {transformer_preds.shape}')
                    
                    imputation_transformer = np.squeeze(transformer_preds)
                    imputed_transformer = imputation_transformer[row_indices, feature_idx].cpu().detach().numpy()

                    ret_eval = copy.deepcopy(eval_)
                    # imputed_mean = unnormalize(ret_eval, mean, std, feature_idx)
                    imputed_mean = copy.deepcopy(ret_eval)
                    imputed_mean[row_indices, feature_idx] = 0
                    imputed_mean = imputed_mean[row_indices, feature_idx]

                    ret_eval[row_indices, feature_idx] = np.nan
                    ret_eval_df = pd.DataFrame(ret_eval, columns=features)
                    imputed_linear = ret_eval_df.interpolate(method='linear', limit_direction='both')
                    imputed_linear = imputed_linear.to_numpy()
                    imputed_linear = imputed_linear[row_indices, feature_idx]

                    real_values = eval_[row_indices, feature_idx]

                brits_mse += ((real_values - imputed_brits) ** 2).mean()
                saits_mse += ((real_values - imputed_saits) ** 2).mean()
                mice_mse += ((real_values - imputed_mice) ** 2).mean()
                transformer_mse += ((real_values - imputed_transformer) ** 2).mean()
                mean_mse += ((real_values - imputed_mean) ** 2).mean()
                linear_mse += ((real_values - imputed_linear) ** 2).mean()
                # print(f"real: {real_values}\nlinear mse: {linear_mse}")
                total_count += 1
                
            if total_count <= 0:
                neg+=1
                continue
            l_needed.append(l)
            # print(f"AVG MSE for {iter} runs (sliding window of Length = {l}):\n\tBRITS: {brits_mse/total_count}\n\tSAITS: {saits_mse/total_count}\n\tMVTS: {transformer_mse/total_count}\n\tMICE: {mice_mse/total_count}\n\tMEAN: {mean_mse/total_count}")
            print(f"AVG MSE for {iter} runs (sliding window of Length = {l}):\n\tLinear: {linear_mse/total_count}\n\tMEAN: {mean_mse/total_count}")
            # results['BRITS'][l] = brits_mse/total_count# f"MSE: {brits_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits)),5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits)), 5)}",
            # results['SAITS'][l] = saits_mse/total_count# f"MSE: {mice_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_mice)), 5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_mice)))}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_mice)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_mice)))}",
            # results['MICE'][l] = mice_mse/total_count
            # results['MVTS'][l] = transformer_mse/total_count
            results['MEAN'][l] = mean_mse/total_count
            results['Linear'][l] = linear_mse/total_count
            # result_mse_plots['BRITS'].append(brits_mse/total_count)
            # result_mse_plots['SAITS'].append(saits_mse/total_count)
            # result_mse_plots['MICE'].append(mice_mse/total_count)
            # result_mse_plots['MVTS'].append(transformer_mse/total_count)
            result_mse_plots['MEAN'].append(mean_mse/total_count)
            result_mse_plots['Linear'].append(linear_mse/total_count)
            
        end_time = time.time()
        result_df = pd.DataFrame(results)
        if not os.path.isdir(f'{mse_folder}/imputation_results/{eval_season}'):
            os.makedirs(f'{mse_folder}/imputation_results/{eval_season}')
        result_df.to_csv(f'{mse_folder}/imputation_results/{eval_season}/{given_feature}_results_impute.csv')
        # result_df.to_latex(f'{mse_folder}/{eval_type}/imputation_results/{given_feature}/{eval_season}/{given_feature}_results_impute.tex')

        plot_folder = f'{mse_folder}/plots/{eval_season}'
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)

        # x_axis = 
        # plt.figure(figsize=(16,9))
        # plt.plot(l_needed, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(l_needed, result_mse_plots['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend(fontsize=20)
        # plt.savefig(f'{plot_folder}/L-vs-MSE-BRITS-SAITS-{given_feature}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(20,13))
        # plt.plot(l_needed, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(l_needed, result_mse_plots['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.plot(l_needed, result_mse_plots['MVTS'], 'tab:cyan', label='MVTS', marker='o')
        # plt.plot(l_needed, result_mse_plots['MICE'], 'tab:purple', label='MICE', marker='o')
        # plt.plot(l_needed, result_mse_plots['MEAN'], 'm', label='MEAN', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=24)
        # plt.yticks(fontsize=24)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=24)
        # plt.ylabel(f'MSE', fontsize=24)
        # plt.legend(fontsize=24)
        # plt.savefig(f'{plot_folder}/L-vs-MSE-all-{given_feature}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(l_needed, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plot_folder}/L-vs-MSE-BRITS-{given_feature}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(l_needed, result_mse_plots['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plot_folder}/L-vs-MSE-SAITS-{given_feature}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(l_needed, result_mse_plots['MVTS'], 'tab:cyan', label='MVTS', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plot_folder}/L-vs-MSE-MVTS-{given_feature}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(l_needed, result_mse_plots['MICE'], 'tab:purple', label='MICE', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plot_folder}/L-vs-MSE-MICE-{given_feature}.png', dpi=300)
        # plt.close()


def forward_parse_id(fs, x, y, feature_impute_idx, trial_num=-1, all=False):

    idx_temp = np.where(~np.isnan(x[:,feature_impute_idx]))[0]
    # print(f"idx1: {idx_temp}")
    idx1 = idx_temp * len(features) + feature_impute_idx
    # print(f"idx1-1: {idx1}")
    # exit()

    # randomly eliminate 10% values as the imputation ground-truth
    # print('not null: ',np.where(~np.isnan(evals)))
    indices = idx1.tolist()

    start_idx = indices[(trial_num + 1)] #np.random.choice(indices, 1)
    print(f"indices: {indices}")
    start = indices.index(start_idx)
    # if start + length <= len(indices): 
    #     end = start + length
    # else:
    #     end = len(indices)
    end = len(indices)
    print(f"start: {start}, end: {end}")
    indices = np.array(indices)[start:end]

    # global real_values
    # real_values = evals[indices]
    if all:
        print(f"x: {x.shape}")
        x_copy = x.copy()
        # inv_indices = (indices - feature_impute_idx)//len(features)
        # if length > 1:
        # print('inv: ', inv_indices)
        features_to_nan = [features.index(f) for f in features if features.index(f) != feature_impute_idx]
        print(f'features: {features_to_nan}\nstart: {start_idx}')
        # for i in inv_indices:
        x_copy[idx_temp[trial_num + 2]:, features_to_nan] = np.nan 
        print(f"x copy: {x_copy}")
        evals = x_copy
    else:
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
    return indices



def forward_prediction(forward_folder):
    filename = 'json/json_eval_forward_LT'
    feature_idx = features.index('LTE50')
    X, Y = split_XY(season_df, max_length, season_array)
    season_names = ['2020-2021', '2021-2022']
    for given_season in season_names:
        season_idx = seasons[given_season]
        non_missing_indices = np.where(~np.isnan(X[season_idx, :, feature_idx]))[0]
        original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
        draws = []
        x_axis = []
        print(f"\n\nseason: {given_season}")
        for i in tqdm(range(len(non_missing_indices)-2)):
            print(f"i = {i}")
            fs = open(filename, 'w')
            print(f"season x: {X[season_idx]}")
            missing_indices = forward_parse_id(fs, X[season_idx], Y[season_idx], feature_idx, i)#, all=True)

            fs.close()
            if len(missing_indices) == 0:
                continue
            val_iter = data_loader.get_loader(batch_size=1, filename=filename)
            draw_data = {}
            for idx, data in enumerate(val_iter):
                data = utils.to_var(data)
                row_indices = missing_indices // len(features)
                print(f"rows: {row_indices}")
                ret = model_brits.run_on_batch(data, None)
                eval_ = ret['evals'].data.cpu().numpy()
                eval_ = np.squeeze(eval_)
                imputation_brits = ret['imputations'].data.cpu().numpy()
                imputation_brits = np.squeeze(imputation_brits)
                imputed_brits = imputation_brits[row_indices, feature_idx]

                ret_eval = copy.deepcopy(eval_)
                ret_eval[row_indices, feature_idx] = np.nan

                real_values = eval_[row_indices, feature_idx]

                brits_mse = ((real_values - imputed_brits) ** 2).mean()
                print(f"real: {real_values}\nbrits: {imputed_brits}\nmse: {brits_mse}\n")
                draws.append(brits_mse)
                x_axis.append(i+1)


                real_value = eval_[:, feature_idx]
                # real_values.extend(copy.deepcopy(real_value))
                
                
                draw_data['real'] = unnormalize(real_value, mean, std, feature_idx)
                draw_data['real'][original_missing_indices] = 0

                missing_values = copy.deepcopy(real_value)
                missing_values[row_indices] = 0
                draw_data['missing'] = unnormalize(missing_values, mean, std, feature_idx)
                draw_data['missing'][original_missing_indices] = 0
                draw_data['missing'][row_indices] = 0

                draw_data['BRITS'] = unnormalize(imputation_brits[:, feature_idx], mean, std, feature_idx)

                # draws['real'][row_indices] = 0
                draw_data_plot(draw_data, features[feature_idx], given_season, folder=f'data_folder_forward', existing=i)
                graph_bar_diff_multi(draw_data['real'][row_indices], draw_data, f'Difference From Gorund Truth for LTE50 in {given_season} existing = {i}', np.arange(len(row_indices)), 'Days of existing LTE50', 'Degrees', given_season, 'LTE50', missing=row_indices, existing=i)

        if not os.path.isdir(f'{forward_folder}/plots/LTE50/'):
            os.makedirs(f'{forward_folder}/plots/LTE50/')

        plt.figure(figsize=(16,9))
        plt.plot(x_axis, draws, 'tab:orange', label='BRITS', marker='o')
        plt.title(f'Length of previous existing LTE50 vs Imputation MSE for feature = {features[feature_idx]}, year={given_season}', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(f'Length of exisiting LTE50 values', fontsize=16)
        plt.ylabel(f'MSE', fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(f'{forward_folder}/plots/LTE50/L-vs-MSE-brits-{features[feature_idx]}-{given_season}.png', dpi=300)
        plt.close()
                

def forward_parse_id_day(fs, x, y, feature_impute_idx, existing_LT, trial_num=-1, all=False, same=True):

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
    return indices, values




def forward_prediction_LT_day(forward_folder, slide=True, same=True, data_folder=None, diff_folder=None):
    filename = 'json/json_eval_forward_LT'
    feature_idx = features.index('LTE50')
    X, Y, pads = split_XY(season_df, max_length, season_array, features, is_pad=True)
    season_names = ['2020-2021', '2021-2022']
    for given_season in season_names:
        season_idx = seasons[given_season]
        non_missing_indices = np.where(~np.isnan(X[season_idx, :, feature_idx]))[0]
        original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]

        draws_1 = {
            'BRITS': [],
            'SAITS': [],
            'MICE': [],
            'MVTS': []
        }

        draws_2 = {
            'BRITS': [],
            'SAITS': [],
            'MICE': [],
            'MVTS': []
        }
        # draws_1_brits = []
        # draws_1_saits = []
        # draws_2_brits = []
        # draws_2_saits = []
        draw_data = {}
        # draws_2 = []
        x_axis = []

        season_mse_1 = {
            'BRITS': [],
            'SAITS': [],
            'MICE': [],
            'MVTS': []
        }
        season_mse_2 = {
            'BRITS': [],
            'SAITS': [],
            'MICE': [],
            'MVTS': []
        }

        season_mse_3 = {
            'BRITS': [],
            'SAITS': [],
            'MICE': [],
            'MVTS': []
        }
        # season_mse_brits = []
        # season_mse_saits = []
        # season_mse_2_brits = []
        # season_mse_2_saits = []
        print(f"\n\nseason: {given_season}")
        for i in range(len(non_missing_indices)-3):
            # print(f"i = {i}")
            # mse_1_brits = 0
            # mse_1_saits = 0
            # mse_2_brits = 0
            # mse_2_saits = 0

            mse_1 = {
                'BRITS': 0,
                'SAITS': 0,
                'MICE': 0,
                'MVTS': 0
            }
            mse_2 = {
                'BRITS': 0,
                'SAITS': 0,
                'MICE': 0,
                'MVTS': 0
            }

            mse_3 = {
                'BRITS': 0,
                'SAITS': 0,
                'MICE': 0,
                'MVTS': 0
            }
            # mse_2 = 0
            trial_count = 0
            if slide:
                num_trials = len(non_missing_indices) - i - 3 - pads[season_idx]
            else:
                num_trials = 1
            for trial in tqdm(range(num_trials)):
                fs = open(filename, 'w')
                
                missing_indices, Xeval = forward_parse_id_day(fs, X[season_idx], Y[season_idx], feature_idx, i, trial_num=trial, all=True, same=same)
                
                fs.close()
                if len(missing_indices) == 0:
                    continue

                val_iter = data_loader.get_loader(batch_size=1, filename=filename)
                
                for idx, data in enumerate(val_iter):
                    data = utils.to_var(data)
                    row_indices = missing_indices // len(features)
                    with torch.no_grad():
                        # BRITS
                        ret = model_brits.run_on_batch(data, None)
                        eval_ = ret['evals'].data.cpu().numpy()
                        eval_ = np.squeeze(eval_)
                        imputation_brits = ret['imputations'].data.cpu().numpy()
                        imputation_brits = np.squeeze(imputation_brits)
                        imputed_brits = imputation_brits[row_indices, feature_idx]

                        # SAITS
                        Xeval = np.reshape(Xeval, (1, Xeval.shape[0], Xeval.shape[1]))
                        imputation_saits = model_saits.impute(Xeval)
                        imputation_saits = np.squeeze(imputation_saits)
                        imputed_saits = imputation_saits[row_indices, feature_idx]

                        # MICE
                        ret_eval = copy.deepcopy(eval_)
                        ret_eval[row_indices, feature_idx] = np.nan
                        imputation_mice = model_mice.transform(ret_eval)
                        imputed_mice = imputation_mice[row_indices, feature_idx]
                        
                        ret_eval = copy.deepcopy(eval_)
                        ret_eval = unnormalize(ret_eval, mean, std, feature_idx)
                        ret_eval[row_indices, feature_idx] = np.nan
                        test_df = pd.DataFrame(ret_eval, columns=features)
                        # print(f"test df: {test_df.columns}")
                        add_season_id_and_save('./transformer/data_dir', test_df, filename=f'ColdHardiness_Grape_Merlot_test_2.csv')

                        params = {
                            'config_filepath': None, 
                            'output_dir': './transformer/output/', 
                            'data_dir': './transformer/data_dir/', 
                            'load_model': f'./transformer/output/mvts-orig/checkpoints/model_best.pth', 
                            'resume': False, 
                            'change_output': False, 
                            'save_all': False, 
                            'experiment_name': 'MVTS_test_2',
                            'comment': 'imputation test', 
                            'no_timestamp': False, 
                            'records_file': 'Imputation_records.csv', 
                            'console': False, 
                            'print_interval': 1, 
                            'gpu': '-1', 
                            'n_proc': 1, 
                            'num_workers': 0, 
                            'seed': None, 
                            'limit_size': None, 
                            'test_only': 'testset', 
                            'data_class': 'agaid', 
                            'labels': None, 
                            'test_from': './transformer/test_indices.txt', 
                            'test_ratio': 0, 
                            'val_ratio': 0, 
                            'pattern': None, 
                            'val_pattern': None, 
                            'test_pattern': 'Merlot_test', 
                            'normalization': 'standardization', 
                            'norm_from': None, 
                            'subsample_factor': None, 
                            'task': 'imputation', 
                            'masking_ratio': 0.15, 
                            'mean_mask_length': 10.0, 
                            'mask_mode': 'separate', 
                            'mask_distribution': 'geometric', 
                            'exclude_feats': None, 
                            'mask_feats': [0, 1], 
                            'start_hint': 0.0, 
                            'end_hint': 0.0, 
                            'harden': True, 
                            'epochs': 1000, 
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
                            'max_seq_len': 252, 
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

                        transformer_preds = run_transformer(params)
                        # # print(f'trasformer preds: {transformer_preds.shape}')
                        
                        imputation_transformer = np.squeeze(transformer_preds)
                        imputed_transformer = imputation_transformer[row_indices, feature_idx].cpu().detach().numpy()


                        # REAL
                        real_values = eval_[row_indices, feature_idx]

                        # Same day MSE
                        mse_1['BRITS'] += ((real_values[0] - imputed_brits[0]) ** 2)
                        season_mse_1['BRITS'].append(((real_values[0] - imputed_brits[0]) ** 2))

                        mse_1['SAITS'] += ((real_values[0] - imputed_saits[0]) ** 2)
                        season_mse_1['SAITS'].append(((real_values[0] - imputed_saits[0]) ** 2))

                        mse_1['MICE'] += ((real_values[0] - imputed_mice[0]) ** 2)
                        season_mse_1['MICE'].append(((real_values[0] - imputed_mice[0]) ** 2))

                        mse_1['MVTS'] += ((real_values[0] - imputed_transformer[0]) ** 2)
                        season_mse_1['MVTS'].append(((real_values[0] - imputed_transformer[0]) ** 2))

                        # Next day MSE
                        mse_2['BRITS'] += ((real_values[1] - imputed_brits[1]) ** 2)
                        season_mse_2['BRITS'].append(((real_values[1] - imputed_brits[1]) ** 2))

                        mse_2['SAITS'] += ((real_values[1] - imputed_saits[1]) ** 2)
                        season_mse_2['SAITS'].append(((real_values[1] - imputed_saits[1]) ** 2))

                        mse_2['MICE'] += ((real_values[1] - imputed_mice[1]) ** 2)
                        season_mse_2['MICE'].append(((real_values[1] - imputed_mice[1]) ** 2))

                        mse_2['MVTS'] += ((real_values[1] - imputed_transformer[1]) ** 2)
                        season_mse_2['MVTS'].append(((real_values[1] - imputed_transformer[1]) ** 2))

                        mse_3['BRITS'] += ((real_values[2] - imputed_brits[2]) ** 2)
                        season_mse_3['BRITS'].append(((real_values[2] - imputed_brits[2]) ** 2))

                        mse_3['SAITS'] += ((real_values[2] - imputed_saits[2]) ** 2)
                        season_mse_3['SAITS'].append(((real_values[2] - imputed_saits[2]) ** 2))

                        mse_3['MICE'] += ((real_values[2] - imputed_mice[2]) ** 2)
                        season_mse_3['MICE'].append(((real_values[2] - imputed_mice[2]) ** 2))

                        mse_3['MVTS'] += ((real_values[2] - imputed_transformer[2]) ** 2)
                        season_mse_2['MVTS'].append(((real_values[2] - imputed_transformer[2]) ** 2))
                        # mse_2 += ((real_values[1] - imputed_brits[1]) ** 2)
                        trial_count += 1

                        real_value = eval_[:, feature_idx]
                    
                        draw_data['real'] = unnormalize(real_value, mean, std, feature_idx)
                        draw_data['real'][original_missing_indices] = 0

                        missing_values = copy.deepcopy(real_value)
                        missing_values[row_indices] = 0
                        draw_data['missing'] = unnormalize(missing_values, mean, std, feature_idx)
                        draw_data['missing'][original_missing_indices] = 0
                        draw_data['missing'][row_indices] = 0

                        draw_data['BRITS'] = unnormalize(imputation_brits[:, feature_idx], mean, std, feature_idx)
                        draw_data['SAITS'] = unnormalize(imputation_saits[:, feature_idx], mean, std, feature_idx)
                        draw_data['MICE'] = unnormalize(imputation_mice[:, feature_idx], mean, std, feature_idx)
                        draw_data['MVTS'] = unnormalize(imputation_transformer[:, feature_idx], mean, std, feature_idx).numpy()

                        # draws['real'][row_indices] = 0
                        if data_folder is not None:
                            draw_data_plot(draw_data, features[feature_idx], given_season, folder=data_folder, existing=i)
                        if diff_folder is not None:
                            graph_bar_diff_multi(diff_folder, draw_data['real'][row_indices], draw_data, f'Difference From Gorund Truth for LTE50 in {given_season} existing = {i+1}', np.arange(len(row_indices)), 'Days where LTE50 value is available', 'Degrees', given_season, 'LTE50', missing=row_indices, existing=i)

            if trial_count <= 0:
                continue
            mse_1['BRITS'] /= trial_count
            mse_1['SAITS'] /= trial_count
            mse_1['MICE'] /= trial_count
            mse_1['MVTS'] /= trial_count

            mse_2['BRITS'] /= trial_count
            mse_2['SAITS'] /= trial_count
            mse_2['MICE'] /= trial_count
            mse_2['MVTS'] /= trial_count

            mse_3['BRITS'] /= trial_count
            mse_3['SAITS'] /= trial_count
            mse_3['MICE'] /= trial_count
            mse_3['MVTS'] /= trial_count

            # mse_2 /= trial_count
            draws_1['BRITS'].append(mse_1['BRITS'])
            draws_1['SAITS'].append(mse_1['SAITS'])
            draws_1['MICE'].append(mse_1['MICE'])
            draws_1['MVTS'].append(mse_1['MVTS'])

            draws_2['BRITS'].append(mse_2['BRITS'])
            draws_2['SAITS'].append(mse_2['SAITS'])
            draws_2['MICE'].append(mse_2['MICE'])
            draws_2['MVTS'].append(mse_2['MVTS'])
            # draws_2.append(mse_2)
            x_axis.append(i+1)
        print(f"For season = {given_season} same day prediction results:\n\tBRITS mse = {np.array(season_mse_1['BRITS']).mean()}\n\tSAITS mse = {np.array(season_mse_1['SAITS']).mean()}\n\tMICE mse = {np.array(season_mse_1['MICE']).mean()}\n\tMVTS mse = {np.array(season_mse_1['MVTS']).mean()}")
        print(f"For season = {given_season} next day prediction results:\n\tBRITS mse = {np.array(season_mse_2['BRITS']).mean()}\n\tSAITS mse = {np.array(season_mse_2['SAITS']).mean()}\n\tMICE mse = {np.array(season_mse_2['MICE']).mean()}\n\tMVTS mse = {np.array(season_mse_2['MVTS']).mean()}")
        print(f"For season = {given_season} next 2 day prediction results:\n\tBRITS mse = {np.array(season_mse_3['BRITS']).mean()}\n\tSAITS mse = {np.array(season_mse_3['SAITS']).mean()}\n\tMICE mse = {np.array(season_mse_3['MICE']).mean()}\n\tMVTS mse = {np.array(season_mse_3['MVTS']).mean()}")
        
        ferguson_mse, ferguson_preds = get_FG(season_df, 'LTE50', 'PREDICTED_LTE50', season_array[season_idx])
        print(f"For season = {given_season} Ferguson mse = {ferguson_mse}")
        # result_df = pd.DataFrame(results)
        # folder = f'{mse_folder}/{feature}/remove-{r_feat}/mse_results'
        # if not os.path.isdir(folder):
        #     os.makedirs(folder)
        # result_df.to_csv(f'{folder}/results-mse-{season}.csv')
        
        # plots_folder = f'{forward_folder}/plots'
        # if not os.path.isdir(plots_folder):
        #     os.makedirs(plots_folder)

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_1['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-SAITS-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_2['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-SAITS-next-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_1['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.plot(x_axis, draws_1['MICE'], 'tab:cyan', label='MICE', marker='o')
        # plt.plot(x_axis, draws_1['MVTS'], 'tab:purple', label='MVTS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-all-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_2['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.plot(x_axis, draws_2['MICE'], 'tab:cyan', label='MICE', marker='o')
        # plt.plot(x_axis, draws_2['MVTS'], 'tab:purple', label='MVTS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-all-next-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_1['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.plot(x_axis, draws_1['MICE'], 'tab:cyan', label='MICE', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-SAITS-MICE-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_2['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.plot(x_axis, draws_2['MICE'], 'tab:cyan', label='MICE', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-SAITS-MICE-next-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_1['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.plot(x_axis, draws_1['MVTS'], 'tab:purple', label='MVTS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-SAITS-MVTS-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.plot(x_axis, draws_2['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.plot(x_axis, draws_2['MVTS'], 'tab:purple', label='MVTS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-SAITS-MVTS-next-{given_season}.png', dpi=300)
        # plt.close()


        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-SAITS-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['SAITS'], 'tab:blue', label='SAITS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-SAITS-next-{given_season}.png', dpi=300)
        # plt.close()


        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['BRITS'], 'tab:orange', label='BRITS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-BRITS-next-{given_season}.png', dpi=300)
        # plt.close()


        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['MICE'], 'tab:cyan', label='MICE', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-MICE-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['MICE'], 'tab:cyan', label='MICE', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-MICE-next-{given_season}.png', dpi=300)
        # plt.close()


        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1['MVTS'], 'tab:purple', label='MVTS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-MVTS-{given_season}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2['MVTS'], 'tab:purple', label='MVTS', marker='o')
        # plt.title(f'Length of existing values vs Imputation MSE for feature = LTE50, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel(f'Length of existing observed values', fontsize=20)
        # plt.ylabel(f'MSE', fontsize=20)
        # plt.legend()
        # plt.savefig(f'{plots_folder}/forward-LT-MSE-MVTS-next-{given_season}.png', dpi=300)
        # plt.close()


        # if not os.path.isdir(f'{forward_folder}/plots/LTE50/'):
        #     os.makedirs(f'{forward_folder}/plots/LTE50/')

        # slide_or_not = 'slide'
        # if not slide:
        #     slide_or_not = 'non-slide'
        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_1, 'tab:orange', label='BRITS', marker='o')
        # plt.title(f"Length of previous existing LTE50 vs Imputation MSE ({'same' if same else 'next'} day) for feature = {features[feature_idx]}, year={given_season}", fontsize=20)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.xlabel(f'Length of exisiting LTE50 values', fontsize=16)
        # plt.ylabel(f'MSE', fontsize=16)
        # plt.legend(fontsize=16)
        # plt.savefig(f"{forward_folder}/plots/LTE50/L-vs-MSE-brits-{'same' if same else 'next'}-{features[feature_idx]}-{given_season}-{slide_or_not}.png", dpi=300)
        # plt.close()



        # plt.figure(figsize=(16,9))
        # plt.plot(x_axis, draws_2, 'tab:orange', label='BRITS', marker='o')
        # plt.title(f'Length of previous existing LTE50 vs Imputation MSE (next day) for feature = {features[feature_idx]}, year={given_season}', fontsize=20)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.xlabel(f'Length of exisiting LTE50 values', fontsize=16)
        # plt.ylabel(f'MSE', fontsize=16)
        # plt.legend(fontsize=16)
        # plt.savefig(f'{forward_folder}/plots/LTE50/L-vs-MSE-brits-next-{features[feature_idx]}-{given_season}-{slide_or_not}.png', dpi=300)
        # plt.close()








####################### Draw data plots #######################

def do_data_plots(data_folder, missing_length, is_original=False):
    # print(f'Season array: {len(season_array)}')
    filename = 'json/json_eval_3_LT'
    missing_num = missing_length
    if is_original:
        data_folder += '/original_missing'
    else:
        data_folder += '/eval_missing'
    for given_season in seasons.keys():
        season_idx = seasons[given_season]
        given_features = ['LTE50']#features
        print(f"season: {given_season}")
        for given_feature in tqdm(given_features):
            fs = open(filename, 'w')
            X, Y = split_XY(season_df, max_length, season_array)

            feature_idx = features.index(given_feature)

            original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
            # print(f"feature: {given_feature}, season: {given_season}, season idx: {seasons[given_season]}, Original missing: {len(original_missing_indices)}")
            
            if given_feature.split('_')[-1] not in feature_dependency.keys():
                dependent_feature_ids = []
            else:
                dependent_feature_ids = [features.index(f) for f in feature_dependency[given_feature.split('_')[-1]] if f != given_feature]
            if is_original:
                missing_indices = parse_id(fs, X[season_idx], Y[season_idx], feature_idx, missing_num, dependent_features=dependent_feature_ids, real_test=False)
            else:
                missing_indices = parse_id(fs, X[season_idx], Y[season_idx], feature_idx, missing_num, dependent_features=dependent_feature_ids)
            fs.close()
            if missing_indices is not None and len(missing_indices) == 0:
                continue 

            val_iter = data_loader.get_loader(batch_size=1, filename=filename)
            draws = {}
            for idx, data in enumerate(val_iter):
                data = utils.to_var(data)
                if not is_original:
                    row_indices = missing_indices // len(features)

                ret = model_brits.run_on_batch(data, None)
                eval_ = ret['evals'].data.cpu().numpy()
                eval_ = np.squeeze(eval_)
                
                real_value = eval_[:, feature_idx]
                # real_values.extend(copy.deepcopy(real_value))
                
                
                draws['real'] = unnormalize(real_value, mean, std, feature_idx)
                draws['real'][original_missing_indices] = 0
                # draws['real'][row_indices] = 0
                if not is_original:
                    missing_values = copy.deepcopy(real_value)
                    missing_values[row_indices] = 0
                    draws['missing'] = unnormalize(missing_values, mean, std, feature_idx)
                    draws['missing'][original_missing_indices] = 0
                    draws['missing'][row_indices] = 0

                imputation_brits = ret['imputations'].data.cpu().numpy()
                imputation_brits = np.squeeze(imputation_brits)
                draws['BRITS'] = unnormalize(imputation_brits[:, feature_idx], mean, std, feature_idx)

                ret_eval = copy.deepcopy(eval_)
                # ret_eval[row_indices, feature_idx] = np.nan
                ret_eval[original_missing_indices, feature_idx] = np.nan

                # imputation_mice = mice_impute.transform(ret_eval)
                # draws['MICE'] = unnormalize(imputation_mice[:, feature_idx], mean, std, feature_idx)

                draw_data_plot(draws, features[feature_idx], given_season, folder=data_folder, is_original=is_original)

                

# eval_folder = 'eval_dir_abstract_basline/'
# if not os.path.isdir(eval_folder):
#     os.makedirs(eval_folder)
# do_evaluation(eval_folder, 'cont', '2020-2021')
# do_evaluation(eval_folder, 'cont', '2021-2022')
# data_plots_folder = 'data_plots_LT/LT'
# if not os.path.isdir(data_plots_folder):
#     os.makedirs(data_plots_folder)
# do_data_plots(data_plots_folder, 10, is_original=True)
# do_data_plots(data_plots_folder, 10, is_original=False)

# forward_folder = 'forward_LT_brits_saits_13'
# forward_data_folder = 'forward_LT_data_brits_saits_13'
# forward_prediction_LT_day(forward_folder, slide=True)# data_folder=forward_data_folder)

# forward_folder = 'forward_LT_2'
# forward_data_folder = None#f"{forward_folder}/data"
# forward_diff_folder = None#f"{forward_folder}/diff"
# forward_prediction_LT_day(forward_folder, slide=False, data_folder=forward_data_folder, diff_folder=forward_diff_folder)
# forward_folder = 'forward_LT_abstract_2'
# forward_prediction_LT_day(forward_folder, same=False)
# forward_prediction_LT_day(forward_folder, slide=False, data_folder=forward_data_folder)
# forward_prediction_LT_day(forward_folder, slide=True)#, data_folder=forward_data_folder)#, same=False)


# data_plots_LT = f'{forward_folder}/data_plots'

# if not os.path.isdir(data_plots_LT):
#     os.makedirs(data_plots_LT)

# diff_plots_LT = f'{forward_folder}/diffs'

# if not os.path.isdir(diff_plots_LT):
#     os.makedirs(diff_plots_LT)

# forward_prediction_LT_day(forward_folder, data_folder=data_plots_LT, diff_folder=diff_plots_LT, slide=False)

evaluate_imputation(None)