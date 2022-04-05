import os
import time
import numpy as np

from naomi.model import *
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.utils
import torch.utils.data
from process_data import *
import json
from naomi.helpers import *
import data_loader
import copy


############## Draw Functions ##############
def _graph_bar_diff_multi(GT_values, result_dict, title, x, xlabel, ylabel, i, drop_linear=False):
    plot_dict = {}
    plt.figure(figsize=(16,9))
    for key, value in result_dict.items():
      plot_dict[key] = np.abs(GT_values) - np.abs(value)
    # ind = np.arange(prediction.shape[0])
    x = np.array(x)
    width = 0.3
    pos = 0
    remove_keys = ['GT']
    if drop_linear:
      remove_keys.append('LinearInterp')
    for key, value in plot_dict.items():
        if key not in remove_keys:
          plt.bar(x + pos + 1, value, width, label = key)
          pos += width

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis([0, 100, -2, 3])

    plt.legend(loc='best')
    plt.savefig(f'diff_imgs/result-diff-{i}.png', dpi=300)
    plt.close()
    # plt.show()
    return


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

def draw_data_plot(results, f, season_idx):
    plt.figure(figsize=(32,18))
    plt.title(f"For feature = {f}")
    ax = plt.subplot(321)
    ax.set_title(f+' original data')
    plt.plot(np.arange(results['real'].shape[0]), results['real'])
    ax = plt.subplot(322)
    ax.set_title(f+' missing data')
    plt.plot(np.arange(results['missing'].shape[0]), results['missing'])
    ax = plt.subplot(323)
    ax.set_title('feature = '+f+' imputed by BRITS')
    plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'])
    ax = plt.subplot(324)
    ax.set_title('feature = '+f+' imputed by BRITS_I')
    plt.plot(np.arange(results['BRITS_I'].shape[0]), results['BRITS_I'])
    ax = plt.subplot(325)
    ax.set_title('feature = '+f+' imputed by KNN')
    plt.plot(np.arange(results['KNN'].shape[0]), results['KNN'])
    ax = plt.subplot(326)
    ax.set_title('feature = '+f+' imputed by MICE')
    plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'])
    plt.savefig(f"subplots/{f}-imputations-season-{season_idx}.png", dpi=300)
    plt.close()

def draw_data_trend(results, f, season_idx):
    plt.figure(figsize=(16,9))
    plt.title(f"For feature = {f} and season = {season_idx}", fontsize=22)
    plt.plot(np.arange(len(results)), results, 'tab:blue')
    plt.xlabel('Days of a season', fontsize=18)
    plt.ylabel('Values', fontsize=18)
    # plt.tight_layout()
    plt.savefig(f"missing-{f}-{season_idx}.png", dpi=300)
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


def parse_id(x, y, feature_impute_idx, length, trial_num=-1):
    evals = x

    evals = (evals - mean) / std
    # print('eval: ', evals)
    # print('eval shape: ', evals.shape)
    shp = evals.shape
    
    idx1 = np.where(~np.isnan(evals[:,feature_impute_idx]))[0]
    idx1 = idx1 * len(features) + feature_impute_idx
    # exit()

    evals = evals.reshape(-1)
    # randomly eliminate 10% values as the imputation ground-truth
    # print('not null: ',np.where(~np.isnan(evals)))
    indices = idx1.tolist()
    if trial_num == -1:
        start_idx = np.random.choice(indices, 1)
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

    values = evals.copy()
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
    # for key in rec.keys():
    #     print(f"{key}: {type(rec[key])}")# and {rec[key].shape}")
    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return indices


folder = './json/'
file = 'json_eval_naomi'
filename = folder + file

df = pd.read_csv('ColdHardiness_Grape_Merlot.csv')
modified_df, dormant_seasons = preprocess_missing_values(df)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons)
mean, std = get_mean_std(season_df, features)

given_feature = 'MEAN_AT'

fs = open(filename, 'w')
X, Y = split_XY(season_df, max_length, season_array)

test_X = X[-1]
test_Y = Y[-1]

feature_idx = features.index(given_feature)
# for i in range(X.shape[0]):
missing_indices = parse_id(X[-1], Y[-1], feature_idx, 10)
fs.close()

params = {
    'task' : '',#args.task,
    'batch' : 16, #args.batch_size,
    'y_dim' : 20, #args.y_dim,
    'rnn_dim' : 300, #args.rnn_dim,
    'dec1_dim' : 200, #args.dec1_dim,
    'dec2_dim' : 200, #args.dec2_dim,
    'dec4_dim' : 200, #args.dec4_dim,
    'dec8_dim' : 200, #args.dec8_dim,
    'dec16_dim' : 200, #args.dec16_dim,
    'n_layers' : 2, #args.n_layers,
    'discrim_rnn_dim' : 128, #args.discrim_rnn_dim,
    'discrim_num_layers' : 1, #args.discrim_layers,
    'cuda' : True, #args.cuda,
    'highest' : 8, #args.highest,
}
# model = NAOMI(params)
# if os.path.exists('saved/NAOMI_001/policy_step8_training.pth'):
#     model.load_state_dict(torch.load('saved/NAOMI_001/policy_step8_training.pth'))

idx = 19
real_values = season_df[given_feature].loc[season_array[idx]]
print(season_array[idx])
print(np.isnan(real_values).sum())
real_values[np.isnan(real_values)] = 0
draw_data_trend(real_values, features[feature_idx], '2007-2008')

# val_iter = data_loader.get_loader(batch_size=1, filename=filename)
# draws = {}
# for idx, data in enumerate(val_iter):
#     row_indices = missing_indices // len(features)
#     evals_ = data['forward']['evals']
#     data_list = []
#     print(evals_.shape)
#     # evals = evals_.transpose(0, 1)

#     mse, imputed_values = run_test(model, evals_, 1, row_indices, feature_idx)
#     imputed_values = imputed_values[0].transpose(1,0).squeeze().detach().numpy()
#     print(f"real values: {real_values.shape}, imputed: {imputed_values.shape}")
#     missing_values = copy.deepcopy(real_values)
#     missing_values[row_indices] = 0
#     draws['real'] = real_values
#     draws['missing'] = missing_values

#     # print(f"real: {real_values.shape}, imputed: {imputed_values[0].shape}")
#     mse = np.round(((real_values - imputed_values[:, feature_idx].squeeze()) ** 2).mean(), 5)
#     print(f"mse: {mse}")
