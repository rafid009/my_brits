# coding: utf-8

import os
import numpy as np
import pandas as pd
from process_data import *
import json


attributes = features

folder = './json/'
if not os.path.exists(folder):
    os.makedirs(folder)
fs = open(folder+'json_without_LT', 'w')

# def to_time_bin(x):
#     h, m = map(int, x.split(':'))
#     return h


# def parse_data(x):
#     x = x.set_index('Parameter').to_dict()['Value']

#     values = []

#     for attr in attributes:
#         if x.has_key(attr):
#             values.append(x[attr])
#         else:
#             values.append(np.nan)
#     return values


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(masks.shape[0]):
        if h == 0:
            deltas.append(np.ones(len(attributes)))
        else:
            deltas.append(np.ones(len(attributes)) + (1 - masks[h]) * deltas[-1])

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


def parse_id(x, y, mean, std):
    # data = pd.read_csv('./raw/{}.txt'.format(id_))
    # accumulate the records within one hour
    # data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))
    
    evals = x
    # print('x: ', x)

    # merge all the metrics within one hour
    # for h in range(48):
    #     evals.append(parse_data(data[data['Time'] == h]))
    evals = (evals - mean) / std
    # print('eval: ', evals)
    # print('eval shape: ', evals.shape)
    shp = evals.shape

    evals = evals.reshape(-1)

    # randomly eliminate 10% values as the imputation ground-truth
    # print('not null: ',np.where(~np.isnan(evals)))
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 20)
    

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

# df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
# modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#False, is_year=True)
# season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)
# # idx_LT_not_null = get_non_null_LT(season_df)
# # train_idx = get_train_idx(season_array, idx_LT_not_null)
# X, Y = split_XY(season_df, max_length, season_array, features)
# print(f"X: {X.shape}")
# X = X[:-2]
# Y = Y[:-2]


# train_season_df = season_df.drop(season_array[-1], axis=0)
# train_season_df = train_season_df.drop(season_array[-2], axis=0)
# # train_season_df = train_season_df.drop(season_array[-3], axis=0)
# # train_season_df = train_season_df.drop(season_array[-4], axis=0)
# mean, std = get_mean_std(train_season_df, features)
# print(f"X: {X.shape}")
# # print('season mean at: ',np.where(~np.isnan(season_npy)))

# mean = np.array(mean) #np.mean(season_df[attributes].to_numpy(), axis=0)
# std = np.array(std) #np.std(season_df[attributes].to_numpy(), axis=0)


def prepare_brits_input(df, season_array, max_length, features, mean, std, model_dir='./model_abstract', complete_seasons=None):
    if complete_seasons is not None:
        seasons = [season_array[s] for s in complete_seasons]
    else:
        seasons = season_array
    X, Y = split_XY(df, max_length, seasons, features)
    np.save(f'{model_dir}/mean.npy', mean)
    np.save(f'{model_dir}/std.npy', std)
    X = X[:-2]
    Y = Y[:-2]
    for i in range(X.shape[0]):
        parse_id(X[i], Y[i], mean, std)

    fs.close()

# if __name__ == "__main__":
#     prepare_brits_input()