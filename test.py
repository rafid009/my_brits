import copy
from operator import index
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from process_data import *
# from input_process import *
import json
import os
import torch
from models.brits import BRITSModel as BRITS
from models.brits_i import BRITSModel as BRITS_I
import utils
import data_loader
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.3
LABEL_WEIGHT = 1


folder = './json/test/'
if not os.path.exists(folder):
    os.makedirs(folder)
fs = open(folder+'json', 'w')

real_values = None

mean = []
std = []

# features_impute = [features.index('MEAN_AT'), features.index('AVG_REL_HUMIDITY')]

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
    
def unnormalize(X, mean, std, feature_idx=-1):
    if feature_idx == -1:
        return (X * std) + mean
    else:
        return (X * std[feature_idx]) + mean[feature_idx]

def parse_id(x, y):
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
    
    # idx1 = np.where(~np.isnan(evals[:,features_impute[0]]))[0]
    # idx2 = np.where(~np.isnan(evals[:,features_impute[1]]))[0]
    # idx1 = idx1 * len(features) + features_impute[0]
    # idx2 = idx2 * len(features) + features_impute[1]
    # exit()

    evals = evals.reshape(-1)
    # randomly eliminate 10% values as the imputation ground-truth
    # print('not null: ',np.where(~np.isnan(evals)))
    # indices = np.concatenate((idx1, idx2)).tolist()
    # indices = np.random.choice(indices, len(indices) // 10)
    
    # global real_values
    # real_values = evals[indices]

    values = evals.copy()
    # values[indices] = np.nan

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

train_df = pd.read_csv("ColdHardiness_Grape_Merlot_2.csv")
print('Now train')
train_modified_df, train_dormant_seasons = preprocess_missing_values(train_df, is_dormant=False, is_year=True)
train_season_df, train_season_array, train_max_length = get_seasons_data(train_modified_df, train_dormant_seasons, is_dormant=False, is_year=True)
train_season_df = train_season_df.drop(train_season_array[-1], axis=0)
train_season_df = train_season_df.drop(train_season_array[-2], axis=0)
train_season_df = train_season_df.drop(train_season_array[-3], axis=0)
# train_season_df = train_season_df.drop(train_season_array[-4], axis=0)
mean, std = get_mean_std(train_season_df, features)

normalized_season_df = train_season_df[features].copy()
# print(f"norm: {normalized_season_df.shape}, mean: {mean.shape}, std: {std.shape}")
normalized_season_df = (normalized_season_df - mean) /std

mice_impute = IterativeImputer(random_state=0, max_iter=20)
mice_impute.fit(normalized_season_df[features])


print('Now test')
test_df = pd.read_csv("ColdHardiness_Grape_Merlot_2.csv")
modified_df, dormant_seasons = preprocess_missing_values(test_df, is_dormant=False)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, is_dormant=False)
# print('season array: ', len(season_array), '\n', season_array)

test_normalized_df = season_df[features].copy()
test_normalized_df = (test_normalized_df - mean) /std
test_imputed_mice = mice_impute.transform(test_normalized_df[features])
test_imputed = unnormalize(test_imputed_mice, mean, std)

print(f"{test_df.iloc[11824]}")

mice_df = test_df.copy()
mice_df[features] = np.round(test_imputed, 2)
mice_df.iloc[11824:] = test_df.iloc[11824:] 
mice_df.to_csv('ColdHardiness_Grape_Merlot_imputed_predormant_mice.csv', index=False)

X, Y = split_XY(season_df, max_length, season_array)

print('X: ', X.shape)
zero_pads = []
for i in range(X.shape[0]):
    indices = np.where(~X[i].any(axis=1))[0]
    # print('zero pads: ', indices, '\n\n')
    zero_pads.append(indices)
    parse_id(X[i], Y[i])
fs.close()


model_brits = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

if os.path.exists('./model_brits.model'):
    model_brits.load_state_dict(torch.load('./model_brits.model'))

model_brits.eval()

print('season array loader: ', len(season_array))
val_iter = data_loader.get_loader(batch_size=len(season_array), filename=folder + 'json', shuffle=False)

imputed_array_brits = None
for idx, data in enumerate(val_iter):
    data = utils.to_var(data)
    ret = model_brits.run_on_batch(data, None)
    eval_ = ret['evals'].data.cpu().numpy()
    print('idx: ', idx, ' eval: ', eval_.shape)
    imputation_brits = ret['imputations'].data.cpu().numpy()
    imputation_brits = unnormalize(imputation_brits, mean, std)
    print(f"brits imputed seasons: {len(imputation_brits)}")
    for i in range(imputation_brits.shape[0]):
        # print(imputation_brits[i])
        without_paddings = np.delete(imputation_brits[i], zero_pads[i], 0)
        if imputed_array_brits is None:
            imputed_array_brits = np.round(without_paddings, 2)
        else:
            imputed_array_brits = np.concatenate((imputed_array_brits, np.round(without_paddings, 2)), axis=0)
brits_df = test_df.copy()
brits_df[features] = imputed_array_brits
brits_df.iloc[11824:] = test_df.iloc[11824:] 
print(f"test: {test_df.iloc[12088]}")
print(f"brits: {brits_df.iloc[12088]}")
brits_df.to_csv('ColdHardiness_Grape_Merlot_imputed_predormant_brits.csv', index=False)



# evals = np.array(evals)
# imputations_brits = np.array(imputations_brits)

# imputations_brits_I = np.array(imputations_brits_I)
# print("imp len: ", len(imputations_brits))
# result_dict = {
#     # 'knn': X_knn_imputed,
#     'brits': imputations_brits,
#     'brits_I': imputations_brits_I
# }

# limit = 100
# start = 0

# for i in range(int(len(imputations_brits)/limit)):
#     end = start + limit + 1
#     result_dict = {
#         'brits': imputations_brits[start:end],
#         'brits_I': imputations_brits_I[start:end]
#     }
#     _graph_bar_diff_multi(evals[start:end], result_dict, "Difference of the different imputations from GT", np.arange(len(imputations_brits[start:end])), "# of missing values from observed data", "imputed value difference form GT", i)
#     start = end

# start = 0

# for i in range(int(len(imputations_brits)/limit)):
#     end = start+limit+1
#     x_range = np.arange(len(imputations_brits[start:end]))
#     plt.figure(figsize=(16,9))
#     plt.plot(x_range, imputations_brits[start:end], 'r', label='BRITS')
#     plt.plot(x_range, imputations_brits_I[start:end], 'b', label='BRITS_I')
#     plt.plot(x_range, evals[start:end], 'g', label='Ground Truth')
#     plt.title('GT vs Bidirectional RNN based imputations')
#     plt.xlabel('# of missing values from observed data')
#     plt.ylabel('Actual/imputed values (normalized)')
#     plt.legend()
#     plt.savefig(f'plot_imgs/gt-imputed-{i}.png', dpi=300)
#     plt.close()
#     start = end