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
from sklearn.impute import IterativeImputer
from pypots.data import mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae, cal_mse
from transformer.src.transformer import run_transformer, add_season_id_and_save
import pickle 
import warnings
import sys
warnings.filterwarnings("ignore")

np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_random = 0.2

RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.3
LABEL_WEIGHT = 1




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

def parse_rec(values, masks, evals, eval_masks, dir_, features):
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

def parse_id(x, y, fs, mean, std, features):
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
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward', features=features)
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', features=features)
    # for key in rec.keys():
    #     print(f"{key}: {type(rec[key])}")# and {rec[key].shape}")
    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return values
complete_seasons = [4, 5, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]



train_df = pd.read_csv(f"ColdHardiness_Grape_Merlot_2.csv")#new_synthetic_{n_random}.csv")
# print('Now train')
train_modified_df, train_dormant_seasons = preprocess_missing_values(train_df, features, is_dormant=True)#, is_year=True)
train_season_df, train_season_array, train_max_length = get_seasons_data(train_modified_df, train_dormant_seasons, features, is_dormant=True)#, is_year=True)
# train_season_array = [train_season_array[i] for i in complete_seasons]

# train_complete = []
# for i in range(len(train_season_array)):
#     indices = copy.deepcopy(train_season_array[i])
#     train_complete.extend(indices)
# train_season_df = train_season_df.loc[train_complete]
train_season_df = train_season_df.drop(train_season_array[-1], axis=0)
train_season_df = train_season_df.drop(train_season_array[-2], axis=0)
# train_season_df = train_season_df.drop(train_season_array[-3], axis=0)
# train_season_df = train_season_df.drop(train_season_array[-4], axis=0)
mean, std = get_mean_std(train_season_df, features)

# normalized_season_df = train_season_df[features].copy()
# # print(f"norm: {normalized_season_df.shape}, mean: {mean.shape}, std: {std.shape}")
# normalized_season_df = (normalized_season_df - mean) /std

# mice_impute = IterativeImputer(random_state=0, max_iter=20)
# mice_impute.fit(normalized_season_df[features])


print('Now test')
test_df = pd.read_csv(f"ColdHardiness_Grape_Merlot_2.csv")#new_synthetic_{n_random}.csv")
modified_df, dormant_seasons = preprocess_missing_values(test_df, features, is_dormant=True)#, is_year=True)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#, is_year=True)
# season_array = [season_array[i] for i in complete_seasons]
# seasons_complete = []
# for i in range(len(season_array)):
#     indices = copy.deepcopy(season_array[i])
#     seasons_complete.extend(indices)
# season_df = season_df.loc[seasons_complete]
# print('season array: ', len(season_array), '\n', season_array)



# print(f"{test_df.iloc[11824]}")

# mice_df = test_df.copy()
# mice_df[features] = np.round(test_imputed, 2)
# mice_df.iloc[11824:] = test_df.iloc[11824:] 
# mice_df.to_csv('ColdHardiness_Grape_Merlot_imputed_yearly_mice.csv', index=False)

X, Y = split_XY(season_df, max_length, season_array, features)
folder = './json/test/'
if not os.path.exists(folder):
    os.makedirs(folder)
fs = open(folder+'json', 'w')
print('X: ', X.shape)
zero_pads = []
Xeval = copy.deepcopy(X)
for i in range(X.shape[0]):
    indices = np.where(~X[i].any(axis=1))[0]
    # print('zero pads: ', indices, '\n\n')
    zero_pads.append(indices)
    Xeval[i] = parse_id(X[i], Y[i], fs, mean, std, features)
fs.close()

model_brits = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT, feature_len=len(features))

if os.path.exists(f'./model_abstract/model_BRITS_LT_synth_{n_random}.model'):
    model_brits.load_state_dict(torch.load(f'./model_abstract/model_BRITS_LT_synth_{n_random}.model'))
model_brits.to(device=device)
model_brits.eval()

saits_file = f'./model_abstract/model_saits_synth_{n_random}.model'
model_saits = pickle.load(open(saits_file, 'rb'))

mice_file = f'./model_abstract/model_mice_synth_{n_random}.model'
model_mice = pickle.load(open(mice_file, 'rb'))

test_normalized_df = season_df[features].copy()
test_normalized_df = (test_normalized_df - mean) /std
test_imputed_mice = model_mice.transform(test_normalized_df[features])
imputation_mice = unnormalize(test_imputed_mice, mean, std)

add_season_id_and_save('./transformer/data_dir', season_df[features], season_array=season_array, filename=f'ColdHardiness_Grape_Merlot_test_{n_random}.csv')

params = {
    'config_filepath': None, 
    'output_dir': './transformer/output/', 
    'data_dir': './transformer/data_dir/', 
    'load_model': f'./transformer/output/mvts-orig/checkpoints/model_best.pth', 
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
    'batch_size': len(X), 
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



print('season array loader: ', len(season_array))
val_iter = data_loader.get_loader(batch_size=len(season_array), filename=folder + 'json', shuffle=False)

imputed_array_brits = None
imputed_array_saits = None
imputed_array_mvts = None
for idx, data in enumerate(val_iter):
    transformer_preds = run_transformer(params)
    transformer_preds = unnormalize(transformer_preds.cpu().detach().numpy(), mean, std)
    # print(f"mvts: {transformer_preds.shape}")
    # print(f'season_df: {season_df[features].shape}')
    data = utils.to_var(data)
    ret = model_brits.run_on_batch(data, None)
    eval_ = ret['evals'].data.cpu().numpy()
    # # print('idx: ', idx, ' eval: ', eval_.shape)
    imputation_brits = ret['imputations'].data.cpu().numpy()
    imputation_brits = unnormalize(imputation_brits, mean, std)
    # print(f"imputed brits: {np.isnan(imputation_brits).sum()}")
    # print(f"brits imputed seasons: {len(imputation_brits)}")

    Xeval = np.reshape(Xeval, (len(season_array), Xeval.shape[1], Xeval.shape[2]))
    # X_intact, Xe, missing_mask, indicating_mask = mcar(Xeval, 0.1) # hold out 10% observed values as ground truth
    
    # Xe = masked_fill(Xe, 1 - missing_mask, np.nan)
    imputation_saits = model_saits.impute(Xeval)
    imputation_saits = unnormalize(imputation_saits, mean, std)

    # imputation_mice = model_mice.transform(season_df[features].to_numpy())

    
    for i in range(imputation_brits.shape[0]):
        # print(imputation_brits[i])
        without_paddings = np.delete(imputation_brits[i], zero_pads[i], 0)
        # print(f"without pd: {without_paddings}")
        if imputed_array_brits is None:
            imputed_array_brits = np.round(without_paddings, 2)
        else:
            imputed_array_brits = np.concatenate((imputed_array_brits, np.round(without_paddings, 2)), axis=0)

    for i in range(imputation_saits.shape[0]):
        # print(imputation_brits[i])
        without_paddings = np.delete(imputation_saits[i], zero_pads[i], 0)
        # print(f"without pd: {without_paddings}")
        if imputed_array_saits is None:
            imputed_array_saits = np.round(without_paddings, 2)
        else:
            imputed_array_saits = np.concatenate((imputed_array_saits, np.round(without_paddings, 2)), axis=0)
    print(f"transformer preds: {transformer_preds.shape}")
    for i in range(transformer_preds.shape[0]):
        # print(imputation_brits[i])
        without_paddings = np.delete(transformer_preds[i], zero_pads[i], 0)
        # print(f"without pd: {without_paddings}")
        if imputed_array_mvts is None:
            imputed_array_mvts = np.round(without_paddings, 2)
        else:
            imputed_array_mvts = np.concatenate((imputed_array_mvts, np.round(without_paddings, 2)), axis=0)
print(f"season indices: {len(season_df.index.tolist())}")

model_name = f"brits_orig"#synth_{n_random}"

brits_df = test_df.copy()
brits_df.loc[season_df.index.tolist(), features] = imputed_array_brits
brits_df['LTE50'] = test_df['LTE50']
data_imputed_folder = './abstract_imputed'
if not os.path.isdir(data_imputed_folder):
    os.makedirs(data_imputed_folder)
filename = 'ColdHardiness_Grape_Merlot_imputed'

brits_df.to_csv(f"{data_imputed_folder}/{filename}_{model_name}.csv", index=False)

model_name = f"saits_orig"#synth_{n_random}"

saits_df = test_df.copy()
saits_df.loc[season_df.index.tolist(), features] = imputed_array_saits
saits_df['LTE50'] = test_df['LTE50']
data_imputed_folder = './abstract_imputed'
if not os.path.isdir(data_imputed_folder):
    os.makedirs(data_imputed_folder)
filename = 'ColdHardiness_Grape_Merlot_imputed'

saits_df.to_csv(f"{data_imputed_folder}/{filename}_{model_name}.csv", index=False)

model_name = f"mice_orig"#synth_{n_random}"

mice_df = test_df.copy()
mice_df.loc[season_df.index.tolist(), features] = imputation_mice
mice_df['LTE50'] = test_df['LTE50']
data_imputed_folder = './abstract_imputed'
if not os.path.isdir(data_imputed_folder):
    os.makedirs(data_imputed_folder)
filename = 'ColdHardiness_Grape_Merlot_imputed'

mice_df.to_csv(f"{data_imputed_folder}/{filename}_{model_name}.csv", index=False)

model_name = f"mvts_orig"#synth_{n_random}"

mvts_df = test_df.copy()
print(f"season df: {season_df.shape}\nimputed array: {imputed_array_mvts.shape}")
mvts_df.loc[season_df.index.tolist(), features] = imputed_array_mvts
mvts_df['LTE50'] = test_df['LTE50']
data_imputed_folder = './abstract_imputed'
if not os.path.isdir(data_imputed_folder):
    os.makedirs(data_imputed_folder)
filename = 'ColdHardiness_Grape_Merlot_imputed'

mvts_df.to_csv(f"{data_imputed_folder}/{filename}_{model_name}.csv", index=False)

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