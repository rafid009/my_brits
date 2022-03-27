import copy
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
from models.brits_i import BRITSModel as BRITS_I
import utils
import data_loader
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import time
import warnings
warnings.filterwarnings("ignore")

RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.3
LABEL_WEIGHT = 1


folder = './json/'
file = 'json_eval'
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
df = pd.read_csv('ColdHardiness_Grape_Merlot.csv')
modified_df, dormant_seasons = preprocess_missing_values(df)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons)
mean, std = get_mean_std(season_df, features)

############## Load Models ##############
knn_impute = KNNImputer(n_neighbors=7, weights='distance')
knn_impute.fit(season_df[features])

mice_impute = IterativeImputer(random_state=0)
mice_impute.fit(season_df[features])

model_brits = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

model_brits_I = BRITS_I(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

if os.path.exists('./model_BRITS.model'):
    model_brits.load_state_dict(torch.load('./model_BRITS.model'))

if os.path.exists('./model_BRITS_I.model'):
    model_brits_I.load_state_dict(torch.load('./model_BRITS_I.model'))

if torch.cuda.is_available():
    model_brits = model_brits.cuda()
    model_brits_I = model_brits_I.cuda()

model_brits.eval()
model_brits_I.eval()

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


def parse_id(x, y, feature_impute_idx, length):
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
    start_idx = np.random.choice(indices, 1)
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

given_feature = 'MEAN_AT'
L = [i for i in range(1, 51)]
iter = 100

results = {
    'BRITS': {},
    'BRITS_I': {},
    'KNN': {},
    'MICE': {}
}
start_time = time.time()
result_mse_plots = {
    'BRITS': [],
    'BRITS_I': [],
    'KNN': [],
    'MICE': []
}
for l in L:
    real_values = []
    imputed_brits = []
    imputed_brits_I = []
    imputed_knn = []
    imputed_mice = []

#     for i in range(iter):
#         fs = open(filename, 'w')
#         X, Y = split_XY(season_df, max_length, season_array)

#         missing_season, season_idx = get_minimum_missing_season(season_df, given_feature, season_array)

#         feature_idx = features.index(given_feature)
#         # for i in range(X.shape[0]):
#         missing_indices = parse_id(X[season_idx], Y[season_idx], feature_idx, l)
#         fs.close()


#         val_iter = data_loader.get_loader(batch_size=1, filename=filename)

#         for idx, data in enumerate(val_iter):
#             data = utils.to_var(data)
#             row_indices = missing_indices // len(features)

#             ret = model_brits.run_on_batch(data, None)
#             eval_ = ret['evals'].data.cpu().numpy()
#             eval_ = np.squeeze(eval_)
#             imputation_brits = ret['imputations'].data.cpu().numpy()
#             imputation_brits = np.squeeze(imputation_brits)

#             ret = model_brits_I.run_on_batch(data, None)
#             imputation_brits_I = ret['imputations'].data.cpu().numpy()
#             imputation_brits_I = np.squeeze(imputation_brits_I)

#             ret_eval = copy.deepcopy(eval_)
#             ret_eval[row_indices, feature_idx] = np.nan

#             imputation_knn = knn_impute.transform(ret_eval)
#             imputation_mice = mice_impute.transform(ret_eval)

#             imputed_brits.extend(imputation_brits[row_indices, feature_idx].tolist())
#             imputed_brits_I.extend(imputation_brits_I[row_indices, feature_idx].tolist())
#             imputed_knn.extend(imputation_knn[row_indices, feature_idx].tolist())
#             imputed_mice.extend(imputation_mice[row_indices, feature_idx].tolist())
#             real_values.extend(eval_[row_indices, feature_idx].tolist())
            
#             # print(f"\reval: {eval_[row_indices, feature_idx]}\nimputed(BRITS): {imputation_brits[row_indices, feature_idx]}\
#             #     \nimputed(BRITS_I): {imputation_brits_I[row_indices, feature_idx]}\nimputed(KNN): {imputed_knn[row_indices, feature_idx]}\
#             #     \nimputed(MICE): {imputed_mice[row_indices, feature_idx]}")
#     imputed_brits = np.array(imputed_brits)
#     imputed_brits_I = np.array(imputed_brits_I)
#     imputed_knn = np.array(imputed_knn)
#     imputed_mice = np.array(imputed_mice)
#     real_values = np.array(real_values)

#     brits_mse = np.round(((real_values - imputed_brits) ** 2).mean(), 5) 
#     brits_I_mse = np.round(((real_values - imputed_brits_I) ** 2).mean(), 5)
#     knn_mse = np.round(((real_values - imputed_knn) ** 2).mean(), 5)
#     mice_mse = np.round(((real_values - imputed_mice) ** 2).mean(), 5)

#     diff_brits = np.abs(real_values) - np.abs(imputed_brits)
#     diff_brits_I = np.abs(real_values) - np.abs(imputed_brits_I)
#     diff_knn = np.abs(real_values) - np.abs(imputed_knn)
#     diff_mice = np.abs(real_values) - np.abs(imputed_mice)


#     results['BRITS'][l] = brits_mse# f"MSE: {brits_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits)),5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits)), 5)}",
#     results['BRITS_I'][l] = brits_I_mse# f"MSE: {brits_I_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits_I)))}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits_I)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits_I)))}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits_I)), 5)}",
#     results['KNN'][l] = knn_mse# f"MSE: {knn_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_knn)), 5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_knn)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_knn)))}\\STD (diff GT): {np.round(np.std(np.abs(diff_knn)), 5)}",
#     results['MICE'][l] = mice_mse# f"MSE: {mice_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_mice)), 5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_mice)))}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_mice)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_mice)))}",
    
#     result_mse_plots['BRITS'].append(brits_mse)
#     result_mse_plots['BRITS_I'].append(brits_I_mse)
#     result_mse_plots['KNN'].append(knn_mse)
#     result_mse_plots['MICE'].append(mice_mse)
    
#     print(f"For L = {l}:\n\tMSE (BRITS): {brits_mse}\n\tDifference from GT:\
#         \n\t\tmin: {np.round(np.min(np.abs(diff_brits)), 5)}\n\t\tmax: {np.round(np.max(np.abs(diff_brits)), 5)}\n\t\tmean: {np.round(np.mean(np.abs(diff_brits)), 5)}\n\t\tstd: {np.round(np.std(np.abs(diff_brits)), 5)} \
#         \n\n\tMSE (BRITS_I): {brits_I_mse}\n\tDifference from GT:\
#         \n\t\tmin: {np.round(np.min(np.abs(diff_brits_I)), 5)}\n\t\tmax: {np.round(np.max(np.abs(diff_brits_I)), 5)}\n\t\tmean: {np.round(np.mean(np.abs(diff_brits_I)), 5)}\n\t\tstd: {np.round(np.std(np.abs(diff_brits_I)), 5)} \
#         \n\n\tMSE (KNN): {knn_mse}\n\tDifference from GT:\
#         \n\t\tmin: {np.round(np.min(np.abs(diff_knn)), 5)}\n\t\tmax: {np.round(np.max(np.abs(diff_knn)), 5)}\n\t\tmean: {np.round(np.mean(np.abs(diff_knn)), 5)}\n\t\tstd: {np.round(np.std(np.abs(diff_knn)), 5)} \
#         \n\n\tMSE (MICE): {mice_mse}\n\tDifference from GT:\
#         \n\t\tmin: {np.round(np.min(np.abs(diff_mice)), 5)}\n\t\tmax: {np.round(np.max(np.abs(diff_mice)), 5)}\n\t\tmean: {np.round(np.mean(np.abs(diff_mice)), 5)}\n\t\tstd: {np.round(np.std(np.abs(diff_mice)), 5)}\n\n")
# end_time = time.time()
# result_df = pd.DataFrame(results)
# result_df.to_csv('results_impute.csv')
# result_df.to_latex('results_impute.tex')

# print(f"Total time: {(end_time-start_time)/1000}s")

# plt.figure(figsize=(16,9))
# plt.plot(L, result_mse_plots['BRITS'], 'r', label='BRITS', marker='o')
# plt.plot(L, result_mse_plots['BRITS_I'], 'b', label='BRITS_I', marker='o')
# plt.plot(L, result_mse_plots['KNN'], 'g', label='KNN', marker='o')
# plt.plot(L, result_mse_plots['MICE'], 'c', label='MICE', marker='o')
# plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}')
# plt.xlabel(f'Length of contiguous missing values')
# plt.ylabel(f'MSE')
# plt.legend()
# plt.savefig(f'plots/L-vs-MSE-{features[feature_idx]}-{L}.png', dpi=300)
# plt.close()

# plt.figure(figsize=(16,9))
# plt.plot(L, result_mse_plots['BRITS'], 'r', label='BRITS', marker='o')
# plt.plot(L, result_mse_plots['MICE'], 'c', label='MICE', marker='o')
# plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}')
# plt.xlabel(f'Length of contiguous missing values')
# plt.ylabel(f'MSE')
# plt.legend()
# plt.savefig(f'plots/L-vs-MSE-BRITS-MICE-{features[feature_idx]}-{L}.png', dpi=300)
# plt.close()

fs = open(filename, 'w')
X, Y = split_XY(season_df, max_length, season_array)

missing_season, season_idx = get_minimum_missing_season(season_df, given_feature, season_array)

feature_idx = features.index(given_feature)
# for i in range(X.shape[0]):
missing_indices = parse_id(X[season_idx], Y[season_idx], feature_idx, 40)
fs.close()

val_iter = data_loader.get_loader(batch_size=1, filename=filename)
draws = {}
for idx, data in enumerate(val_iter):
    data = utils.to_var(data)
    row_indices = missing_indices // len(features)

    ret = model_brits.run_on_batch(data, None)
    eval_ = ret['evals'].data.cpu().numpy()
    eval_ = np.squeeze(eval_)
    imputation_brits = ret['imputations'].data.cpu().numpy()
    imputation_brits = np.squeeze(imputation_brits)
    draws['BRITS'] = imputation_brits[:, feature_idx]

    ret = model_brits_I.run_on_batch(data, None)
    imputation_brits_I = ret['imputations'].data.cpu().numpy()
    imputation_brits_I = np.squeeze(imputation_brits_I)
    draws['BRITS_I'] = imputation_brits_I[:, feature_idx]

    ret_eval = copy.deepcopy(eval_)
    ret_eval[row_indices, feature_idx] = np.nan

    imputation_knn = knn_impute.transform(ret_eval)
    draws['KNN'] = imputation_knn[:, feature_idx]
    imputation_mice = mice_impute.transform(ret_eval)
    draws['MICE'] = imputation_mice[:, feature_idx]

    real_values = eval_[:, feature_idx]
    missing_values = copy.deepcopy(real_values)
    missing_values[row_indices] = 0

    draws['real'] = real_values
    draws['missing'] = missing_values
    print(f"missing: {draws['missing'].shape}")
    print(f"BRITS: {draws['BRITS'].shape}")

    draw_data_plot(draws, features[feature_idx], season_idx)











# for feature_idx in features_impute:
#     start = 0
#     size = len(season_impute_brits)
#     limit = 252
#     iters = int(np.ceil(size / limit))
#     print('size: ', size, 'iters: ', iters)
#     for i in range(iters):
#         end = start + limit
#         print(f"start: {start}, end: {end}")
#         if size < end:
#             end = size
#         # draw_data_trend(season_eval[start:end,feature_idx], season_impute_brits[start:end,feature_idx], features[feature_idx], 'BRITS', i)
#         # draw_data_trend(season_eval[start:end,feature_idx], season_impute_brits_i[start:end,feature_idx], features[feature_idx], 'BRITS_I', i)
#         # draw_data_trend(season_eval[start:end,feature_idx], season_impute_knn[start:end,feature_idx], features[feature_idx], 'KNN', i)
#         # draw_data_trend(season_eval[start:end,feature_idx], season_impute_mice[start:end,feature_idx], features[feature_idx], 'MICE', i)
        
#         x_range = np.arange(len(season_eval[start:end,feature_idx]))
#         plt.figure(figsize=(16,9))
#         plt.plot(x_range, season_impute_brits[start:end,feature_idx], 'r', label='BRITS', scaley='log')
#         plt.plot(x_range, season_impute_brits_i[start:end,feature_idx], 'b', label='BRITS_I', scaley='log')
#         # plt.plot(x_range, season_impute_knn[start:end,feature_idx], 'c', label='KNN', scaley='log')
#         # plt.plot(x_range, season_impute_mice[start:end,feature_idx], 'm', label='MICE', scaley='log')
#         plt.plot(x_range, season_eval[start:end,feature_idx], 'g', label='Ground Truth', scaley='log')
#         plt.title(f'Ground Truth vs The imputations of {features[feature_idx]}')
#         plt.xlabel(f'Days of the season  no. {i}')
#         plt.ylabel(f'{features[feature_idx]}')
#         plt.legend()
#         plt.savefig(f'plots/gt-imputed-{features[feature_idx]}-{i}.png', dpi=300)
#         plt.close()
#         start = end

