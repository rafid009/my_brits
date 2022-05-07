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
from naomi.model import *
from naomi.helpers import run_test
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


seasons = {
'1988': 0,
'1989': 1,
'1990': 2,
'1991': 3,
'1992': 4,
'1993': 5,
'1994': 6,
'1995': 7,
'1996': 8,
'1997': 9,
'1998': 10,
'1999': 11,
'2000': 12,
'2001': 13,
'2002': 14,
'2003': 15,
'2004': 16,
'2005': 17,
'2006': 18,
'2007': 19,
'2008': 20,
'2009': 21,
'2010': 22,
'2011': 23,
'2012': 24,
'2013': 25,
'2014': 26,
'2015': 27,
'2016': 28,
'2017': 29,
'2018': 30,
'2019': 31,
'2020': 32,
'2021': 33,
'2022': 34
}

RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.5
LABEL_WEIGHT = 1


folder = './json/'
file = 'json_eval_2'
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
df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
modified_df, dormant_seasons = preprocess_missing_values(df, is_dormant=False, is_year=True)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, is_dormant=False, is_year=True)
train_season_df = season_df.drop(season_array[-1], axis=0)
train_season_df = train_season_df.drop(season_array[-2], axis=0)

mean, std = get_mean_std(train_season_df, features)


############## Load Models ##############
# print(f'seasons: {len(season_df)}, train season: {len(train_season_df)}')
normalized_season_df = train_season_df[features].copy()
# print(f"norm: {normalized_season_df.shape}, mean: {mean.shape}, std: {std.shape}")
normalized_season_df = (normalized_season_df - mean) /std

# knn_impute = KNNImputer(n_neighbors=7, weights='distance')
# knn_impute.fit(normalized_season_df[features].to_numpy())

mice_impute = IterativeImputer(random_state=0, max_iter=20)
mice_impute.fit(normalized_season_df[features].to_numpy())

model_brits = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

# model_brits_I = BRITS_I(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

# model_naomi = NAOMI(params)
# if os.path.exists('./policy_step8_training.pth'):
#     model_naomi.load_state_dict(torch.load('./policy_step8_training.pth', map_location=torch.device('cpu')))

if os.path.exists('./model_BRITS.model'):
    model_brits.load_state_dict(torch.load('./model_BRITS.model'))

# if os.path.exists('./model_BRITS_I.model'):
#     model_brits_I.load_state_dict(torch.load('./model_BRITS_I.model'))

if torch.cuda.is_available():
    model_brits = model_brits.cuda()
    # model_brits_I = model_brits_I.cuda()
    # model_naomi = model_naomi.cuda()

model_brits.eval()
# model_brits_I.eval()
# model_naomi.eval()

############## Draw Functions ##############
def graph_bar_diff_multi(GT_values, result_dict, title, x, xlabel, ylabel, season, feature, drop_linear=False, missing=None):
    plot_dict = {}
    plt.figure(figsize=(16,9))
    for key, value in result_dict.items():
      plot_dict[key] = np.abs(GT_values) - np.abs(value[missing])
    # ind = np.arange(prediction.shape[0])
    # x = np.array(x)
    width = 0.3
    pos = 0
    remove_keys = ['real', 'missing']
    # if drop_linear:
    #   remove_keys.append('LinearInterp')
    for key, value in plot_dict.items():
        if key not in remove_keys:
            # print(f"x = {len(x)}, value = {len(value)}")
            plt.bar(x + pos + 5, value, width, label = key)
            pos += width

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20)
    # plt.axis([0, 80, -2, 3])

    plt.legend(loc='best')
    # plt.tight_layout(pad=5)
    if not os.path.isdir('diff_imgs'):
        os.makedirs('diff_imgs')
    plt.savefig(f'diff_imgs/result-diff-{feature}-{season}.png', dpi=300)
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

def draw_data_plot(results, f, season_idx, folder='subplots', is_original=False):
    if not os.path.isdir(f"{folder}/{f}"):
        os.makedirs(f"{folder}/{f}")
    plt.figure(figsize=(32,18))
    plt.title(f"For feature = {f} in Season {season_idx}", fontsize=24)
    if is_original:
        ax = plt.subplot(311)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' original data', fontsize=20)
        plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Values', fontsize=16)

        ax = plt.subplot(312)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by MICE', fontsize=20)
        plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Values', fontsize=16)

        ax = plt.subplot(313)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by BRITS', fontsize=20)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Values', fontsize=16)
    else:
        ax = plt.subplot(411)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' original data', fontsize=20)
        plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Values', fontsize=16)

        ax = plt.subplot(412)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' missing data', fontsize=20)
        plt.plot(np.arange(results['missing'].shape[0]), results['missing'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Values', fontsize=16)

        ax = plt.subplot(413)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by MICE', fontsize=20)
        plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Values', fontsize=16)

        ax = plt.subplot(414)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by BRITS', fontsize=20)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=16)
        ax.set_ylabel('Values', fontsize=16)

    plt.tight_layout(pad=5)
    plt.savefig(f"{folder}/{f}/{f}-imputations-season-{season_idx}.png", dpi=300)
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

def parse_id(fs, x, y, feature_impute_idx, length, trial_num=-1, dependent_features=None, real_test=True, random_start=False):
    # evals = copy.deepcopy(x)
    # if length > 1:
    #      print(dependent_feature_ids)
    if real_test:
        idx1 = np.where(~np.isnan(x[:,feature_impute_idx]))[0]
        idx1 = idx1 * len(features) + feature_impute_idx
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
    # for key in rec.keys():
    #     print(f"{key}: {type(rec[key])}")# and {rec[key].shape}")
    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return indices

# given_feature = 'AVG_REL_HUMIDITY'
L = [i for i in range(1, 50)]
iter = 30


start_time = time.time()


given_features = features

given_features = [
    # 'MEAN_AT', # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
    # 'MIN_AT',
    # 'AVG_AT', # average temp is AgWeather Network
    # 'MAX_AT',
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
    'ETR' # ???
]

test_df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
test_modified_df, test_dormant_seasons = preprocess_missing_values(test_df, is_dormant=False, is_year=True)
# print(f"dormant seasons: {len(test_dormant_seasons)}\n {test_dormant_seasons}")
season_df, season_array, max_length = get_seasons_data(test_modified_df, test_dormant_seasons, is_dormant=False, is_year=True)

# print(f"season array: {season_array[1]}")
plot_mse_folder = 'overlapping_mse/'

def do_evaluation(mse_folder, eval_type, eval_season='2021'):
    filename = 'json/json_eval_2'
    for given_feature in given_features:
        result_mse_plots = {
        'BRITS': [],
        'MICE': [],
        }
        results = {
            'BRITS': {},
            'MICE': {},
        }
        for l in L:
            season_idx = seasons[eval_season]
            feature_idx = features.index(given_feature)
            X, Y = split_XY(season_df, max_length, season_array)
            # print(f'X: {X.shape}, Y: {Y.shape}')
            original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
            if eval_type != 'random':
                iter = len(season_array[-2]) - (l-1) - len(original_missing_indices)
            print(f"For feature = {given_feature} and length = {l}")
            
            total_count = 0
            brits_mse = 0
            mice_mse = 0

            for i in tqdm(range(iter)):
                # i.set_description(f"For {given_feature} & L = {l}")
                real_values = []
                imputed_brits = []
                imputed_mice = []

                fs = open(filename, 'w')

                if given_feature.split('_')[-1] not in feature_dependency.keys():
                    dependent_feature_ids = []
                else:
                    dependent_feature_ids = [features.index(f) for f in feature_dependency[given_feature.split('_')[-1]] if f != given_feature]
                if eval_type == 'random':
                    missing_indices = parse_id(fs, X[season_idx], Y[season_idx], feature_idx, l, i, dependent_feature_ids, random_start=True)
                else:
                    missing_indices = parse_id(fs, X[season_idx], Y[season_idx], feature_idx, l, i, dependent_feature_ids)
                fs.close()

                val_iter = data_loader.get_loader(batch_size=1, filename=filename)

                for idx, data in enumerate(val_iter):
                    data = utils.to_var(data)
                    row_indices = missing_indices // len(features)

                    ret = model_brits.run_on_batch(data, None)
                    eval_ = ret['evals'].data.cpu().numpy()
                    eval_ = np.squeeze(eval_)
                    imputation_brits = ret['imputations'].data.cpu().numpy()
                    imputation_brits = np.squeeze(imputation_brits)
                    # imputation_brits = unnormalize(imputation_brits, mean, std, -1)

                    ret_eval = copy.deepcopy(eval_)
                    ret_eval[row_indices, feature_idx] = np.nan

                    imputation_mice = mice_impute.transform(ret_eval)

                    imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)
                    
                    imputed_mice = imputation_mice[row_indices, feature_idx]#unnormalize(imputation_mice[row_indices, feature_idx], mean, std, feature_idx)
                    
                    real_values = eval_[row_indices, feature_idx]#unnormalize(eval_[row_indices, feature_idx], mean, std, feature_idx)

                brits_mse += ((real_values - imputed_brits) ** 2).mean()

                mice_mse += ((real_values - imputed_mice) ** 2).mean()
                total_count += 1

            print(f"AVG MSE for {iter} runs (sliding window of Length = {l}):\n\tBRITS: {brits_mse/total_count}\n\tMICE: {mice_mse/total_count}")

            results['BRITS'][l] = brits_mse/total_count# f"MSE: {brits_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits)),5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits)), 5)}",
            results['MICE'][l] = mice_mse/total_count# f"MSE: {mice_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_mice)), 5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_mice)))}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_mice)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_mice)))}",

            result_mse_plots['BRITS'].append(brits_mse/total_count)
            result_mse_plots['MICE'].append(mice_mse/total_count)
            
        end_time = time.time()
        result_df = pd.DataFrame(results)
        if not os.path.isdir(f'{mse_folder}/{eval_type}/imputation_results/'+given_feature):
            os.makedirs(f'{mse_folder}/{eval_type}/imputation_results/'+given_feature)
        result_df.to_csv(f'{mse_folder}/{eval_type}/imputation_results/{given_feature}/{given_feature}_results_impute.csv')
        result_df.to_latex(f'{mse_folder}/{eval_type}/imputation_results/{given_feature}/{given_feature}_results_impute.tex')

        if not os.path.isdir(f'{mse_folder}/{eval_type}/plots/{given_feature}'):
            os.makedirs(f'{mse_folder}/{eval_type}/plots/{given_feature}')

        plt.figure(figsize=(16,9))
        plt.plot(L, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')

        plt.plot(L, result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
        plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
        plt.xlabel(f'Length of contiguous missing values', fontsize=16)
        plt.ylabel(f'MSE', fontsize=16)
        plt.legend()
        plt.savefig(f'{mse_folder}/{eval_type}/plots/{given_feature}/L-vs-MSE-brits-mice-models-{features[feature_idx]}-{len(L)}.png', dpi=300)
        plt.close()

        plt.figure(figsize=(16,9))
        plt.plot(L, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
        plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
        plt.xlabel(f'Length of contiguous missing values', fontsize=16)
        plt.ylabel(f'MSE', fontsize=16)
        plt.legend()
        plt.savefig(f'{mse_folder}/{eval_type}/plots/{given_feature}/L-vs-MSE-BRITS-{features[feature_idx]}-{len(L)}.png', dpi=300)
        plt.close()

        plt.figure(figsize=(16,9))
        plt.plot(L, result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
        plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
        plt.xlabel(f'Length of contiguous missing values', fontsize=16)
        plt.ylabel(f'MSE', fontsize=16)
        plt.legend()
        plt.savefig(f'{mse_folder}/{eval_type}/plots/{given_feature}/L-vs-MSE-MICE-{features[feature_idx]}-{len(L)}.png', dpi=300)
        plt.close()



# L = [i for i in range(4, 22, 2)]
# L.append(36)
# for given_feature in given_features:
#     result_mse_plots = {
#     'BRITS': [],
#     # 'BRITS_I': [],
#     # 'KNN': [],
#     'MICE': [],
#     # 'NAOMI': []
#     }
#     results = {
#         'BRITS': {},
#         'MICE': {},
#         # 'NAOMI': {}
#     }
#     index = 0
#     for l in L:
#         # imputed_naomi = []
#         # with tqdm(range(iter), unit='iter') as titer:
#         season_idx = seasons['2020-2021']
#         feature_idx = features.index(given_feature)
#         X, Y = split_XY(season_df, max_length, season_array)
#         original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
#         iter = 4 #len(season_array[-2]) - (l-1) - len(original_missing_indices)
#         if l == 36:
#             iter = 1
#         else:
#             iter = 2
#         print(f"For feature = {given_feature} and length = {l}")
#         brits_mse = 0
#         mice_mse = 0
#         total_count = 0
#         for i in tqdm(range(iter)):
#             real_values = []
#             imputed_brits = []
#             imputed_mice = []
#             # i.set_description(f"For {given_feature} & L = {l}")
#             fs = open(filename, 'w')
            
#             # missing_season, season_idx = get_minimum_missing_season(season_df, given_feature, season_array)
            
            
#             # for i in range(X.shape[0]):
#             if given_feature.split('_')[-1] not in feature_dependency.keys():
#                 dependent_feature_ids = []
#             else:
#                 dependent_feature_ids = [features.index(f) for f in feature_dependency[given_feature.split('_')[-1]] if f != given_feature]
#             missing_indices = parse_id(X[season_idx], Y[season_idx], feature_idx, l, index, dependent_feature_ids)
#             fs.close()


#             val_iter = data_loader.get_loader(batch_size=1, filename=filename)

#             for idx, data in enumerate(val_iter):
#                 data = utils.to_var(data)
#                 row_indices = missing_indices // len(features)

#                 ret = model_brits.run_on_batch(data, None)
#                 eval_ = ret['evals'].data.cpu().numpy()
#                 eval_ = np.squeeze(eval_)
#                 imputation_brits = ret['imputations'].data.cpu().numpy()
#                 imputation_brits = np.squeeze(imputation_brits)

#                 # ret = model_brits_I.run_on_batch(data, None)
#                 # imputation_brits_I = ret['imputations'].data.cpu().numpy()
#                 # imputation_brits_I = np.squeeze(imputation_brits_I)

#                 ret_eval = copy.deepcopy(eval_)
#                 ret_eval[row_indices, feature_idx] = np.nan

#                 # imputation_knn = knn_impute.transform(ret_eval)
#                 imputation_mice = mice_impute.transform(ret_eval)

#                 # data = torch.tensor(np.expand_dims(ret_eval, axis=0))
#                 # _, imputed_values = run_test(model_naomi, data, 1, row_indices, feature_idx)
#                 # imputation_naomi = imputed_values[0].transpose(1,0).squeeze().detach().numpy()

#                 imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)
#                 # imputed_brits_I.extend(copy.deepcopy(imputation_brits_I[row_indices, feature_idx].tolist()))
#                 # imputed_knn.extend(copy.deepcopy(imputation_knn[row_indices, feature_idx].tolist()))
#                 imputed_mice = imputation_mice[row_indices, feature_idx]#unnormalize(imputation_mice[row_indices, feature_idx], mean, std, feature_idx)
#                 # imputed_naomi.extend(copy.deepcopy(imputation_naomi[row_indices, feature_idx].tolist()))
#                 real_values = eval_[row_indices, feature_idx]#unnormalize(eval_[row_indices, feature_idx], mean, std, feature_idx)
                
#                 # print(f"\reval: {eval_[row_indices, feature_idx]}\nimputed(BRITS): {imputation_brits[row_indices, feature_idx]}\
#                 #     \nimputed(BRITS_I): {imputation_brits_I[row_indices, feature_idx]}\nimputed(KNN): {imputed_knn[row_indices, feature_idx]}\
#                 #     \nimputed(MICE): {imputed_mice[row_indices, feature_idx]}")
#             index += l


#             brits_mse += ((real_values - imputed_brits) ** 2).mean()
        
#             mice_mse += ((real_values - imputed_mice) ** 2).mean()
#             total_count += 1

#         print(f"AVG MSE for {iter} runs (sliding window of Length = {l}):\n\tBRITS: {brits_mse/total_count}\n\tMICE: {mice_mse/total_count}")

#         # diff_brits = np.abs(real_values) - np.abs(imputed_brits)
#         # diff_brits_I = np.abs(real_values) - np.abs(imputed_brits_I)
#         # diff_knn = np.abs(real_values) - np.abs(imputed_knn)
#         # diff_mice = np.abs(real_values) - np.abs(imputed_mice)
#         # diff_naomi = np.abs(real_values) - np.abs(imputed_naomi)


#         results['BRITS'][l] = brits_mse/total_count# f"MSE: {brits_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits)),5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits)), 5)}",
#         results['MICE'][l] = mice_mse/total_count# f"MSE: {mice_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_mice)), 5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_mice)))}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_mice)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_mice)))}",
#         # results['NAOMI'][l] = naomi_mse

#         result_mse_plots['BRITS'].append(brits_mse/total_count)
#         # result_mse_plots['BRITS_I'].append(brits_I_mse)
#         # result_mse_plots['KNN'].append(knn_mse)
#         result_mse_plots['MICE'].append(mice_mse/total_count)
#         # result_mse_plots['NAOMI'].append(naomi_mse)
        
#     end_time = time.time()
#     result_df = pd.DataFrame(results)
#     if not os.path.isdir('non_imputation_results/'+given_feature):
#         os.makedirs('non_imputation_results/'+given_feature)
#     result_df.to_csv(f'non_imputation_results/{given_feature}/{given_feature}_results_impute.csv')
#     result_df.to_latex(f'non_imputation_results/{given_feature}/{given_feature}_results_impute.tex')

#     if not os.path.isdir('non_overlap_plots/'+given_feature):
#         os.makedirs('non_overlap_plots/'+given_feature)
#     plt.figure(figsize=(16,9))
#     plt.plot(L, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
#     plt.plot(L, result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
#     plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
#     plt.xlabel(f'Length of contiguous missing values', fontsize=16)
#     plt.ylabel(f'MSE', fontsize=16)
#     plt.legend()
#     plt.savefig(f'non_overlap_plots/{given_feature}/L-vs-MSE-BRITS-MICE-{features[feature_idx]}-{L[-1]}.png', dpi=300)
#     plt.close()

#     plt.figure(figsize=(16,9))
#     plt.plot(L, result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
#     plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
#     plt.xlabel(f'Length of contiguous missing values', fontsize=16)
#     plt.ylabel(f'MSE', fontsize=16)
#     plt.legend()
#     plt.savefig(f'non_overlap_plots/{given_feature}/L-vs-MSE-MICE-{features[feature_idx]}-{L[-1]}.png', dpi=300)
#     plt.close()

#     plt.figure(figsize=(16,9))
#     plt.plot(L, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
#     plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
#     plt.xlabel(f'Length of contiguous missing values', fontsize=16)
#     plt.ylabel(f'MSE', fontsize=16)
#     plt.legend()
#     plt.savefig(f'non_overlap_plots/{given_feature}/L-vs-MSE-BRITS-{features[feature_idx]}-{L[-1]}.png', dpi=300)
#     plt.close()

# L = [i for i in range(1, 81, 2)]
# # L.append(36)
# for given_feature in given_features:
#     result_mse_plots = {
#     'BRITS': [],
#     # 'BRITS_I': [],
#     # 'KNN': [],
#     'MICE': [],
#     # 'NAOMI': []
#     }
#     results = {
#         'BRITS': {},
#         'MICE': {},
#         # 'NAOMI': {}
#     }
    
#     for l in L:
#         # imputed_naomi = []
#         # with tqdm(range(iter), unit='iter') as titer:
#         season_idx = -2
#         feature_idx = features.index(given_feature)
#         X, Y = split_XY(season_df, max_length, season_array)
#         original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
#         # iter = 4 #len(season_array[-2]) - (l-1) - len(original_missing_indices)
#         # if l == 36:
#         #     iter = 1
#         # else:
#         #     iter = 2
#         iter = (len(season_array[-2]) - len(original_missing_indices))//l
#         index = 0
#         print(f"For feature = {given_feature} and length = {l}")
#         brits_mse = 0
#         mice_mse = 0
#         total_count = 0       
#         for i in tqdm(range(iter)):
#             real_values = []
#             imputed_brits = []
#             imputed_mice = []
#             # i.set_description(f"For {given_feature} & L = {l}")
#             fs = open(filename, 'w')
            
#             # missing_season, season_idx = get_minimum_missing_season(season_df, given_feature, season_array)
            
            
#             # for i in range(X.shape[0]):
#             if given_feature.split('_')[-1] not in feature_dependency.keys():
#                 dependent_feature_ids = []
#             else:
#                 dependent_feature_ids = [features.index(f) for f in feature_dependency[given_feature.split('_')[-1]] if f != given_feature]
#             missing_indices = parse_id(X[season_idx], Y[season_idx], feature_idx, l, index, dependent_feature_ids)
#             fs.close()


#             val_iter = data_loader.get_loader(batch_size=1, filename=filename)

#             for idx, data in enumerate(val_iter):
#                 data = utils.to_var(data)
#                 row_indices = missing_indices // len(features)

#                 ret = model_brits.run_on_batch(data, None)
#                 eval_ = ret['evals'].data.cpu().numpy()
#                 eval_ = np.squeeze(eval_)
#                 imputation_brits = ret['imputations'].data.cpu().numpy()
#                 imputation_brits = np.squeeze(imputation_brits)

#                 # ret = model_brits_I.run_on_batch(data, None)
#                 # imputation_brits_I = ret['imputations'].data.cpu().numpy()
#                 # imputation_brits_I = np.squeeze(imputation_brits_I)

#                 ret_eval = copy.deepcopy(eval_)
#                 ret_eval[row_indices, feature_idx] = np.nan

#                 # imputation_knn = knn_impute.transform(ret_eval)
#                 imputation_mice = mice_impute.transform(ret_eval)

#                 # data = torch.tensor(np.expand_dims(ret_eval, axis=0))
#                 # _, imputed_values = run_test(model_naomi, data, 1, row_indices, feature_idx)
#                 # imputation_naomi = imputed_values[0].transpose(1,0).squeeze().detach().numpy()

#                 imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)
#                 # imputed_brits_I.extend(copy.deepcopy(imputation_brits_I[row_indices, feature_idx].tolist()))
#                 # imputed_knn.extend(copy.deepcopy(imputation_knn[row_indices, feature_idx].tolist()))
#                 imputed_mice = imputation_mice[row_indices, feature_idx]#unnormalize(imputation_mice[row_indices, feature_idx], mean, std, feature_idx)
#                 # imputed_naomi.extend(copy.deepcopy(imputation_naomi[row_indices, feature_idx].tolist()))
#                 real_values = eval_[row_indices, feature_idx]#unnormalize(eval_[row_indices, feature_idx], mean, std, feature_idx)
                
#                 # print(f"\reval: {eval_[row_indices, feature_idx]}\nimputed(BRITS): {imputation_brits[row_indices, feature_idx]}\
#                 #     \nimputed(BRITS_I): {imputation_brits_I[row_indices, feature_idx]}\nimputed(KNN): {imputed_knn[row_indices, feature_idx]}\
#                 #     \nimputed(MICE): {imputed_mice[row_indices, feature_idx]}")
#             index += l
        

#             brits_mse += ((real_values - imputed_brits) ** 2).mean()
#             mice_mse += ((real_values - imputed_mice) ** 2).mean()
#             total_count += 1

#         print(f"AVG MSE for {iter} runs (sliding window of Length = {l}):\n\tBRITS: {brits_mse/total_count}\n\tMICE: {mice_mse/total_count}")

#         # diff_brits = np.abs(real_values) - np.abs(imputed_brits)
#         # diff_brits_I = np.abs(real_values) - np.abs(imputed_brits_I)
#         # diff_knn = np.abs(real_values) - np.abs(imputed_knn)
#         # diff_mice = np.abs(real_values) - np.abs(imputed_mice)
#         # diff_naomi = np.abs(real_values) - np.abs(imputed_naomi)


#         results['BRITS'][l] = brits_mse/total_count# f"MSE: {brits_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits)),5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits)), 5)}",
#         results['MICE'][l] = mice_mse/total_count# f"MSE: {mice_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_mice)), 5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_mice)))}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_mice)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_mice)))}",
#         # results['NAOMI'][l] = naomi_mse

#         result_mse_plots['BRITS'].append(brits_mse/total_count)
#         result_mse_plots['MICE'].append(mice_mse/total_count)
        
#     end_time = time.time()
#     result_df = pd.DataFrame(results)
#     if not os.path.isdir('non_l_imputation_results/'+given_feature):
#         os.makedirs('non_l_imputation_results/'+given_feature)
#     result_df.to_csv(f'non_l_imputation_results/{given_feature}/{given_feature}_results_impute.csv')
#     result_df.to_latex(f'non_l_imputation_results/{given_feature}/{given_feature}_results_impute.tex')

#     if not os.path.isdir('non_overlap_l_plots/'+given_feature):
#         os.makedirs('non_overlap_l_plots/'+given_feature)
#     plt.figure(figsize=(16,9))
#     plt.plot(L, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
#     plt.plot(L, result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
#     plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
#     plt.xlabel(f'Length of contiguous missing values', fontsize=16)
#     plt.ylabel(f'MSE', fontsize=16)
#     plt.legend()
#     plt.savefig(f'non_overlap_l_plots/{given_feature}/L-vs-MSE-BRITS-MICE-{features[feature_idx]}-{L[-1]}.png', dpi=300)
#     plt.close()

#     plt.figure(figsize=(16,9))
#     plt.plot(L, result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
#     plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
#     plt.xlabel(f'Length of contiguous missing values', fontsize=16)
#     plt.ylabel(f'MSE', fontsize=16)
#     plt.legend()
#     plt.savefig(f'non_overlap_l_plots/{given_feature}/L-vs-MSE-MICE-{features[feature_idx]}-{L[-1]}.png', dpi=300)
#     plt.close()

#     plt.figure(figsize=(16,9))
#     plt.plot(L, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
#     plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, season=2020-2021', fontsize=20)
#     plt.xlabel(f'Length of contiguous missing values', fontsize=16)
#     plt.ylabel(f'MSE', fontsize=16)
#     plt.legend()
#     plt.savefig(f'non_overlap_l_plots/{given_feature}/L-vs-MSE-BRITS-{features[feature_idx]}-{L[-1]}.png', dpi=300)
#     plt.close()

####################### Draw data plots #######################

def do_data_plots(data_folder, missing_length, is_original=False):
    print(f'Season array: {len(season_array)}')
    filename = 'json/json_eval_3'
    missing_num = missing_length
    if is_original:
        data_folder += '/original_missing'
    else:
        data_folder += '/eval_missing'
    for given_season in seasons:
        season_idx = seasons[given_season]
        given_features = features
        for given_feature in tqdm(given_features):
            fs = open(filename, 'w')
            X, Y = split_XY(season_df, max_length, season_array)

            feature_idx = features.index(given_feature)

            original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
            print(f"feature: {given_feature}, season: {given_season}, season idx: {seasons[given_season]}, Original missing: {len(original_missing_indices)}")
            
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

                imputation_mice = mice_impute.transform(ret_eval)
                draws['MICE'] = unnormalize(imputation_mice[:, feature_idx], mean, std, feature_idx)

                draw_data_plot(draws, features[feature_idx], given_season, folder=data_folder, is_original=is_original)

                # graph_bar_diff_multi(draws['real'][row_indices], draws, f'Difference From Gorund Truth for {given_feature} in 2020-2021', np.arange(missing_num), 'Days', given_feature, '2020-2021', given_feature, missing=row_indices)


eval_folder = 'eval_dir'
do_evaluation(eval_folder, 'cont', '2021')
# data_plots_folder = 'data_plots'
# do_data_plots(data_plots_folder, 50, is_original=True)
# do_data_plots(data_plots_folder, 50, is_original=False)






