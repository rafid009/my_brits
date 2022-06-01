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
from transformer.src.transformer import run_transformer
import warnings
import matplotlib


warnings.filterwarnings("ignore")
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# '2022': 34
}

RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.5
LABEL_WEIGHT = 1

params = {
    'config_filepath': None, 
    'output_dir': './transformer/output/out', 
    'data_dir': './transformer/data_dir/', 
    'load_model': './transformer/output/SeasonData_LT_2022-05-24_11-06-36_lPH_LT/checkpoints/model_best.pth', 
    'resume': False, 
    'change_output': False, 
    'save_all': False, 
    'experiment_name': 'SeasonData_pretrained', 
    'comment': 'pretraining through imputation', 
    'no_timestamp': False, 
    'records_file': './transformer/Imputation_records.csv', 
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
    'pattern': 'Merlot_test', 
    'val_pattern': None, 
    'test_pattern': None, 
    'normalization': 'standardization', 
    'norm_from': './transformer/output/SeasonData_LT_2022-05-24_11-06-36_lPH_LT/normalization.pickle', 
    'subsample_factor': None, 
    'task': 'imputation', 
    'masking_ratio': 0.2, 
    'mean_mask_length': 10.0, 
    'mask_mode': 'separate', 
    'mask_distribution': 'geometric', 
    'exclude_feats': None, 
    'mask_feats': [0, 1], 
    'start_hint': 0.0, 
    'end_hint': 0.0, 
    'harden': True, 
    'epochs': 500, 
    'val_interval': 2, 
    'optimizer': 'Adam', 
    'lr': 0.0009, 
    'lr_step': [1000000], 
    'lr_factor': [0.1], 
    'batch_size': 1, 
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

def add_season_id(data_folder, season_df):
    season_df['season_id'] = 0
    for season_id in range(len(season_array)):
        for idx in season_array[season_id]:
            train_season_df.loc[idx, 'season_id'] = season_id
    season_df.to_csv(f'{data_folder}/ColdHardiness_Grape_Merlot_test.csv', index=False)

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
df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
modified_df, dormant_seasons = preprocess_missing_values(df, is_dormant=True)#False, is_year=True)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, is_dormant=True)#False, is_year=True)
train_season_df = season_df.drop(season_array[-1], axis=0)
train_season_df = train_season_df.drop(season_array[-2], axis=0)

mean, std = get_mean_std(train_season_df, features)


############## Load Models ##############

normalized_season_df = train_season_df[features].copy()
normalized_season_df = (normalized_season_df - mean) /std
mice_impute = IterativeImputer(random_state=0, max_iter=20)
mice_impute.fit(normalized_season_df[features].to_numpy())

model_brits = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

if os.path.exists('./model_BRITS_LT.model'):
    model_brits.load_state_dict(torch.load('./model_BRITS_LT.model'))

if torch.cuda.is_available():
    model_brits = model_brits.cuda()

model_brits.eval()

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
    plt.title(f"For feature = {f} in Season {season_idx}", fontsize=30)
    if is_original:
        ax = plt.subplot(211)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' original data', fontsize=27)
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
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by BRITS', fontsize=27)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)
    else:
        ax = plt.subplot(311)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' original data', fontsize=27)
        plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)

        ax = plt.subplot(312)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' missing data', fontsize=27)
        plt.plot(np.arange(results['missing'].shape[0]), results['missing'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)

        ax = plt.subplot(313)
        # ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by MICE', fontsize=20)
        # plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'], 'tab:blue')
        # ax.set_xlabel('Days', fontsize=16)
        # ax.set_ylabel('Values', fontsize=16)

        # ax = plt.subplot(414)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by BRITS', fontsize=27)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        # ax.xticks(fontsize=20)
        # ax.yticks(fontsize=20)

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

    if real_test:
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
    return indices

# given_feature = 'AVG_REL_HUMIDITY'
L = [i for i in range(1, 30)]
# L = [1, 5, 10, 20]#, 70, 100, 150, 200]
iter = 30


start_time = time.time()


# given_features = features

given_features = [
    # 'MEAN_AT', # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
    # 'MIN_AT',
    # 'AVG_AT', # average temp is AgWeather Network
    # 'MAX_AT',
    # 'MIN_REL_HUMIDITY',
    # 'AVG_REL_HUMIDITY',
    # 'MAX_REL_HUMIDITY',
    # 'MIN_DEWPT',
    # 'AVG_DEWPT',
    # 'MAX_DEWPT',
    # 'P_INCHES', # precipitation
    # 'WS_MPH', # wind speed. if no sensor then value will be na
    # 'MAX_WS_MPH', 
    # 'LW_UNITY', # leaf wetness sensor
    # 'SR_WM2', # solar radiation # different from zengxian
    # 'MIN_ST8', # diff from zengxian
    # 'ST8', # soil temperature # diff from zengxian
    # 'MAX_ST8', # diff from zengxian
    # #'MSLP_HPA', # barrometric pressure # diff from zengxian
    # 'ETO', # evaporation of soil water lost to atmosphere
    # 'ETR',
    'LTE50' # ???
]

test_df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
test_modified_df, test_dormant_seasons = preprocess_missing_values(test_df, is_dormant=True)#, is_year=True)
# print(f"dormant seasons: {len(test_dormant_seasons)}\n {test_dormant_seasons}")
season_df, season_array, max_length = get_seasons_data(test_modified_df, test_dormant_seasons, is_dormant=True)#, is_year=True)

# print(f"season array: {season_array[1]}")
plot_mse_folder = 'overlapping_mse/'

def do_evaluation(mse_folder, eval_type, eval_season='2021'):
    filename = 'json/json_eval_2_LT'
    for given_feature in given_features:
        result_mse_plots = {
        'BRITS': [],
        'MICE': [],
        'Transformer': []
        }
        results = {
            'BRITS': {},
            'MICE': {},
            'Transformer': {}
        }
        for l in L:
            # season_idx = seasons[eval_season]
            season_idx = -2
            feature_idx = features.index(given_feature)
            X, Y = split_XY(season_df, max_length, season_array)
            original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]
            if eval_type != 'random':
                iter = len(season_array[season_idx]) - (l-1) - len(original_missing_indices)
            print(f"For feature = {given_feature} and length = {l}")
            # print(f"original miss: {original_missing_indices.shape[0]}\n{original_missing_indices}\nseason: {len(season_array[season_idx])}")
            total_count = 0
            brits_mse = 0
            mice_mse = 0
            transformer_mse = 0
            neg = 0
            for i in tqdm(range(iter)):
                # i.set_description(f"For {given_feature} & L = {l}")
                real_values = []
                imputed_brits = []
                imputed_mice = []
                imputed_transformer = []

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
                if len(missing_indices) == 0:
                    print(f"0 missing added")
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
                    # imputation_brits = unnormalize(imputation_brits, mean, std, -1)

                    ret_eval = copy.deepcopy(eval_)
                    ret_eval[row_indices, feature_idx] = np.nan
                    imputation_mice = mice_impute.transform(ret_eval)

                    ret_eval = copy.deepcopy(eval_)
                    ret_eval = unnormalize(ret_eval, mean, std, feature_idx)
                    ret_eval[row_indices, feature_idx] = np.nan
                    trans_test_df = pd.DataFrame(ret_eval, columns=features)
                    add_season_id('./transformer/data_dir', trans_test_df)

                    transformer_preds = run_transformer(params)
                    # print(f'trasformer preds: {transformer_preds.shape}')
                    
                    imputation_transformer = np.squeeze(transformer_preds)
                    # print(f"trans: {imputation_transformer.shape}")
                    imputed_transformer = imputation_transformer[row_indices, feature_idx].data.cpu().numpy()
                    # print(f'trans preds: {imputed_transformer}')
                    

                    imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)
                    
                    imputed_mice = imputation_mice[row_indices, feature_idx]#unnormalize(imputation_mice[row_indices, feature_idx], mean, std, feature_idx)
                    
                    real_values = eval_[row_indices, feature_idx]#unnormalize(eval_[row_indices, feature_idx], mean, std, feature_idx)

                brits_mse += ((real_values - imputed_brits) ** 2).mean()

                mice_mse += ((real_values - imputed_mice) ** 2).mean()

                transformer_mse += ((real_values - imputed_transformer) ** 2).mean()
                total_count += 1
            if total_count <= 0:
                neg+=1
                continue
            print(f"AVG MSE for {iter} runs (sliding window of Length = {l}):\n\tBRITS: {brits_mse/total_count}\n\tMICE: {mice_mse/total_count}\n\tTransformer: {transformer_mse/total_count}")

            results['BRITS'][l] = brits_mse/total_count# f"MSE: {brits_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits)),5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits)), 5)}",
            results['MICE'][l] = mice_mse/total_count# f"MSE: {mice_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_mice)), 5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_mice)))}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_mice)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_mice)))}",
            results['Transformer'][l] = transformer_mse/total_count

            result_mse_plots['BRITS'].append(brits_mse/total_count)
            result_mse_plots['MICE'].append(mice_mse/total_count)
            result_mse_plots['Transformer'].append(transformer_mse/total_count)
            
        end_time = time.time()
        result_df = pd.DataFrame(results)
        if not os.path.isdir(f'{mse_folder}/{eval_type}/imputation_results/'+given_feature):
            os.makedirs(f'{mse_folder}/{eval_type}/imputation_results/'+given_feature)
        result_df.to_csv(f'{mse_folder}/{eval_type}/imputation_results/{given_feature}/{given_feature}_results_impute.csv')
        result_df.to_latex(f'{mse_folder}/{eval_type}/imputation_results/{given_feature}/{given_feature}_results_impute.tex')

        if not os.path.isdir(f'{mse_folder}/{eval_type}/plots/{given_feature}'):
            os.makedirs(f'{mse_folder}/{eval_type}/plots/{given_feature}')

        
        plt.figure(figsize=(16,9))
        plt.plot(L[:(-neg-1)], result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
        plt.plot(L[:(-neg-1)], result_mse_plots['Transformer'], 'tab:blue', label='Transformer', marker='o')
        plt.plot(L[:(-neg-1)], result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
        plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(f'Length of contiguous missing values', fontsize=16)
        plt.ylabel(f'MSE', fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(f'{mse_folder}/{eval_type}/plots/{given_feature}/L-vs-MSE-all-models-{features[feature_idx]}-{len(L)}.png', dpi=300)
        plt.close()

        plt.figure(figsize=(16,9))
        plt.plot(L[:(-neg-1)], result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
        plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(f'Length of contiguous missing values', fontsize=16)
        plt.ylabel(f'MSE', fontsize=16)
        plt.legend()
        plt.savefig(f'{mse_folder}/{eval_type}/plots/{given_feature}/L-vs-MSE-BRITS-{features[feature_idx]}-{len(L)}.png', dpi=300)
        plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(L, result_mse_plots['Transformer'], 'tab:blue', label='Transformer', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=16)
        # plt.ylabel(f'MSE', fontsize=16)
        # plt.legend()
        # plt.savefig(f'{mse_folder}/{eval_type}/plots/{given_feature}/L-vs-MSE-BRITS-{features[feature_idx]}-{len(L)}.png', dpi=300)
        # plt.close()

        # plt.figure(figsize=(16,9))
        # plt.plot(L, result_mse_plots['MICE'], 'tab:cyan', label='MICE', marker='o')
        # plt.title(f'Length of missing values vs Imputation MSE for feature = {features[feature_idx]}, year={eval_season}', fontsize=20)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.xlabel(f'Length of contiguous missing values', fontsize=16)
        # plt.ylabel(f'MSE', fontsize=16)
        # plt.legend()
        # plt.savefig(f'{mse_folder}/{eval_type}/plots/{given_feature}/L-vs-MSE-MICE-{features[feature_idx]}-{len(L)}.png', dpi=300)
        # plt.close()


def forward_parse_id(fs, x, y, feature_impute_idx, trial_num=-1):

    idx1 = np.where(~np.isnan(x[:,feature_impute_idx]))[0]
    # print(f"idx1: {idx1}")
    idx1 = idx1 * len(features) + feature_impute_idx
    # print(f"idx1-1: {idx1}")
    # exit()

    # randomly eliminate 10% values as the imputation ground-truth
    # print('not null: ',np.where(~np.isnan(evals)))
    indices = idx1.tolist()

    start_idx = indices[(trial_num + 1)] #np.random.choice(indices, 1)
    start = indices.index(start_idx)
    # if start + length <= len(indices): 
    #     end = start + length
    # else:
    #     end = len(indices)
    end = len(indices) - start + 1
    indices = np.array(indices)[start:end]

    # global real_values
    # real_values = evals[indices]
    # if dependent_features is not None and len(dependent_features) != 0:
    #     inv_indices = (indices - feature_impute_idx)//len(features)
    #     # if length > 1:
    #     #     print('inv: ', inv_indices)
    #     for i in inv_indices:
    #         x[i, dependent_features] = np.nan 

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
        draws = []
        x_axis = []
        for i in range(len(non_missing_indices)-1):
            fs = open(filename, 'w')
            missing_indices = forward_parse_id(fs, X[season_idx], Y[season_idx], feature_idx, i)
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
                imputed_brits = imputation_brits[row_indices, feature_idx]
                real_values = eval_[row_indices, feature_idx]

                brits_mse = ((real_values - imputed_brits) ** 2).mean()
                draws.append(brits_mse)
                x_axis.append(i+1)
        
        if not os.path.isdir(f'{forward_folder}/plots/LTE50/'):
            os.makedirs(f'{forward_folder}/plots/LTE50/')

        plt.figure(figsize=(16,9))
        plt.plot(x_axis, draws, 'tab:blue', label='BRITS', marker='o')
        plt.title(f'Length of previous existing LTE50 vs Imputation MSE for feature = {features[feature_idx]}, year={given_season}', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(f'Length of exisiting LTE50 values', fontsize=16)
        plt.ylabel(f'MSE', fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(f'{forward_folder}/plots/LTE50/L-vs-MSE-brits-{features[feature_idx]}-{given_season}.png', dpi=300)
        plt.close()
                








####################### Draw data plots #######################

def do_data_plots(data_folder, missing_length, is_original=False):
    print(f'Season array: {len(season_array)}')
    filename = 'json/json_eval_3_LT'
    missing_num = missing_length
    if is_original:
        data_folder += '/original_missing'
    else:
        data_folder += '/eval_missing'
    for given_season in seasons.keys():
        season_idx = seasons[given_season]
        given_features = ['LTE50']#features
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

                # imputation_mice = mice_impute.transform(ret_eval)
                # draws['MICE'] = unnormalize(imputation_mice[:, feature_idx], mean, std, feature_idx)

                draw_data_plot(draws, features[feature_idx], given_season, folder=data_folder, is_original=is_original)

                # graph_bar_diff_multi(draws['real'][row_indices], draws, f'Difference From Gorund Truth for {given_feature} in 2020-2021', np.arange(missing_num), 'Days', given_feature, '2020-2021', given_feature, missing=row_indices)


# eval_folder = 'eval_dir_LT/LT'
# if not os.path.isdir(eval_folder):
#     os.makedirs(eval_folder)
# do_evaluation(eval_folder, 'cont', '2020-2021')
data_plots_folder = 'data_plots_LT/LT'
if not os.path.isdir(data_plots_folder):
    os.makedirs(data_plots_folder)
do_data_plots(data_plots_folder, 10, is_original=True)
do_data_plots(data_plots_folder, 10, is_original=False)

# forward_folder = 'forward_folder'
# forward_prediction(forward_folder)






