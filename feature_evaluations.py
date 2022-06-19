from main import *
import copy
import numpy as np
import pandas as pd
from process_data import *
import json
import os
import torch
from models.brits import BRITSModel as BRITS
import utils
import data_loader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import warnings
import matplotlib
warnings.filterwarnings("ignore")
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
import sys
np.set_printoptions(threshold=sys.maxsize)

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

############## Data Load and Preprocess Starts ##############


##############  Data Load and Preprocess Ends  ##############

############## Draw Functions Starts ##############
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
        ax.set_title('Feature = '+f+' Season = '+season_idx+' original data', fontsize=27)
        plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        ax.xticks(fontsize=20)
        ax.yticks(fontsize=20)

        ax = plt.subplot(312)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by MICE', fontsize=27)
        plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        ax.xticks(fontsize=20)
        ax.yticks(fontsize=20)

        ax = plt.subplot(313)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by BRITS', fontsize=27)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        ax.xticks(fontsize=20)
        ax.yticks(fontsize=20)
    else:
        ax = plt.subplot(411)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' original data', fontsize=27)
        plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        ax.xticks(fontsize=20)
        ax.yticks(fontsize=20)

        ax = plt.subplot(412)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' missing data', fontsize=27)
        plt.plot(np.arange(results['missing'].shape[0]), results['missing'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        ax.xticks(fontsize=20)
        ax.yticks(fontsize=20)

        ax = plt.subplot(413)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by MICE', fontsize=27)
        plt.plot(np.arange(results['MICE'].shape[0]), results['MICE'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        ax.xticks(fontsize=20)
        ax.yticks(fontsize=20)

        ax = plt.subplot(414)
        ax.set_title('Feature = '+f+' Season = '+season_idx+' imputed by BRITS', fontsize=27)
        plt.plot(np.arange(results['BRITS'].shape[0]), results['BRITS'], 'tab:blue')
        ax.set_xlabel('Days', fontsize=25)
        ax.set_ylabel('Values', fontsize=25)
        ax.xticks(fontsize=20)
        ax.yticks(fontsize=20)

    plt.tight_layout(pad=5)
    plt.savefig(f"{folder}/{f}/{f}-imputations-season-{season_idx}.png", dpi=300)
    plt.close()
##############  Draw Functions Ends  ##############


############## Model Input Process Functions ##############
def parse_delta(masks, dir_, features):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(masks.shape[0]):
        if h == 0:
            deltas.append(np.ones(len(features)))
        else:
            deltas.append(np.ones(len(features)) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, dir_, features):
    deltas = parse_delta(masks, dir_, features)

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

def parse_id(fs, x, y, mean, std, feature_impute_idx, length, features, trial_num=-1, dependent_features=None, real_test=True, random_start=False, remove_features=None):

    if real_test:
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
            # print(f"tiral_num: {trial_num}")
            if trial_num >= len(indices):
                return []
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
        if remove_features is not None and len(remove_features) != 0:
            inv_indices = (indices - feature_impute_idx)//len(features)
            # if length > 1:
            #     print('inv: ', inv_indices)
            for i in inv_indices:
                x[i, remove_features] = np.nan
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
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward', features=features)
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', features=features)

    rec = json.dumps(rec)

    fs.write(rec + '\n')
    return indices

def train_parse_id(x, y, fs, mean, std, features):
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




def train_evaluate_removed_features(mse_folder):
    RNN_HID_SIZE = 64
    IMPUTE_WEIGHT = 0.5
    LABEL_WEIGHT = 1
    n_epochs = 4000
    batch_size = 16
    model_dir = './saved_models_greedy/'

    L = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    filename = 'json/json_eval_2'
    feature_set = [
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
    ]

    features_to_remove = [
        'MEAN_AT',
        'MIN_REL_HUMIDITY',
        'MIN_DEWPT',
        'P_INCHES', # precipitation
        'WS_MPH', # wind speed. if no sensor then value will be na
        'LW_UNITY', # leaf wetness sensor
        'SR_WM2', # solar radiation # different from zengxian
        'ST8', # soil temperature # diff from zengxian
        #'MSLP_HPA', # barrometric pressure # diff from zengxian
        'ETO', # evaporation of soil water lost to atmosphere
        'ETR', # ???
        'LTE50'
    ]

    for feature in feature_set:
        to_remove = [f for f in features_to_remove if f != feature]
        to_remove.append('all')
        print(f"For feature={feature}")
        for r_feat in to_remove:
            if r_feat == 'all':
                dependent_feature_removes = []
            else:
                dependent_feature_removes = [f for f in feature_dependency[r_feat.split('_')[-1]]]
            curr_features = [f for f in feature_set if f not in dependent_feature_removes]
            print(f"For removed: {r_feat}, current feature length: {len(curr_features)}")
            df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
            modified_df, dormant_seasons = preprocess_missing_values(df, curr_features, is_dormant=True)
            season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, curr_features, is_dormant=True)
            train_season_df = season_df.drop(season_array[-1], axis=0)
            train_season_df = train_season_df.drop(season_array[-2], axis=0)

            mean, std = get_mean_std(train_season_df, curr_features)

            X, Y = split_XY(season_df, max_length, season_array, curr_features)
            X = X[:-2]
            Y = Y[:-2]
            fs = open(filename, 'w')
            for i in range(X.shape[0] - 1):
                train_parse_id(X[i], Y[i], fs, mean, std, curr_features)

            fs.close()

            model = BRITS(RNN_HID_SIZE, IMPUTE_WEIGHT, LABEL_WEIGHT, len(curr_features))
            model = model.to(device=device)
            model_path = f"{model_dir}/{feature}/"
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            model_file_names = ''
            for f in dependent_feature_removes:
                model_file_names += f + '-'
            model_path = f"{model_path}/model-{model_file_names}LT.pth"
            train(model, n_epochs, batch_size, model_path, filename)

            test_df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
            test_modified_df, test_dormant_seasons = preprocess_missing_values(test_df, curr_features, is_dormant=True)
            season_df, season_array, max_length = get_seasons_data(test_modified_df, test_dormant_seasons, curr_features, is_dormant=True)

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
            l_needed = []
            model.eval()
            for season in seasons.keys():
                print(f"For season: {season}")
                season_idx = seasons[season]
                feature_idx = curr_features.index(feature)
                X, Y = split_XY(season_df, max_length, season_array, curr_features)
                # print(f'X: {X.shape}, Y: {Y.shape}')
                original_missing_indices = np.where(np.isnan(X[season_idx, :, feature_idx]))[0]

                for l in L:
                    
                    iter = len(season_array[season_idx]) - (l-1) - len(original_missing_indices)
                    print(f"For length = {l}")
                    
                    total_count = 0
                    brits_mse = 0
                    # iter = 1
                    for i in tqdm(range(iter)):
                        real_values = []
                        imputed_brits = []
                        
                        fs = open(filename, 'w')

                        if feature.split('_')[-1] not in feature_dependency.keys():
                            dependent_feature_ids = []
                        else:
                            dependent_feature_ids = [curr_features.index(f) for f in feature_dependency[feature.split('_')[-1]] if f != feature]
                        
                        missing_indices = parse_id(fs, X[season_idx], Y[season_idx], mean, std, feature_idx, l, curr_features, trial_num=i, dependent_features=dependent_feature_ids)
                        fs.close()

                        if len(missing_indices) == 0:
                            continue
                        val_iter = data_loader.get_loader(batch_size=1, filename=filename)

                        for idx, data in enumerate(val_iter):
                            data = utils.to_var(data)
                            row_indices = missing_indices // len(curr_features)

                            ret = model.run_on_batch(data, None)
                            eval_ = ret['evals'].data.cpu().numpy()
                            eval_ = np.squeeze(eval_)
                            imputation_brits = ret['imputations'].data.cpu().numpy()
                            imputation_brits = np.squeeze(imputation_brits)

                            imputed_brits = imputation_brits[row_indices, feature_idx]#unnormalize(imputation_brits[row_indices, feature_idx], mean, std, feature_idx)

                            real_values = eval_[row_indices, feature_idx]#unnormalize(eval_[row_indices, feature_idx], mean, std, feature_idx)

                        brits_mse += ((real_values - imputed_brits) ** 2).mean()
                        total_count += 1
                    l_needed.append(l)
                    print(f"AVG MSE for {iter} runs (sliding window of Length = {l}):\n\tBRITS: {brits_mse/total_count}")#\n\tMICE: {mice_mse/total_count}\n\tTransformer: {transformer_mse/total_count}")

                    results['BRITS'][l] = brits_mse/total_count# f"MSE: {brits_mse}\\MIN (diff GT): {np.round(np.min(np.abs(diff_brits)),5)}\\MAX (diff GT): {np.round(np.max(np.abs(diff_brits)), 5)}\\MEAN (diff GT): {np.round(np.mean(np.abs(diff_brits)), 5)}\\STD (diff GT): {np.round(np.std(np.abs(diff_brits)), 5)}",

                    result_mse_plots['BRITS'].append(brits_mse/total_count)

                result_df = pd.DataFrame(results)
                mse_folder = f'{mse_folder}/{feature}/remove-{r_feat}/mse_results'
                if not os.path.isdir(mse_folder):
                    os.makedirs(mse_folder)
                result_df.to_csv(f'{mse_folder}/results-mse-{season}.csv')
                
                plots_folder = f'{mse_folder}/plots'
                if not os.path.isdir(plots_folder):
                    os.makedirs(plots_folder)
  
                plt.figure(figsize=(16,9))
                plt.plot(l_needed, result_mse_plots['BRITS'], 'tab:orange', label='BRITS', marker='o')
                plt.title(f'Length of missing values vs Imputation MSE for feature = {feature}, year={season}', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel(f'Length of contiguous missing values', fontsize=20)
                plt.ylabel(f'MSE', fontsize=20)
                plt.legend()
                plt.savefig(f'{plots_folder}/L-vs-MSE-BRITS-{curr_features[feature_idx]}-{season}.png', dpi=300)
                plt.close()



if __name__ == "__main__":
    mse_folder = "MSE_PLOTS_greedy"
    train_evaluate_removed_features(mse_folder)