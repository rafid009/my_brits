import copy
import os
from turtle import clear
import torch
import torch.optim as optim
import numpy as np

from pypots.data import mcar, masked_fill
# from pypots.imputation import SAITS
from saits.custom_saits import SAITS
from pypots.utils.metrics import cal_mse
from process_data import *
import pickle

import time
import utils
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from models.brits import BRITSModel as BRITS
from models.brits_i import BRITSModel as BRITS_I
from transformer.src.transformer import run_transformer, add_season_id_and_save
import data_loader
from tqdm import tqdm
from input_process import prepare_brits_input
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
    # np.save('./result/label', save_label)
    return mse

def random_synthetic_missing(season_df, features, n_random=0.2):
    indices_to_choose_from = season_df.index
    df_copy = season_df.copy()

    actual_features = {
        'AT': ['MEAN_AT', 'AVG_AT', 'MIN_AT', 'MAX_AT'],
        'HUMIDITY': ['MIN_REL_HUMIDITY', 'MAX_REL_HUMIDITY', 'AVG_REL_HUMIDITY'],
        'DEWPT': ['MIN_DEWPT', 'AVG_DEWPT', 'MAX_DEWPT'],
        'ST8': ['MIN_ST8', 'ST8', 'MAX_ST8'],
        # 'INCHES': ['P_INCHES'],
        'MPH': ['WS_MPH', 'MAX_WS_MPH'], # wind speed. if no sensor then value will be na
        # 'UNITY': ['LW_UNITY'], # leaf wetness sensor
        'WM2': ['SR_WM2'], # solar radiation # different from zengxian
        'ETO': ['ETO'], # evaporation of soil water lost to atmosphere
        'ETR': ['ETR']
    }

    for feature in actual_features.keys():
        if feature == 'LTE50':
            continue
        indices = np.random.choice(indices_to_choose_from, size=int(len(indices_to_choose_from) * n_random), replace=False, ).tolist()
        df_copy.loc[indices, actual_features[feature]] = np.nan
    return df_copy#.to_numpy()

if __name__ == "__main__":
    n_features = len(features)
    model_dir = "./model_abstract"
    n_random = 0

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    complete_seasons = [4, 5, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    # BRITS
    # print(f"=========== BRITS Training Starts ===========")
    df_synth = pd.read_csv(f'ColdHardiness_Grape_Merlot_2.csv')
    # modified_df, dormant_seasons = preprocess_missing_values(df_synth, features, is_dormant=True, not_original=True)#False, is_year=True)
    # season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)
    
    # train_season_complete = []#[season_array[i] for i in complete_seasons[:-2]]
    # for s in complete_seasons[:-2]:
    #     s_copy = copy.deepcopy(season_array[s])
    #     train_season_complete.extend(s_copy)

    # train_season_df = season_df.loc[train_season_complete]
    # print(f"synth idx: {df_synth.index.tolist()}\nseaon df idx: {season_df.index.tolist()}")
    # df_synth.loc[train_season_df.index.tolist(), :] = random_synthetic_missing(train_season_df, features, n_random=n_random)

    # df_synth.loc[season_array[-2], :] = random_synthetic_missing(season_df.loc[season_array[-2]], features, n_random)
    # df_synth.loc[season_array[-1], :] = random_synthetic_missing(season_df.loc[season_array[-1]], features, n_random)
    # df_synth.to_csv(f'ColdHardiness_Grape_Merlot_new_synthetic_{n_random}.csv', index=False)
    

    modified_df, dormant_seasons = preprocess_missing_values(df_synth, features, is_dormant=True)# not_original=True)#False, is_year=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)
    
    # train_season_complete = []#[season_array[i] for i in complete_seasons[:-2]]
    # for s in complete_seasons[:-2]:
    #     s_copy = copy.deepcopy(season_array[s])
    #     train_season_complete.extend(s_copy)

    # train_season_df = season_df.loc[train_season_complete]
    train_season_df = season_df.drop(season_array[-1], axis=0)
    train_season_df = train_season_df.drop(season_array[-2], axis=0)
    mean, std = get_mean_std(train_season_df, features)
    
    # prepare_brits_input(season_df, season_array, max_length, features, mean, std, model_dir)#, complete_seasons)
    # batch_size = 16
    # n_epochs = 3000
    # RNN_HID_SIZE = 64
    # IMPUTE_WEIGHT = 0.5
    # LABEL_WEIGHT = 1
    # model_name = 'BRITS'
    # model_path_name = 'BRITS'
    # model_path = f'{model_dir}/model_{model_path_name}_LT_orig_consist.model'#synth_{n_random}.model'
    
    # if model_name == 'BRITS':
    #     model = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT, feature_len=n_features)
    # else:
    #     model = BRITS_I(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))

    # if torch.cuda.is_available():
    #     model = model.cuda()

    # train(model, n_epochs, batch_size, model_path, data_file='./json/json_without_LT')
    # print(f"=========== BRITS Training Ends ===========")

    # SAITS
    print(f"=========== SAITS Training Starts ===========")

    

    X, Y = split_XY(season_df, max_length, season_array, features)

    num_samples = len(season_array) - 2  #len(X['RecordID'].unique())

    X = X[:-2]
    Y = Y[:-2]#[complete_seasons[:-2]]

    for i in range(X.shape[0]):
        X[i] = (X[i] - mean)/std
    k = 2
    filename = f'{model_dir}/model_saits_orig_{k}_orig.model'#synth_{n_random}.model'
    # print(f"X: {X.shape}")
    # X = X.reshape(num_samples, 48, -1)
    # X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
    # X = masked_fill(X, 1 - missing_mask, np.nan)
    # Model training. This is PyPOTS showtime. 
    saits = SAITS(n_steps=252, n_features=len(features), n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=200, k=k, original=True)

    saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
    pickle.dump(saits, open(filename, 'wb'))

    # imputation = saits.impute(X)  # impute the originally-missing values and artificially-missing values
    # mse = cal_mse(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
    # print(f"SAITS Validation MSE: {mse}")
    print(f"=========== SAITS Training Ends ===========")

    # # MICE
    # print(f"=========== MICE Training Starts ===========")
    
    # # train_complete_season_df = train_season_df.loc[train_season_complete]
    # normalized_season_df = train_season_df[features].copy()
    # normalized_season_df = (normalized_season_df - mean) /std
    # mice_impute = IterativeImputer(random_state=0, max_iter=30)
    # mice_impute.fit(normalized_season_df[features].to_numpy())
    # filename = f'{model_dir}/model_mice_orig.model'#synth_{n_random}.model'
    # pickle.dump(mice_impute, open(filename, 'wb'))

    # print(f"=========== MICE Training Ends ===========")

    # # MVTS
    # print(f"=========== MVTS Training Starts ===========")
    # params = {
    #     'config_filepath': None, 
    #     'output_dir': './transformer/output',
    #     'data_dir': './transformer/data_dir/',
    #     'load_model': None,
    #     'resume': False,
    #     'change_output': False,
    #     'save_all': False,
    #     'experiment_name': 'mvts-orig', 
    #     'comment': 'pretraining through imputation', 
    #     'no_timestamp': False, 
    #     'records_file': 'Imputation_records.csv', 
    #     'console': False, 
    #     'print_interval': 1, 
    #     'gpu': '0', 
    #     'n_proc': 1, 
    #     'num_workers': 0, 
    #     'seed': None, 
    #     'limit_size': None, 
    #     'test_only': None, 
    #     'data_class': 'agaid', 
    #     'labels': None, 
    #     'test_from': None, 
    #     'test_ratio': 0, 
    #     'val_ratio': 0.1, 
    #     'pattern': 'Merlot_synth', 
    #     'val_pattern': None, 
    #     'test_pattern': None, 
    #     'normalization': 'standardization', 
    #     'norm_from': None, 
    #     'subsample_factor': None, 
    #     'task': 'imputation', 
    #     'masking_ratio': 0.15, 
    #     'mean_mask_length': 20.0, 
    #     'mask_mode': 'separate', 
    #     'mask_distribution': 'geometric', 
    #     'exclude_feats': None, 
    #     'mask_feats': [0, 1], 
    #     'start_hint': 0.0, 
    #     'end_hint': 0.0, 
    #     'harden': True, 
    #     'epochs': 1500, 
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
    #     'max_seq_len': 252,#366, 
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

    # data_folder = './transformer/data_dir'
    # # df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
    # # modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#False, is_year=True)
    # # season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)

    # # season_df['season_id'] = 0
    # # train_season_complete = [season_array[i] for i in complete_seasons[:-2]]
    # # train_season_df = season_df.drop(season_array[-1], axis=0)
    # # train_season_df = train_season_df.drop(season_array[-2], axis=0)
    # # train_season_df = train_season_df.loc[train_season_complete]
    # add_season_id_and_save(data_folder, train_season_df, season_array[:-2], f'ColdHardiness_Grape_Merlot_synth_transformer_{n_random}.csv')
    # run_transformer(params)
    # print(f"=========== MVTS Training Ends ===========")

    # # NAOMI
    # print(f"=========== NAOMI Training Starts ===========")



    


    

