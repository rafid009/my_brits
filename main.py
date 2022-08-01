import os
import torch
import torch.optim as optim
import numpy as np

from pypots.data import mcar, masked_fill
from pypots.imputation import SAITS
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

def train(model, n_epochs, batch_size, model_path, data_file='./json/json_LT'):
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
            tepoch.set_postfix(MSE=mse)
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



if __name__ == "__main__":
    n_features = 21
    model_dir = "./model_abstract"

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # # BRITS
    # print(f"=========== BRITS Training Starts ===========")
    # prepare_brits_input()
    # batch_size = 16
    # n_epochs = 4000
    # RNN_HID_SIZE = 64
    # IMPUTE_WEIGHT = 0.5
    # LABEL_WEIGHT = 1
    # model_name = 'BRITS'
    # model_path_name = 'BRITS'
    # model_path = f'{model_dir}/model_'+model_path_name+'_LT.model'
    
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

    # # SAITS
    # print(f"=========== SAITS Training Starts ===========")

    # df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
    # modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#False, is_year=True)
    # season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)
    # train_season_df = season_df.drop(season_array[-1], axis=0)
    # train_season_df = train_season_df.drop(season_array[-2], axis=0)
    # mean, std = get_mean_std(season_df, features)

    # X, Y = split_XY(season_df, max_length, season_array, features)

    # num_samples = len(season_array) - 2  #len(X['RecordID'].unique())

    # X = X[:-2]
    # Y = Y[:-2]

    # for i in range(X.shape[0]):
    #     X[i] = (X[i] - mean)/std
    # # print(f"X: {X.shape}")
    # # X = X.reshape(num_samples, 48, -1)
    # X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
    # X = masked_fill(X, 1 - missing_mask, np.nan)
    # # Model training. This is PyPOTS showtime. 
    # saits = SAITS(n_steps=252, n_features=len(features), n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=1000, patience=100)
    # saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.

    # filename = f'{model_dir}/model_saits_e1000_21.model'
    # pickle.dump(saits, open(filename, 'wb'))

    # imputation = saits.impute(X)  # impute the originally-missing values and artificially-missing values
    # mse = cal_mse(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
    # print(f"SAITS Validation MSE: {mse}")
    # print(f"=========== SAITS Training Ends ===========")

    # # MICE
    # print(f"=========== MICE Training Starts ===========")
    # normalized_season_df = train_season_df[features].copy()
    # normalized_season_df = (normalized_season_df - mean) /std
    # mice_impute = IterativeImputer(random_state=0, max_iter=20)
    # mice_impute.fit(normalized_season_df[features].to_numpy())
    # filename = f'{model_dir}/model_mice.model'
    # pickle.dump(mice_impute, open(filename, 'wb'))

    # print(f"=========== MICE Training Ends ===========")

    # MVTS
    print(f"=========== MVTS Training Starts ===========")
    params = {
        'config_filepath': None, 
        'output_dir': './transformer/output', 
        'data_dir': './transformer/data_dir/', 
        'load_model': None, 
        'resume': False, 
        'change_output': False, 
        'save_all': False, 
        'experiment_name': 'Transformer_Imputation_Training', 
        'comment': 'pretraining through imputation', 
        'no_timestamp': False, 
        'records_file': 'Imputation_records.csv', 
        'console': False, 
        'print_interval': 1, 
        'gpu': '0', 
        'n_proc': 1, 
        'num_workers': 0, 
        'seed': None, 
        'limit_size': None, 
        'test_only': None, 
        'data_class': 'agaid', 
        'labels': None, 
        'test_from': None, 
        'test_ratio': 0, 
        'val_ratio': 0.1, 
        'pattern': 'Merlot_train', 
        'val_pattern': None, 
        'test_pattern': None, 
        'normalization': 'standardization', 
        'norm_from': None, 
        'subsample_factor': None, 
        'task': 'imputation', 
        'masking_ratio': 0.15, 
        'mean_mask_length': 20.0, 
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
        'batch_size': 16, 
        'l2_reg': 0, 
        'global_reg': False, 
        'key_metric': 'loss', 
        'freeze': False, 
        'model': 'transformer', 
        'max_seq_len': 252,#366, 
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

    data_folder = './transformer/data_dir'
    df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
    modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#False, is_year=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)

    season_df['season_id'] = 0

    train_season_df = season_df.drop(season_array[-1], axis=0)
    train_season_df = train_season_df.drop(season_array[-2], axis=0)
    add_season_id_and_save(data_folder, train_season_df, season_array[:-2], 'ColdHardiness_Grape_Merlot_train.csv')
    run_transformer(params)
    print(f"=========== MVTS Training Ends ===========")

    # # NAOMI
    # print(f"=========== NAOMI Training Starts ===========")



    


    

