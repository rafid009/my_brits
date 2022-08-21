import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from lam_model.nn import net
from process_data import *
from lam_model.utils import *
import os
from tqdm import tqdm
import sys
import time
import copy
from sklearn.impute import SimpleImputer
np.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

n_random = 0.2

features = [
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
    'ETR'#, # ???
    # 'LTE50'
    # 'SEASON_JDAY'
]

complete_seasons = [4, 5, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]


def initialize_input(impute_model, n_random, imputed=True, original=False, station=False, mean=False):
    if station:
        input_file = f"./ColdHardiness_Grape_Merlot_new_synthetic"
        df = pd.read_csv(f"{input_file}.csv")
    elif imputed:
        input_file = f"./abstract_imputed/ColdHardiness_Grape_Merlot_imputed"
        df = pd.read_csv(f"{input_file}_{impute_model}.csv")#_{n_random}.csv")
    else:
        if original:
            input_file = f"./ColdHardiness_Grape_Merlot_2"
            df = pd.read_csv(f"{input_file}.csv")
        else:
            input_file = f"./ColdHardiness_Grape_Merlot_new_synthetic"
            df = pd.read_csv(f"{input_file}_{n_random}.csv")
    if imputed:
        modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True, imputed=True)#False, is_year=True)
    else:
        modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#, imputed=False)
    season_df, seasons_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)
    # seasons_array = [seasons_array[i] for i in complete_seasons]
    # seasons_complete = []
    # for i in range(len(seasons_array)):
    #     indices = copy.deepcopy(seasons_array[i])
    #     seasons_complete.extend(indices)
    # season_df = season_df.loc[seasons_complete]
    imputed_season_df = season_df.copy()
    if mean:
        train_df = season_df.drop(seasons_array[-2], axis=0)
        train_df = train_df.drop(seasons_array[-1], axis=0)
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        mean_imputer.fit(train_df[features])
        
        imputed_season_df.loc[:, features] = mean_imputer.transform(season_df[features])

    elif not imputed:
        imputed_season_df.loc[:, features] = season_df[features].interpolate(method='linear', limit_direction='both')

    # print(f"imputed: {imputed_season_df.isna().sum()}")
    
    # input_file = f"./ColdHardiness_Grape_Merlot_new_synthetic"
    # train_0_df = pd.read_csv(f"{input_file}.csv")

    # train_0_modified_df, dormant_seasons = preprocess_missing_values(train_0_df, features, is_dormant=True, imputed=True)#False, is_year=True)
    # train_0_season_df, seasons_array, max_length = get_seasons_data(train_0_modified_df, dormant_seasons, features, is_dormant=True)

    # train_season_df_0 = train_0_season_df.drop(seasons_array[-1], axis=0)
    # train_season_df_0 = train_season_df_0.drop(seasons_array[-2], axis=0)

    x_mean, x_std = get_mean_std_rnn(imputed_season_df, features)

    x_train, y_train = split_and_normalize(imputed_season_df, max_length, seasons_array[:-2], features, x_mean, x_std)

    x_test, y_test = split_and_normalize(imputed_season_df, max_length, seasons_array[-2:], features, x_mean, x_std)
    return x_train, y_train, x_test, y_test

def initialize_model(impute_model, x_train, n_random):
    model = net(np.array(x_train).shape[-1])
    model_path = f"./rnn_models/pred_model_{impute_model}.pt"#_{n_random}.pt"
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    criterion.to(device)
    return model, optimizer, criterion

def training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion):
    # cultivar_name = cultivar_file.split('/')[-1].split('.')[0]
    # model = net(np.array(x_train).shape[-1])

    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(x_train, y_train)
    trainLoader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    val_dataset = TensorDataset(x_test, y_test)
    valLoader = DataLoader(
        val_dataset, batch_size=x_test.shape[0], shuffle=False)
    #
    best_loss = 999999
    for epoch in range(args['epochs']):

        # Training Loop
        with tqdm(trainLoader, unit="batch") as tepoch:
            model.train()

            tepoch.set_description(f"Epoch {epoch + 1}/{args['epochs']} [T]")
            total_loss = 0
            count = 0
            for i, (x, y) in enumerate(trainLoader):
                x_torch = x.to(device)
                y_torch = y.to(device)

                count += 1

                out_lt_50, _ = model(x_torch)

                optimizer.zero_grad()       # zero the parameter gradients
                # print(f"y: {y.shape}")
                # print(f"isna: {np.isnan(y)}")
                # yn = np.isnan(y)
                # for i in yn:
                #     print(i)
                n_nan = get_not_nan(y[:, :, 0])  # LT10/50/90 not NAN
                # print(f"nan: {n_nan.shape}, out_50: {out_lt_50.shape}")
                # print(f"y_torch: {y.shape}\nleft shape: {out_lt_50[n_nan[0], n_nan[1]].shape}")
                # print(f"n_nan: {n_nan}\nleft: {out_lt_50[n_nan[0], n_nan[1]]}\nright: {y_torch[n_nan[0], n_nan[1]]}")
                # left = out_lt_50[n_nan[0], n_nan[1]]
                # print(f"left: {np.isnan(left.detach().numpy())}")
                loss_lt_50 = criterion(
                    torch.squeeze(out_lt_50[:, :, 0][n_nan[0], n_nan[1]]), y_torch[:, :, 0][n_nan[0], n_nan[1]])  # LT50 GT

                n_nan = get_not_nan(y[:, :, 1])  # LT10/50/90 not NAN
                loss_lt_50_next = criterion(
                    torch.squeeze(out_lt_50[:, :, 1][n_nan[0], n_nan[1]]), y_torch[:, :, 1][n_nan[0], n_nan[1]])

                # n_nan = get_not_nan(y[:, :,2])  # LT10/50/90 not NAN
                # loss_lt_50_next_2 = criterion(
                #     torch.squeeze(out_lt_50[:, :, 2][n_nan[0], n_nan[1]]), y_torch[:, :, 2][n_nan[0], n_nan[1]])
                #loss = loss_lt_10 + loss_lt_50 + loss_lt_90 + loss_ph
                loss = loss_lt_50 + loss_lt_50_next

                loss.backward()             # backward +
                optimizer.step()            # optimize

                total_loss += loss.item()

                tepoch.set_postfix(Train_Loss=total_loss / count)
                tepoch.update(1)

        # Validation Loop
        with torch.no_grad():
            with tqdm(valLoader, unit="batch") as tepoch:

                model.eval()

                tepoch.set_description(
                    f"Epoch {epoch + 1}/{args['epochs']} [V]")
                total_loss = 0
                count = 0
                for i, (x, y) in enumerate(valLoader):
                    x_torch = x.to(device)
                    y_torch = y.to(device)
                    count += 1
                    out_lt_50, _ = model(x_torch)
                    # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                    
                    n_nan = get_not_nan(y[:, :, 0])  # LT10/50/90 not NAN
                    loss_lt_50 = criterion(
                        torch.squeeze(out_lt_50[:, :, 0][n_nan[0], n_nan[1]]), y_torch[:, :, 0][n_nan[0], n_nan[1]])  # LT50 GT

                    n_nan = get_not_nan(y[:, :, 1])  # LT10/50/90 not NAN
                    loss_lt_50_next = criterion(
                        torch.squeeze(out_lt_50[:, :, 1][n_nan[0], n_nan[1]]), y_torch[:, :, 1][n_nan[0], n_nan[1]])
                    # n_nan = get_not_nan(y[:, :,2])  # LT10/50/90 not NAN
                    # loss_lt_50_next_2 = criterion(
                    #     torch.squeeze(out_lt_50[:, :, 2][n_nan[0], n_nan[1]]), y_torch[:, :, 2][n_nan[0], n_nan[1]])
                    #loss = loss_lt_10 + loss_lt_50 + loss_lt_90 + loss_ph
                    loss = loss_lt_50 + loss_lt_50_next# + loss_lt_50_next_2
                    total_loss += loss.item()

                    tepoch.set_postfix(Val_Loss=total_loss / count)
                    tepoch.update(1)
                val_loss = total_loss / count
                if val_loss < best_loss:
                    best_loss = val_loss
                    modelSavePath = "./rnn_models/"
                    if not os.path.isdir(modelSavePath):
                        os.makedirs(modelSavePath)

                    torch.save(model.state_dict(), os.path.join(modelSavePath, args['name'] + ".pt"))

    return loss_lt_50.item(), None, best_loss

def evaluate(model, x_test, y_test, batch_size, criterion):
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    # print(x_test.shape, y_test.shape)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    seasons = ['2020-2021', '2021-2022']
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:

            model.eval()

            
            total_loss = 0
            count = 0
            for i, (x, y) in enumerate(test_loader):
                tepoch.set_description(
                    f"Test {i + 1} [V]")
                x_torch = x.to(device)
                y_torch = y.to(device)
                count += 1
                out_lt_50, _ = model(x_torch)
                # getting non nan values is slow right now due to copying to cpu, write pytorch gpu version
                
                n_nan = get_not_nan(y[:, :, 0])  # LT10/50/90 not NAN
                loss_lt_50 = criterion(
                    torch.squeeze(out_lt_50[:, :, 0][n_nan[0], n_nan[1]]), y_torch[:, :, 0][n_nan[0], n_nan[1]]) 
                     # LT50 GT

                n_nan = get_not_nan(y[:, :, 1])  # LT10/50/90 not NAN
                # print(f"lt50 next: {out_lt_50[:, :, 1][n_nan[0], n_nan[1]]}\ny next: {y_torch[:, :, 1][n_nan[0], n_nan[1]]}")
                loss_lt_50_next = criterion(
                    torch.squeeze(out_lt_50[:, :, 1][n_nan[0], n_nan[1]]), y_torch[:, :, 1][n_nan[0], n_nan[1]]) 
                
                # n_nan = get_not_nan(y[:, :,2])  # LT10/50/90 not NAN
                # loss_lt_50_next_2 = criterion(
                #     torch.squeeze(out_lt_50[:, :, 2][n_nan[0], n_nan[1]]), y_torch[:, :, 2][n_nan[0], n_nan[1]])
                #loss = loss_lt_10 + loss_lt_50 + loss_lt_90 + loss_ph
                loss = loss_lt_50 + loss_lt_50_next# + loss_lt_50_next_2
                total_loss += loss.item()
                print(f"{seasons[i]} same mse: {loss_lt_50.item()}\nnext mse: {loss_lt_50_next.item()}")#\next 2 mse: {loss_lt_50_next_2.item()}")
                tepoch.set_postfix(Val_Loss=total_loss / count)
                tepoch.update(1)

def format_seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)



# impute_model = 'station_replace'
# args = {
#     'name': f"pred_model_{impute_model}_{n_random}",
#     'batch_size': 16,
#     'epochs': 800
# }
# x_train, y_train, x_test, y_test = initialize_input(impute_model, n_random, station=True)
# model, optimizer, criterion = initialize_model(impute_model, x_train, n_random)
# start_time = time.time()
# _, best_loss = training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion)
# end_time = time.time()
# print(f"total time taken: {format_seconds_to_hhmmss(end_time - start_time)}")
# print(f"Predicitve {impute_model} model mse: {best_loss}")
# model_path = f"./rnn_models/pred_model_{impute_model}_{n_random}.pt"
# model.load_state_dict(torch.load(model_path))
# evaluate(model, x_test, y_test, 1, criterion)
# print()

impute_model = 'mean_orig' 
args = {
    'name': f"pred_model_{impute_model}_nn",
    'batch_size': 16,
    'epochs': 800
}
print(f"Predicitve {impute_model}:")
x_train, y_train, x_test, y_test = initialize_input(impute_model, n_random, imputed=False, original=True, mean=True)
model, optimizer, criterion = initialize_model(impute_model, x_train, n_random)
start_time = time.time()
_, _, best_loss = training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion)
end_time = time.time()
print(f"total time taken: {format_seconds_to_hhmmss(end_time - start_time)}")
print(f"model mse: {best_loss}")
# print(f"Predicitve {impute_model}")
model_path = f"./rnn_models/pred_model_{impute_model}_nn.pt"#_{n_random}.pt"
model.load_state_dict(torch.load(model_path))
evaluate(model, x_test, y_test, 1, criterion)
print()

impute_model = 'linear_orig' 
args = {
    'name': f"pred_model_{impute_model}_nn",
    'batch_size': 16,
    'epochs': 800
}
print(f"Predicitve {impute_model}:")
x_train, y_train, x_test, y_test = initialize_input(impute_model, n_random, imputed=False, original=True)
model, optimizer, criterion = initialize_model(impute_model, x_train, n_random)
start_time = time.time()
_, _, best_loss = training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion)
end_time = time.time()
print(f"total time taken: {format_seconds_to_hhmmss(end_time - start_time)}")
print(f"model mse: {best_loss}")
# print(f"Predicitve {impute_model}")
model_path = f"./rnn_models/pred_model_{impute_model}_nn.pt"#_{n_random}.pt"
model.load_state_dict(torch.load(model_path))
evaluate(model, x_test, y_test, 1, criterion)
print()

impute_model = 'brits_orig' 
args = {
    'name': f"pred_model_{impute_model}_nn",
    'batch_size': 16,
    'epochs': 1100
}
print(f"Predicitve {impute_model}:")
x_train, y_train, x_test, y_test = initialize_input(impute_model, n_random)
model, optimizer, criterion = initialize_model(impute_model, x_train, n_random)
start_time = time.time()
_, _, best_loss = training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion)
end_time = time.time()
print(f"total time taken: {format_seconds_to_hhmmss(end_time - start_time)}")
print(f"model mse: {best_loss}")
# print(f"Predicitve {impute_model}")
model_path = f"./rnn_models/pred_model_{impute_model}_nn.pt"#{n_random}.pt"
model.load_state_dict(torch.load(model_path))
evaluate(model, x_test, y_test, 1, criterion)
print()

impute_model = 'saits_orig' 
args = {
    'name': f"pred_model_{impute_model}_nn",
    'batch_size': 16,
    'epochs': 1100
}
print(f"Predicitve {impute_model}:")
x_train, y_train, x_test, y_test = initialize_input(impute_model, n_random)
model, optimizer, criterion = initialize_model(impute_model, x_train, n_random)
start_time = time.time()
_, _, best_loss = training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion)
end_time = time.time()
print(f"total time taken: {format_seconds_to_hhmmss(end_time - start_time)}")
print(f"model mse: {best_loss}")
# print(f"Predicitve {impute_model}")
model_path = f"./rnn_models/pred_model_{impute_model}_nn.pt"#{n_random}.pt"
model.load_state_dict(torch.load(model_path))
evaluate(model, x_test, y_test, 1, criterion)
print()

impute_model = 'mice_orig' 
args = {
    'name': f"pred_model_{impute_model}_nn",
    'batch_size': 16,
    'epochs': 1100
}
print(f"Predicitve {impute_model}:")
x_train, y_train, x_test, y_test = initialize_input(impute_model, n_random)
model, optimizer, criterion = initialize_model(impute_model, x_train, n_random)
start_time = time.time()
_, _, best_loss = training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion)
end_time = time.time()
print(f"total time taken: {format_seconds_to_hhmmss(end_time - start_time)}")
print(f"model mse: {best_loss}")
# print(f"Predicitve {impute_model}")
model_path = f"./rnn_models/pred_model_{impute_model}_nn.pt"#{n_random}.pt"
model.load_state_dict(torch.load(model_path))
evaluate(model, x_test, y_test, 1, criterion)
print()

impute_model = 'mvts_orig' 
args = {
    'name': f"pred_model_{impute_model}_nn",
    'batch_size': 16,
    'epochs': 1100
}
print(f"Predicitve {impute_model}:")
x_train, y_train, x_test, y_test = initialize_input(impute_model, n_random)
model, optimizer, criterion = initialize_model(impute_model, x_train, n_random)
start_time = time.time()
_, _, best_loss = training_loop(model, x_train, y_train, x_test, y_test, args, optimizer, criterion)
end_time = time.time()
print(f"total time taken: {format_seconds_to_hhmmss(end_time - start_time)}")
print(f"model mse: {best_loss}")
# print(f"model mse: {best_loss}")
model_path = f"./rnn_models/pred_model_{impute_model}_nn.pt"#{n_random}.pt"
model.load_state_dict(torch.load(model_path))
evaluate(model, x_test, y_test, 1, criterion)
print()