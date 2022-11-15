import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from saits.dataset_for_mit import DatasetForMIT
from process_data import *
from saits.diff_model import DiffModel
import os

def create_synthetic_data(config):
    num_seasons = 32
    num_steps = config['n_steps']
    num_features = config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]



    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            if j == 3:
                data[i, :, j] = np.sin(np.linspace(0, 2 * np.pi, data.shape[1]))
                continue
            low = np.random.uniform(value_range[j][0], value_range[j][1])
            high = np.random.uniform(value_range[j][2], value_range[j][3])
            if j == 0:
                data[i, :, j] = np.linspace(low, high, data.shape[1])
            elif j == 1:
                data[i, :, j] = np.linspace(low ** 0.5, high ** 0.5, data.shape[1]) ** 2
            else:
                data[i, :, j] = np.linspace(np.cbrt(low), np.cbrt(high), data.shape[1]) ** 3
            
    data_rows = data.reshape((-1, num_features))
    mean = np.mean(data_rows, axis=0)
    std = np.std(data_rows, axis=0)
    return data, mean, std



def train(
    model,
    config,
    train_loader=None,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    # df = pd.read_csv(f'ColdHardiness_Grape_Merlot_2.csv')
    # modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)# not_original=True)#False, is_year=True)
    # season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)

    # train_season_df = season_df.drop(season_array[-1], axis=0)
    # train_season_df = train_season_df.drop(season_array[-2], axis=0)

    # mean, std = get_mean_std(train_season_df, features)
    
    # X, Y = split_XY(season_df, max_length, season_array, features)

    num_samples = 20#len(season_array) - 2
    num_batches = 2

    # X = X[:-2]
    # Y = Y[:-2]

    rate = 0.1
    is_rand = False
    X, mean, std = create_synthetic_data(config)
    training_set = DatasetForMIT(X, mean, std, rate=rate, is_rand=is_rand)
    train_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True)

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = f"{foldername}/model_diff_saits_mean.model"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0
        model.train()
        # print("Model weights...")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        with tqdm(train_loader) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                
            it.set_postfix(
                ordered_dict={
                    "avg_epoch_loss": avg_loss / num_batches,
                    # "valid_mse": mse_total / evalpoints_total,
                    # "mae_total": mae_total / evalpoints_total
                    "epoch": epoch_no,
                },
                refresh=False,
            )

            lr_scheduler.step()
        # print("Model weights...")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        model.eval()
        with tqdm(train_loader) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                mse_current, mae_current, eval_points = batch_eval(model, train_batch, nsample=num_samples)
                mse_total += (mse_current / eval_points)
                mae_total += (mae_current / eval_points)
                evalpoints_total += eval_points
            print(f"Epoch {epoch_no}: mse: {mse_total  / num_batches} and mae: {mae_total / num_batches}")
            it.set_postfix(
                ordered_dict={
                    # "avg_epoch_loss": avg_loss / batch_no,
                    "valid_mse": mse_total  / num_batches,
                    "valid_mae": mae_total / num_batches,
                    "epoch": epoch_no,
                },
                refresh=False,
            )
        model.train()
        # if epoch_no == 3:
        #     break

    if foldername != "":
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        torch.save(model.state_dict(), output_path)

def batch_eval(model, test_batch, nsample=200, scaler=1, mean_scaler=0):
    mid = True
    with torch.no_grad():
        output = model.evaluate(test_batch, nsample)

        samples, c_target, eval_points, observed_points = output
        # samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
        # c_target = c_target.permute(0, 2, 1)  # (B,L,K)
        # eval_points = eval_points.permute(0, 2, 1)
        # observed_points = observed_points.permute(0, 2, 1)
        if mid:
            samples_median = samples.median(dim=1)
        else:
            samples_median = torch.mean(samples, dim=1)
    # all_target.append(c_target)
    # all_evalpoint.append(eval_points)
    # all_observed_point.append(observed_points)
    # all_observed_time.append(observed_time)
    # all_generated_samples.append(samples)
        if mid:
            mse_current = (
                ((samples_median.values - c_target) * eval_points) ** 2
            ) * (scaler ** 2)
            mae_current = (
                torch.abs((samples_median.values - c_target) * eval_points) 
            ) * scaler
        else:
            mse_current = (
                ((samples_median - c_target) * eval_points) ** 2
            ) * (scaler ** 2)
            mae_current = (
                torch.abs((samples_median - c_target) * eval_points) 
            ) * scaler

    # mse_total += mse_current.sum().item()
    # mae_total += mae_current.sum().item()
    # evalpoints_total += eval_points.sum().item()
    # model.train()
    return mse_current.sum().item(), mae_current.sum().item(), eval_points.sum().item()


if __name__ == '__main__':
    config = {
        'batch_size': 16,
        'epochs': 100,
        'n_steps': 50, #252,
        'diff_steps': 100, #150,
        'n_features': 4, #len(features),
        'layers': 4,
        'n_layers': 4,
        'd_model': 256,
        'd_inner': 128,
        'n_head': 4,
        'd_k': 64,
        "d_v": 64,
        'dropout': 0.1,
        'patience': 300,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.00001,
        'beta_end': 0.05,
        'schedule': "quad",
        'time_emb': 128,
        'target_strategy': "random",
        "lr": 1.0e-3,
        'time_strategy': 'add'
    }
    model = DiffModel(config, is_epsilon=False)
    train(model, config, foldername="saved_diff_model_w_sampling_synth")
    # X, mean, std = create_synthetic_data()
    # print(f"X: {X}\n\nmean: {mean}\nstd: {std}")
