import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


def linear_interp(x, y, missing_indicator, show=True):
    y = np.array(y)
    if (not np.isnan(missing_indicator)):
        missing = np.where(y == missing_indicator)[0]
        not_missing = np.where(y != missing_indicator)[0]
    else:
        missing = np.argwhere(np.isnan(y)).flatten() # special case for nan values
        all_idx = np.arange(0, y.shape[0]) 
        not_missing = np.setdiff1d(all_idx, missing)
        
    interp = np.interp(x, not_missing, y[not_missing])

    return interp

def remove_na(df, column_name):
    total_na = df[column_name].isna().sum()
    
    df[column_name] = df[column_name].replace(np.nan, -100)
    df[column_name] = linear_interp(np.arange(df.shape[0]), df[column_name], -100, False)
    if df[column_name].isna().sum() != 0:
        assert False

    return

def split_and_normalize(_df, season_max_length, seasons, features, x_mean, x_std, label=['LTE50']):
    x = []
    y = []
    
    for i, season in enumerate(seasons):
        _x = (_df[features].loc[season, :]).to_numpy()

        _x = np.concatenate((_x, np.zeros((season_max_length - len(season), len(features)))), axis = 0)

        add_array = np.zeros((season_max_length - len(season), len(label)))
        add_array[:] = np.NaN

        add_array_next = np.zeros((season_max_length - len(season) + 1, len(label)))
        add_array_next[:] = np.NaN

        add_array_next_2 = np.zeros((season_max_length - len(season) + 2, len(label)))
        add_array_next_2[:] = np.NaN

        _y = _df.loc[season, :][label].to_numpy()
        y_next = _df.loc[season[1:], :][label].to_numpy()
        y_next_2 = _df.loc[season[2:], :][label].to_numpy()
        # print(f"y_: {_y.shape}, add_array: {add_array.shape}")
        _y = np.concatenate((_y, add_array), axis=0)
        # print(f"y_next: {y_next.shape}, add_array: {add_array_next.shape}")
        y_next = np.concatenate((y_next, add_array_next), axis=0)
        y_next_2 = np.concatenate((y_next_2, add_array_next_2), axis=0)
        # _y = np.squeeze(_y)
        # print(f"y_1: {_y.shape}, y_2: {y_next.shape}")
        _y = np.concatenate((_y, y_next, y_next_2), axis=1)
        # print(f"y_both: {_y.shape}")
        x.append(_x)
        y.append(_y)

    x = np.array(x)
    y = np.array(y)
    # print(f"y_slit: {y.shape}")
    
    norm_features_idx = np.arange(0, x_mean.shape[0])
        
    x[:, :, norm_features_idx]  = (x[:, :, norm_features_idx] - x_mean) / x_std # normalize
    
    return x, y

def get_dormant_seasons(df):
    seasons = []
    last_x = 0
    idx = -1
    season_max_length = 0
    for x in df[df["DORMANT_SEASON"] == 1].index.tolist():
        if x - last_x > 1:
            seasons.append([])
            if idx > -1:
                season_max_length = max(season_max_length, len(seasons[idx]))
            idx += 1
        seasons[idx].append(x)
        last_x = x
        
    season_max_length = max(season_max_length, len(seasons[idx]))
    return seasons, season_max_length

def get_mean_std_rnn(df, features):
    x_mean = df[features].mean().to_numpy()
    x_std = df[features].std().to_numpy()
    return x_mean, x_std

def get_not_nan(y):
    return np.argwhere(np.isnan(y) != 1)
