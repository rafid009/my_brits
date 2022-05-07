import pandas as pd
import numpy as np

# df = pd.read_csv('FrostMitigation_Merlot.csv')

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
    'ETR' # ???
]
label = ['LTE50']

feature_dependency = {
  'AT': ['MEAN_AT', 'AVG_AT', 'MIN_AT', 'MAX_AT'],
  'HUMIDITY': ['MIN_REL_HUMIDITY', 'MAX_REL_HUMIDITY', 'AVG_REL_HUMIDITY'],
  'DEWPT': ['MIN_DEWPT', 'AVG_DEWPT', 'MAX_DEWPT'],
  'ST8': ['MIN_ST8', 'ST8', 'MAX_ST8']
  }

def preprocess_missing_values(df, is_dormant=True, is_year=False):
  df['AVG_AT'].replace(-100, np.nan, inplace=True)
  df['MIN_AT'].replace(-100, np.nan, inplace=True)
  df['MAX_AT'].replace(-100, np.nan, inplace=True)
  df['MEAN_AT'].replace(-100, np.nan, inplace=True)

  missing_data_features = []
  for feature in features:
    if df[feature].isna().any():
      missing_data_features.append(feature)

  modified_df = df.copy()

  modified_df['MIN_REL_HUMIDITY'].replace(0, np.nan, inplace=True)
  modified_df['MAX_REL_HUMIDITY'].replace(0, np.nan, inplace=True)
  modified_df['AVG_REL_HUMIDITY'].replace(0, np.nan, inplace=True)
  modified_df['SR_WM2'].replace(0, np.nan, inplace=True)
  modified_df['WS_MPH'].replace(0, np.nan, inplace=True)
  modified_df['MAX_WS_MPH'].replace(0, np.nan, inplace=True)
  # modified_df['LW_UNITY'].replace(0, np.nan, inplace=True)
  # modified_df['P_INCHES'].replace(0, np.nan, inplace=True)

  start_idx = df[df['DATE'] == '2007-07-21'].index.tolist()[0]
  end_idx = df[df['DATE'] == '2007-12-30'].index.tolist()[0]

  start_idx += 1
  for i in range(start_idx, end_idx + 1):
    modified_df.at[i, 'MIN_DEWPT'] = np.nan
    modified_df.at[i, 'MAX_DEWPT'] = np.nan
    modified_df.at[i, 'AVG_DEWPT'] = np.nan
  # print(f"is dormant: {is_dormant}")
  if is_year:
    dormant_seasons = modified_df.index.tolist()
  elif is_dormant:
    dormant_seasons = modified_df[modified_df["DORMANT_SEASON"] == 1].index.tolist()
  else:
    dormant_seasons = modified_df.index.tolist()
  return modified_df, dormant_seasons


def get_seasons_data(modified_df, dormant_seasons, is_dormant=True, is_year=False):
  seasons = []
  last_x = 0
  idx = -1
  season_max_length = 0
  last_seen_drmancy_stat = None

  if is_year:
    # print(modified_df['DATE'])
    prev_year = ''
    season_max_length = 366
    for i in range(len(modified_df)):
      if modified_df.iloc[i]['DATE'].split('-')[0] != prev_year:
        seasons.append([i])
        idx += 1
        prev_year = modified_df.iloc[i]['DATE'].split('-')[0]
      else:
        seasons[idx].append(i)
  else:
    for x in dormant_seasons: # modified_df[modified_df["DORMANT_SEASON"] == 1].index.tolist():
      if is_dormant:
        if x - last_x > 1:
            seasons.append([])
            if idx > -1:
                season_max_length = max(season_max_length, len(seasons[idx]))
            idx += 1
      elif last_seen_drmancy_stat is None:
        last_seen_drmancy_stat = modified_df.iloc[x]['DORMANT_SEASON'] == 1
        seasons.append([])
        idx += 1
      else:
        if modified_df.iloc[x]['DORMANT_SEASON'] == 1:
          if not last_seen_drmancy_stat:
            seasons.append([])
            if idx > -1:
                season_max_length = max(season_max_length, len(seasons[idx]))
            idx += 1
            last_seen_drmancy_stat = True
        else:
          if last_seen_drmancy_stat:
            seasons.append([])
            if idx > -1:
                season_max_length = max(season_max_length, len(seasons[idx]))
            idx += 1
            last_seen_drmancy_stat = False
      seasons[idx].append(x)
      last_x = x
  season_max_length = max(season_max_length, len(seasons[idx]))

  # season_idx = []
  # for season in seasons:
  #   season_idx.extend(season)

  # if is_dormant:
  season_df = pd.DataFrame(modified_df.iloc[dormant_seasons], copy=True)
  # else:
  #   season_df = pd.DataFrame(modified_df, copy=True)
  return season_df, seasons, season_max_length

def get_non_null_LT(df):
  # get all indexes where LT10 is not null
#   idx_LT10_not_null = df[df['LT10'].notnull()].index.tolist()
  try:
    idx_LT50_not_null = df[df['LT50'].notnull()].index.tolist()
  except KeyError:
    idx_LT50_not_null = df[df['LTE50'].notnull()].index.tolist()
#   idx_LT90_not_null = df[df['LT90'].notnull()].index.tolist()

  idx_LT_not_null = set(idx_LT50_not_null) #& set(idx_LT90_not_null) # intersection where LT10, LT50, LT90 are not null
  idx_LT_not_null = sorted(idx_LT_not_null)

  return idx_LT_not_null

def get_season_idx(season_array, index):
  for season_idx in range(len(season_array)):
    # print(f"season_idx: {season_idx}")
    if (index >= season_array[season_idx][0]) and (index <= season_array[season_idx][-1] ):
      # print(f"index: {index}")
      return season_array[season_idx]
  return False

def get_train_idx(season_array, idx_LT_not_null):
  timeseries_idx_train = []

  for _, idx in enumerate(idx_LT_not_null):
    _season = get_season_idx(season_array, idx)

    if _season != False:
      timeseries_idx_train.append(np.arange(_season[0], idx + 1))

  # timeseries_idx_train = np.array(timeseries_idx_train)

  return timeseries_idx_train

def split_XY(df, max_season_len, seasons):#, x_mean, x_std):
    X = []
    Y = []
    
    for i, season in enumerate(seasons):
        x = (df[features].loc[season, :]).to_numpy()

        x = np.concatenate((x, np.zeros((max_season_len - len(season), len(features)))), axis = 0)

        add_array = np.zeros((max_season_len - len(season), len(label)))
        add_array[:] = np.NaN

        y = df.loc[season,:][label].to_numpy()
        # print(f'y: {y.shape}, add_arr: {add_array.shape}')
        y = np.concatenate((y, add_array), axis=0)

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    
    # norm_features_idx = np.arange(0, x_mean.shape[0])
        
    # x[:, :, norm_features_idx]  = (x[:, :, norm_features_idx] - x_mean) / x_std # normalize
    
    return X, Y

def create_xy(df, timeseries_idx, max_length):
    x = []
    y = []
    actual_seq_lenghts = []
    for i, r_idx in enumerate(timeseries_idx):
        try:
          # print(f"ridx: {r_idx[-1]}")
          _x = df[features].loc[r_idx, :].to_numpy()
          # print(f'label: {df[label].loc[r_idx[-1]]}')
          _y = df[label].loc[r_idx[-1]]
          actual_seq_lenghts.append(_x.shape[0])
          pad_length = max_length - _x.shape[0]
    
          padding = np.zeros(shape=(pad_length, _x.shape[-1]))
          _x = np.append(_x, padding, axis=0)
          
          x.append(_x)
          y.append(_y)
        except KeyError:
          continue
        
    return np.array(x), np.array(y), actual_seq_lenghts

def get_mean_std(df, features):
    mean = []
    std = []
    for feature in features:
        season_npy = df[feature].to_numpy()
        idx = np.where(~np.isnan(season_npy))
        mean.append(np.mean(season_npy[idx]))
        std.append(np.std(season_npy[idx]))
    mean = np.array(mean)
    std = np.array(std)
    return mean, std