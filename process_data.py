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
    'MSLP_HPA', # barrometric pressure # diff from zengxian
    'ETO', # evaporation of soil water lost to atmosphere
    'ETR' # ???
]
label = 'LT50'

def preprocess_missing_values(df):
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

  start_idx = df[df['DATE'] == '2007-07-21'].index.tolist()[0]
  end_idx = df[df['DATE'] == '2007-12-30'].index.tolist()[0]

  start_idx += 1
  for i in range(start_idx, end_idx + 1):
    modified_df.at[i, 'MIN_DEWPT'] = np.nan
    modified_df.at[i, 'MAX_DEWPT'] = np.nan
    modified_df.at[i, 'AVG_DEWPT'] = np.nan
  dormant_seasons = modified_df[modified_df["DORMANT_SEASON"] == 1].index.tolist()
  return modified_df, dormant_seasons


def get_seasons_data(modified_df, dormant_seasons):
  seasons = []
  last_x = 0
  idx = -1
  season_max_length = 0
  for x in dormant_seasons: # modified_df[modified_df["DORMANT_SEASON"] == 1].index.tolist():
      if x - last_x > 1:
          seasons.append([])
          if idx > -1:
              season_max_length = max(season_max_length, len(seasons[idx]))
          idx += 1
      seasons[idx].append(x)
      last_x = x
  season_max_length = max(season_max_length, len(seasons[idx]))

  # season_idx = []
  # for season in seasons:
  #   season_idx.extend(season)

  season_df = pd.DataFrame(modified_df.iloc[dormant_seasons], copy=True)
  return season_df, seasons, season_max_length

def get_non_null_LT(df):
  # get all indexes where LT10 is not null
#   idx_LT10_not_null = df[df['LT10'].notnull()].index.tolist()
  idx_LT50_not_null = df[df['LT50'].notnull()].index.tolist()
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