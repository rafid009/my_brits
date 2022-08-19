import pandas as pd
import numpy as np
# from evaluate_imputations import get_minimum_missing_season
from process_data import *
df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')

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
    'ETR', # ???
    'LTE50'
]

print(f'Feature length: {len(features)}')

# df_not_null = df[features].dropna()
# print(f"not null instances: {len(df_not_null)}")

modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#, is_year=True)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#, is_year=True)

print(f"season size: {season_df.shape}, max_length: {max_length}")

print(f"Season 2020-2021:\n{season_df.loc[season_array[-2], features].isna().sum()}\n")
print(f"Season 2021-2022:\n{season_df.loc[season_array[-1], features].isna().sum()}")
# X, Y = split_XY(season_df, max_length, season_array)
# print(f"X: {X.shape}, Y: {Y.shape}")
# season_not_null = season_df[features].dropna()
# print(f"not null instances in seasons: {len(df_not_null)}")
# print(f'missing: {season_df[features].isna().sum()}')

# print(f"\n\nSeasons: {len(season_array)}\n")
# for feature in features:
#     min_season, season_idx = get_minimum_missing_season(season_df, feature, season_array)
#     print(f"feature: {feature}\tseason: {season_idx}")

# print(f"\n\nMissing for the last season:\n")
# print(f"last season: {np.isnan(X[-1])}")
# print(f"season array: {season_array[-1]}")
