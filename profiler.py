import pandas as pd
import numpy as np
from process_data import *
df = pd.read_csv('ColdHardiness_Grape_Merlot.csv')

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

print(f'Feature length: {len(features)}')

df_not_null = df[features].dropna()
print(f"not null instances: {len(df_not_null)}")

modified_df, dormant_seasons = preprocess_missing_values(df)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons)
print(f"season size: {season_df.shape}, max_length: {max_length}")

X, Y = split_XY(season_df, max_length, season_array)
print(f"X: {X.shape}, Y: {Y.shape}")
season_not_null = season_df[features].dropna()
print(f"not null instances in seasons: {len(df_not_null)}")
print(f'missing: {season_df[features].isna().sum()}')