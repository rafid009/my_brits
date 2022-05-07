import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from process_data import *
import os
df = pd.read_csv('ColdHardiness_Grape_Merlot.csv')

def draw_data_plot(results, f, season_idx):
    if not os.path.isdir('subplots/' + f):
        os.makedirs('subplots/' + f)
    plt.figure(figsize=(32,18))
    plt.title(f"For feature = {f} and season = {season_idx}", fontsize=24)
    
    plt.xlabel('Days of a Season', fontsize=18)
    plt.ylabel('Values', fontsize=18)
    plt.tight_layout()
    plt.savefig(f"subplots/{f}/{f}-imputations-season-{season_idx}.png", dpi=300)
    plt.close()

temp_features = [
    'MEAN_AT', # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
    'MIN_AT',
    'AVG_AT', # average temp is AgWeather Network
    'MAX_AT',
    'MIN_REL_HUM',
    'AVG_REL_HUM',
    'MAX_REL_HUM',
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

def rounded_rate(df_feature, total):
    return np.round((df_feature.isna().sum() * 100) / total, 2)

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center', fontsize=50)


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
total = len(season_df)
# print(f"mean at: {season_df['MEAN_AT'].isna().sum()}, {(season_df['MEAN_AT'].isna().sum()*100)/total}")
missing_percentage = []
for feature in features:
    rate = rounded_rate(season_df[feature], total)
    missing_percentage.append(rate)

plt.figure(figsize = (66, 38))

plt.bar(temp_features, missing_percentage, align='center', width=0.5)

addlabels(temp_features, missing_percentage)

plt.xlabel("Features", fontsize=40)
plt.ylabel('Percent of Missing Values (%)', fontsize=40)
plt.title('Rate of Missing Values in Different Features', fontsize=48)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tight_layout()
plt.savefig('missing.png', dpi=300)
plt.close()

# given_features = ['MEAN_AT', 'AVG_REL_HUMIDITY']


# for feature in given_features:
#     mse_df = pd.read_csv('imputation_results/'+feature+'/'+feature+'_results_impute.csv')
#     L = [i for i in range(len(mse_df))]
#     plt.figure(figsize=(16,9))
#     plt.plot(L, mse_df['BRITS'], 'tab:orange', label='BRITS', marker='o')
#     plt.plot(L, mse_df['BRITS_I'], 'tab:blue', label='BRITS_I', marker='o')
#     plt.title(f'Length of missing values vs Imputation MSE for feature = {feature}, season=2020-2021', fontsize=20)
#     plt.xlabel(f'Length of contiguous missing values', fontsize=16)
#     plt.ylabel(f'MSE', fontsize=16)
#     plt.legend()
#     plt.savefig(f'plots/{feature}/L-vs-MSE-BRITS-comp-models{feature}-{len(L)}.png', dpi=300)
#     plt.close()