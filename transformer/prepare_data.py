import copy
import pandas as pd
import numpy as np
from src.datasets.data_utils import *

data_folder = './transformer/data_dir'
df = pd.read_csv(f'{data_folder}/ColdHardiness_Grape_Merlot_2.csv')
modified_df, dormant_seasons = preprocess_missing_values(df, is_dormant=False, is_year=True)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, is_dormant=False, is_year=True)

season_df['season_id'] = 0

# train_season_df = season_df.drop(season_array[-1], axis=0)
# train_season_df = train_season_df.drop(season_array[-2], axis=0)
# indices = copy.deepcopy(season_array[-2])
# idx2 = copy.deepcopy()
# indices.extend()
train_season_df = season_df.copy()
# train_season_df = season_df.iloc[season_array]
print(train_season_df.columns.values)
for season_id in range(len(season_array)):
    # print(season_id)
    for idx in season_array[season_id]:
        # print(train_season_df.columns.loc('season_id'))
        train_season_df.loc[idx, 'season_id'] = season_id
print(f'Seasons: {len(season_array)}')
print(train_season_df.columns.values)
train_season_df.to_csv(f'{data_folder}/ColdHardiness_Grape_Merlot_test.csv', index=False)
np.save(f'{data_folder}/seasons.npy', season_array)