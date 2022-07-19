# Install PyPOTS first: pip install pypots
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae, cal_mse
from process_data import *
import pickle
# Data preprocessing. Tedious, but PyPOTS can help.
# data = load_specific_dataset('physionet_2012')  # For datasets in PyPOTS database, PyPOTS will automatically download and extract it.

df = pd.read_csv('ColdHardiness_Grape_Merlot_2.csv')
modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)#False, is_year=True)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)#False, is_year=True)
train_season_df = season_df.drop(season_array[-1], axis=0)
train_season_df = train_season_df.drop(season_array[-2], axis=0)
mean, std = get_mean_std(season_df, features)

X, Y = split_XY(season_df, max_length, season_array, features)

num_samples = len(season_array) - 2  #len(X['RecordID'].unique())

X = X[:-2]
Y = Y[:-2]

for i in range(X.shape[0]):
    X[i] = (X[i] - mean)/std
print(f"X: {X.shape}")
# X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
# Model training. This is PyPOTS showtime. 
saits = SAITS(n_steps=252, n_features=len(features), n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=1000, patience=100)
saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.

filename = 'model_saits_e1000_13.model'
pickle.dump(saits, open(filename, 'wb'))

imputation = saits.impute(X)  # impute the originally-missing values and artificially-missing values
mse = cal_mse(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(f"MSE: {mse}")