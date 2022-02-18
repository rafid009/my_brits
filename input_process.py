# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
from process_data import *
import json

# patient_ids = []

# for filename in os.listdir('./raw'):
#     # the patient data in PhysioNet contains 6-digits
#     match = re.search('\d{6}', filename)
#     if match:
#         id_ = match.group()
#         patient_ids.append(id_)

# out = pd.read_csv('./raw/Outcomes-a.txt').set_index('RecordID')['In-hospital_death']

# we select 35 attributes which contains enough non-values
attributes = features
# attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
#               'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
#               'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
#               'Creatinine', 'ALP']

# mean and std of 35 attributes
# mean = np.array([59.540976152469405, 86.72320413227443, 139.06972964987443, 2.8797765291788986, 58.13833409690321,
#                  147.4835678885565, 12.670222585415166, 7.490957887101613, 2.922874149659863, 394.8899400819931,
#                  141.4867570064675, 96.66380228136883, 37.07362841054398, 505.5576196473552, 2.906465787821709,
#                  23.118951553526724, 27.413004968675743, 19.64795551193981, 2.0277491155660416, 30.692432164676188,
#                  119.60137167841977, 0.5404785381886381, 4.135790642787733, 11.407767149315339, 156.51746031746032,
#                  119.15012244292181, 1.2004983498349853, 80.20321011673151, 7.127188940092161, 40.39875518672199,
#                  191.05877024038804, 116.1171573535279, 77.08923183026529, 1.5052390166989214, 116.77122488658458])

# std = np.array(
#     [13.01436781437145, 17.789923096504985, 5.185595006246348, 2.5287518090506755, 15.06074282896952, 85.96290370390257,
#      7.649058756791069, 8.384743923130074, 0.6515057685658769, 1201.033856726966, 67.62249645388543, 3.294112002091972,
#      1.5604879744921516, 1515.362517984297, 5.902070316876287, 4.707600932877377, 23.403743427107095, 5.50914416318306,
#      0.4220051299992514, 5.002058959758486, 23.730556355204214, 0.18634432509312762, 0.706337033602292,
#      3.967579823394297, 45.99491531484596, 21.97610723063014, 2.716532297586456, 16.232515568438338, 9.754483687298688,
#      9.062327978713556, 106.50939503021543, 170.65318497610315, 14.856134327604906, 1.6369529387005546,
#      133.96778334724377])
folder = './json/'
if not os.path.exists(folder):
    os.makedirs(folder)
fs = open(folder+'json', 'w')

# def to_time_bin(x):
#     h, m = map(int, x.split(':'))
#     return h


# def parse_data(x):
#     x = x.set_index('Parameter').to_dict()['Value']

#     values = []

#     for attr in attributes:
#         if x.has_key(attr):
#             values.append(x[attr])
#         else:
#             values.append(np.nan)
#     return values


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(48):
        if h == 0:
            deltas.append(np.ones(len(attributes)))
        else:
            deltas.append(np.ones(len(attributes)) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, seq_len, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).to_numpy()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()
    rec['seq_len'] = seq_len

    return rec


def parse_id(x, y, seq_len):
    # data = pd.read_csv('./raw/{}.txt'.format(id_))
    # accumulate the records within one hour
    # data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))
    
    evals = x
    print('x: ', x)
    print('x shape: ', x.shape)

    # merge all the metrics within one hour
    # for h in range(48):
    #     evals.append(parse_data(data[data['Time'] == h]))
    evals[:seq_len] = (evals[:seq_len] - mean) / std
    print('eval: ', evals)
    print('eval shape: ', evals.shape)
    shp = evals.shape

    evals = evals.reshape(-1)
    print(f"eval reshaped: {evals}")
    # randomly eliminate 10% values as the imputation ground-truth
    print('not null: ',np.where(~np.isnan(evals)))
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 10)
    

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    label = y #out.loc[int(id_)]

    rec = {'label': label}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, seq_len, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], seq_len, dir_='backward')
    
    rec = json.dumps(rec)

    fs.write(rec + '\n')


df = pd.read_csv('FrostMitigation_Merlot.csv')
modified_df, dormant_seasons = preprocess_missing_values(df)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons)
idx_LT_not_null = get_non_null_LT(season_df)
train_idx = get_train_idx(season_array, idx_LT_not_null)
X, Y, seq_length = create_xy(season_df, train_idx, max_length)
print(f"X: {X.shape}, Y: {Y.shape}, seq_len: {len(seq_length)}")

mean = []
std = []
for feature in attributes:
    season_npy = season_df[feature].to_numpy()
    idx = np.where(~np.isnan(season_npy))
    mean.append(np.mean(season_npy[idx]))
    std.append(np.std(season_npy[idx]))

print('season mean at: ',np.where(~np.isnan(season_npy)))

mean = np.array(mean) #np.mean(season_df[attributes].to_numpy(), axis=0)
std = np.array(std) #np.std(season_df[attributes].to_numpy(), axis=0)
np.save('mean.npy', mean)
np.save('std.npy', std)

for i in range(X.shape[0]):
    # print('X: ',X[i].shape)
    parse_id(X[i], Y[i], seq_length[i])

# for id_ in patient_ids:
#     print('Processing patient {}'.format(id_))
#     try:
#         parse_id(id_)
#     except Exception as e:
#         print(e)
#         continue

fs.close()

