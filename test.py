import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from process_data import *
# from input_process import *
import json
import os
import torch
from models.brits import BRITSModel as BRITS
from models.brits_i import BRITSModel as BRITS_I
import utils
import data_loader
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

RNN_HID_SIZE = 64
IMPUTE_WEIGHT = 0.3
LABEL_WEIGHT = 1

df = pd.read_csv('FrostMitigation_Merlot.csv')

folder = './json/'
if not os.path.exists(folder):
    os.makedirs(folder)
fs = open(folder+'json', 'w')

real_values = None

mean = []
std = []

features_impute = [features.index('MEAN_AT'), features.index('AVG_REL_HUMIDITY')]

def _graph_bar_diff_multi(GT_values, result_dict, title, x, xlabel, ylabel, i, drop_linear=False):
    plot_dict = {}
    plt.figure(figsize=(16,9))
    for key, value in result_dict.items():
      plot_dict[key] = np.abs(GT_values) - np.abs(value)
    # ind = np.arange(prediction.shape[0])
    x = np.array(x)
    width = 0.3
    pos = 0
    remove_keys = ['GT']
    if drop_linear:
      remove_keys.append('LinearInterp')
    for key, value in plot_dict.items():
        if key not in remove_keys:
          plt.bar(x + pos + 1, value, width, label = key)
          pos += width

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis([0, 100, -2, 3])

    plt.legend(loc='best')
    plt.savefig(f'diff_imgs/result-diff-{i}.png', dpi=300)
    plt.close()
    # plt.show()
    return

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(masks.shape[0]):
        if h == 0:
            deltas.append(np.ones(len(features)))
        else:
            deltas.append(np.ones(len(features)) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def draw_data_trend(df_t, df_c, f, interp_name, i):
    plt.figure(figsize=(32,9))
    plt.title(f)
    ax = plt.subplot(121)
    ax.set_title(f+' original')
    plt.plot(np.arange(df_t.shape[0]), df_t)
    ax = plt.subplot(122)
    ax.set_title(f+' imputed by '+interp_name)
    plt.plot(np.arange(df_c.shape[0]), df_c)
    plt.savefig(f"subplots/{f}-{interp_name}-{i}.png", dpi=300)
    plt.close()

def parse_rec(values, masks, evals, eval_masks, dir_):
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

    return rec


def parse_id(x, y):
    # data = pd.read_csv('./raw/{}.txt'.format(id_))
    # accumulate the records within one hour
    # data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))
    
    evals = x
    # print('x: ', x)

    # merge all the metrics within one hour
    # for h in range(48):
    #     evals.append(parse_data(data[data['Time'] == h]))
    evals = (evals - mean) / std
    # print('eval: ', evals)
    # print('eval shape: ', evals.shape)
    shp = evals.shape
    
    idx1 = np.where(~np.isnan(evals[:,features_impute[0]]))[0]
    idx2 = np.where(~np.isnan(evals[:,features_impute[1]]))[0]
    idx1 = idx1 * len(features) + features_impute[0]
    idx2 = idx2 * len(features) + features_impute[1]
    # exit()

    evals = evals.reshape(-1)
    # randomly eliminate 10% values as the imputation ground-truth
    # print('not null: ',np.where(~np.isnan(evals)))
    indices = np.concatenate((idx1, idx2)).tolist()
    indices = np.random.choice(indices, len(indices) // 10)
    
    global real_values
    real_values = evals[indices]

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)

    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)
    label = y.tolist() #out.loc[int(id_)]
    # print(f'rec y: {list(y)}')
    rec = {'label': label}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')
    # for key in rec.keys():
    #     print(f"{key}: {type(rec[key])}")# and {rec[key].shape}")
    rec = json.dumps(rec)

    fs.write(rec + '\n')

modified_df, dormant_seasons = preprocess_missing_values(df)
season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons)

X, Y = split_XY(season_df, max_length, season_array)



for feature in features:
    season_npy = season_df[feature].to_numpy()
    idx = np.where(~np.isnan(season_npy))
    mean.append(np.mean(season_npy[idx]))
    std.append(np.std(season_npy[idx]))
mean = np.array(mean)
std = np.array(std)

for i in range(X.shape[0]):
    parse_id(X[i], Y[i])
fs.close()


knn_impute = KNNImputer(n_neighbors=7, weights='distance')

mice_impute = IterativeImputer(random_state=0)

model_brits = BRITS(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

model_brits_I = BRITS_I(rnn_hid_size=RNN_HID_SIZE, impute_weight=IMPUTE_WEIGHT, label_weight=LABEL_WEIGHT)

if os.path.exists('./model_brits_notI.model'):
    model_brits.load_state_dict(torch.load('./model_brits_notI.model'))

if os.path.exists('./model_brits.model'):
    model_brits_I = torch.load('./model_brits.model')

if torch.cuda.is_available():
    model_brits = model_brits.cuda()
    model_brits_I = model_brits_I.cuda()


model_brits.eval()
model_brits_I.eval()

labels = []
preds = []

evals = []
imputations_brits = []

imputations_brits_I = []

imputations_knn = []

imputations_mice = []

save_impute = []
save_label = []

val_iter = data_loader.get_loader(batch_size=1)#X.shape[0])

knn_impute.fit(season_df[features])
mice_impute.fit(season_df[features])

season_eval = []
season_impute_brits = []
season_impute_brits_i = []
season_impute_knn = []
season_impute_mice = []
season = 1
if not os.path.isdir('plots'):
    os.makedirs('plots')
if not os.path.isdir('subplots'):
    os.makedirs('subplots')
for idx, data in enumerate(val_iter):
    data = utils.to_var(data)
    ret = model_brits.run_on_batch(data, None)

    # save the imputation results which is used to test the improvement of traditional methods with imputed values
    # save_impute.append(ret['imputations'].data.cpu().numpy())
    # save_label.append(ret['labels'].data.cpu().numpy())

    eval_masks = ret['eval_masks'].data.cpu().numpy()
    eval_ = ret['evals'].data.cpu().numpy()
    print(f"eval:\n{eval_.shape}")
    imputation_brits = ret['imputations'].data.cpu().numpy()

    # evals += eval_[np.where(eval_masks == 1)].tolist()
    # imputations_brits += imputation[np.where(eval_masks == 1)].tolist()



    ret = model_brits_I.run_on_batch(data, None)

    # eval_masks = ret['eval_masks'].data.cpu().numpy()
    # eval_ = ret['evals'].data.cpu().numpy()
    imputation_brits_I = ret['imputations'].data.cpu().numpy()

    season_eval.extend(np.squeeze(eval_))
    season_impute_brits.extend(np.squeeze(imputation_brits))
    season_impute_brits_i.extend(np.squeeze(imputation_brits_I))

    # ret_eval = copy.deepcopy(np.squeeze(ret['evals']))
    # ret_eval[ret_eval == 0] = np.nan
    # imputed_knn = knn_impute.transform(ret_eval)
    # imputed_mice = mice_impute.transform(ret_eval)

    # season_impute_knn.extend(imputed_knn)
    # season_impute_mice.extend(imputed_mice)

    # imputations_brits_I += imputation[np.where(eval_masks == 1)].tolist()
    # for feature_idx in features_impute:
    #     # print(eval_[:, feature_idx].shape)
    #     evals = np.squeeze(eval_[:, :, feature_idx])
    #     brits = np.squeeze(imputation_brits[:, :, feature_idx])
    #     brits_i = np.squeeze(imputation_brits_I[:, :, feature_idx])
    #     x_range = np.arange(len(evals))
    #     # print('x range: ', x_range)
    #     plt.figure(figsize=(16,9))
    #     plt.plot(x_range, brits, 'r', label='BRITS', scaley='log')
    #     plt.plot(x_range, brits_i, 'b', label='BRITS_I', scaley='log')
    #     plt.plot(x_range, evals, 'g', label='Ground Truth', scaley='log')
    #     plt.title(f'Ground Truth vs The imputations of {features[feature_idx]}')
    #     plt.xlabel(f'Days of season  no. {season}')
    #     plt.ylabel(f'{features[feature_idx]}')
    #     plt.legend()
    #     plt.savefig(f'plots/gt-imputed-{features[feature_idx]}-{season}.png', dpi=300)
    #     plt.close()
    
    season += 1
season_eval = np.array(season_eval)
print(f"eval seasons: {season_eval.shape}")
season_impute_brits = np.array(season_impute_brits)
season_impute_brits_i = np.array(season_impute_brits_i)
# season_impute_knn = np.array(season_impute_knn)
# season_impute_mice = np.array(season_impute_mice)

for feature_idx in features_impute:
    start = 0
    size = len(season_impute_brits)
    limit = 252
    iters = int(np.ceil(size / limit))
    print('size: ', size, 'iters: ', iters)
    for i in range(iters):
        end = start + limit
        print(f"start: {start}, end: {end}")
        if size < end:
            end = size
        # draw_data_trend(season_eval[start:end,feature_idx], season_impute_brits[start:end,feature_idx], features[feature_idx], 'BRITS', i)
        # draw_data_trend(season_eval[start:end,feature_idx], season_impute_brits_i[start:end,feature_idx], features[feature_idx], 'BRITS_I', i)
        # draw_data_trend(season_eval[start:end,feature_idx], season_impute_knn[start:end,feature_idx], features[feature_idx], 'KNN', i)
        # draw_data_trend(season_eval[start:end,feature_idx], season_impute_mice[start:end,feature_idx], features[feature_idx], 'MICE', i)
        
        x_range = np.arange(len(season_eval[start:end,feature_idx]))
        plt.figure(figsize=(16,9))
        plt.plot(x_range, season_impute_brits[start:end,feature_idx], 'r', label='BRITS', scaley='log')
        plt.plot(x_range, season_impute_brits_i[start:end,feature_idx], 'b', label='BRITS_I', scaley='log')
        # plt.plot(x_range, season_impute_knn[start:end,feature_idx], 'c', label='KNN', scaley='log')
        # plt.plot(x_range, season_impute_mice[start:end,feature_idx], 'm', label='MICE', scaley='log')
        plt.plot(x_range, season_eval[start:end,feature_idx], 'g', label='Ground Truth', scaley='log')
        plt.title(f'Ground Truth vs The imputations of {features[feature_idx]}')
        plt.xlabel(f'Days of the season  no. {i}')
        plt.ylabel(f'{features[feature_idx]}')
        plt.legend()
        plt.savefig(f'plots/gt-imputed-{features[feature_idx]}-{i}.png', dpi=300)
        plt.close()
        start = end



# evals = np.array(evals)
# imputations_brits = np.array(imputations_brits)

# imputations_brits_I = np.array(imputations_brits_I)
# print("imp len: ", len(imputations_brits))
# result_dict = {
#     # 'knn': X_knn_imputed,
#     'brits': imputations_brits,
#     'brits_I': imputations_brits_I
# }

# limit = 100
# start = 0

# for i in range(int(len(imputations_brits)/limit)):
#     end = start + limit + 1
#     result_dict = {
#         'brits': imputations_brits[start:end],
#         'brits_I': imputations_brits_I[start:end]
#     }
#     _graph_bar_diff_multi(evals[start:end], result_dict, "Difference of the different imputations from GT", np.arange(len(imputations_brits[start:end])), "# of missing values from observed data", "imputed value difference form GT", i)
#     start = end

# start = 0

# for i in range(int(len(imputations_brits)/limit)):
#     end = start+limit+1
#     x_range = np.arange(len(imputations_brits[start:end]))
#     plt.figure(figsize=(16,9))
#     plt.plot(x_range, imputations_brits[start:end], 'r', label='BRITS')
#     plt.plot(x_range, imputations_brits_I[start:end], 'b', label='BRITS_I')
#     plt.plot(x_range, evals[start:end], 'g', label='Ground Truth')
#     plt.title('GT vs Bidirectional RNN based imputations')
#     plt.xlabel('# of missing values from observed data')
#     plt.ylabel('Actual/imputed values (normalized)')
#     plt.legend()
#     plt.savefig(f'plot_imgs/gt-imputed-{i}.png', dpi=300)
#     plt.close()
#     start = end