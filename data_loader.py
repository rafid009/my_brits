import os
import time

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, filename='./json/json'):
        super(MySet, self).__init__()
        self.content = open(filename).readlines()

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        # print(f"rec type: {type(rec)}")
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    # print(f"forward: {forward}")
    backward = list(map(lambda x: x['backward'], recs))
    # print(f"backward: {backward.__next__()['masks']}")
    def to_tensor_dict(recs):
        # print(f"recs: {list(recs['masks'])}")
        values = list(map(lambda r: r['values'], recs))
        # print(f"r[values]: {value_list}")
        values = torch.FloatTensor(values)
        # print(f"values: {values}")
        masks = list(map(lambda r: r['masks'], recs))
        
        masks = torch.FloatTensor(masks)
        # print(f"masks: {masks.shape}")
        deltas = list(map(lambda r: r['deltas'], recs))
        deltas = torch.FloatTensor(deltas)
        # print('deltas: ', deltas.shape)

        evals = list(map(lambda r: r['evals'], recs))
        evals = torch.FloatTensor(evals)
        eval_masks = list(map(lambda r: r['eval_masks'], recs))
        eval_masks = torch.FloatTensor(eval_masks)
        forwards = list(map(lambda r: r['forwards'], recs))
        forwards = torch.FloatTensor(forwards)

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    labels = list(map(lambda x: x['label'], recs))
    # print('loader label: ', labels)
    ret_dict['labels'] = torch.FloatTensor(labels)
    # print('loader rec label: ', ret_dict['labels'].shape)
    is_trains = list(map(lambda x: x['is_train'], recs))
    ret_dict['is_train'] = torch.FloatTensor(is_trains)

    return ret_dict

def get_loader(batch_size = 64, shuffle = True, filename='./json/json'):
    data_set = MySet(filename)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 0, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
