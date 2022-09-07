"""
PyTorch SAITS model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.data.base import BaseDataset
from pypots.data.dataset_for_mit import DatasetForMIT
from pypots.data.integration import mcar, masked_fill
from pypots.imputation.base import BaseNNImputer
from pypots.imputation.transformer import EncoderLayer, PositionalEncoding
from pypots.utils.metrics import cal_mae


class _SAITS(nn.Module):
    def __init__(self, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 diagonal_attention_mask=True, ORT_weight=1, MIT_weight=1, k=2):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.k = k

        self.layer_stack_for_first_block = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        self.layer_stack_for_second_block = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, inputs, k=-1):
        X, masks = inputs['X'], inputs['missing_mask']

        X_tilde_1 = None
        X_tilde_2 = None
        X_prime = X
        X_tildes = []
        attn_weights = None
        combining_weights = []
        if k == -1:
            k = self.k
        print(f"k: {k}")
        for i in range(k):
            input_X = torch.cat([X_prime, masks], dim=2)
            if i == 0:
                input_X = self.embedding_1(input_X) 
            else:
                input_X = self.embedding_2(input_X)
                
            if i == (k - 1):
                enc_output = self.position_enc(input_X)
            else:
                enc_output = self.dropout(self.position_enc(input_X)) 

            for encoder_layer in self.layer_stack_for_first_block:
                enc_output, attn_weights = encoder_layer(enc_output)

            if i == 0:
                X_tilde_1 = self.reduce_dim_z(enc_output)
                X_prime = masks * X_prime + (1 - masks) * X_tilde_1
            else:
                # enc_output_1 = F.relu(self.reduce_dim_z(enc_output))
                enc_output = F.relu(self.reduce_dim_beta(enc_output))
                # enc_output = enc_output_1 + enc_output_2
                X_tildes.append(self.reduce_dim_gamma(enc_output))
                X_prime = masks * X_prime + (1 - masks) * X_tildes[-1]
            attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
            if len(attn_weights.shape) == 4:
                # if having more than 1 head, then average attention weights from all heads
                attn_weights = torch.transpose(attn_weights, 1, 3)
                attn_weights = attn_weights.mean(dim=3)
                attn_weights = torch.transpose(attn_weights, 1, 2)

            combining_weights.append(torch.sigmoid(
                self.weight_combine(torch.cat([masks, attn_weights], dim=2))
            ))
        # # first DMSA block
        # input_X_for_first = torch.cat([X, masks], dim=2)
        # input_X_for_first = self.embedding_1(input_X_for_first)
        # enc_output = self.dropout(self.position_enc(input_X_for_first))  # namely, term e in the math equation
        # for encoder_layer in self.layer_stack_for_first_block:
        #     enc_output, _ = encoder_layer(enc_output)

        # X_tilde_1 = self.reduce_dim_z(enc_output)
        # X_prime = masks * X + (1 - masks) * X_tilde_1

        # # second DMSA block
        # input_X_for_second = torch.cat([X_prime, masks], dim=2)
        # input_X_for_second = self.embedding_2(input_X_for_second)
        # enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        # for encoder_layer in self.layer_stack_for_second_block:
        #     enc_output, attn_weights = encoder_layer(enc_output)

        # X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        # attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        # if len(attn_weights.shape) == 4:
        #     # if having more than 1 head, then average attention weights from all heads
        #     attn_weights = torch.transpose(attn_weights, 1, 3)
        #     attn_weights = attn_weights.mean(dim=3)
        #     attn_weights = torch.transpose(attn_weights, 1, 2)

        # combining_weights = torch.sigmoid(
        #     self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        # )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        # X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1

        X_tilde_final = 0
        for i in range(len(X_tildes)):
            X_tilde_final += combining_weights[i] * X_tildes[i]
        X_c = masks * X + (1 - masks) * X_tilde_final#3  # replace non-missing part with original data
        X_tildes.append(X_tilde_final)
        return X_c, X_tildes
        # return X_c, [X_tildes[0], X_tildes[-1], X_tilde_final]#3]

    def forward(self, inputs, k=-1):
        X, masks = inputs['X'], inputs['missing_mask']
        reconstruction_loss = 0
        # imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        # reconstruction_loss += cal_mae(X_tilde_1, X, masks)
        # reconstruction_loss += cal_mae(X_tilde_2, X, masks)

        # final_reconstruction_MAE = cal_mae(X_tilde_3, X, masks)
        # reconstruction_loss += final_reconstruction_MAE
        # reconstruction_loss /= 3


        imputed_data, X_finals = self.impute(inputs, k)
        total_count = 0
        for X_tilde in X_finals:
            reconstruction_loss += cal_mae(X_tilde, X, masks)
            total_count += 1
        reconstruction_loss /= total_count 

        

        # have to cal imputation loss in the val stage; no need to cal imputation loss here in the tests stage
        # imputation_loss = cal_mae(X_tilde_3, inputs['X_intact'], inputs['indicating_mask'])

        imputation_loss = cal_mae(X_finals[-1], inputs['X_intact'], inputs['indicating_mask'])

        loss = self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_loss

        return {
            'imputed_data': imputed_data,
            'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_loss,
            'loss': loss
        }


class SAITS(BaseNNImputer):
    def __init__(self,
                 n_steps,
                 n_features,
                 n_layers,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout,
                 diagonal_attention_mask=True,
                 ORT_weight=1,
                 MIT_weight=1,
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None,
                 k=2):
        super().__init__(learning_rate, epochs, patience, batch_size, weight_decay, device)

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.model = _SAITS(self.n_layers, self.n_steps, self.n_features, self.d_model, self.d_inner, self.n_head,
                            self.d_k, self.d_v, self.dropout, self.diagonal_attention_mask,
                            self.ORT_weight, self.MIT_weight, k=k)
        self.model = self.model.to(self.device)
        self._print_model_size()

    def fit(self, train_X, val_X=None):
        train_X = self.check_input(self.n_steps, self.n_features, train_X)
        if val_X is not None:
            val_X = self.check_input(self.n_steps, self.n_features, val_X)

        training_set = DatasetForMIT(train_X)
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        if val_X is None:
            self._train_model(training_loader)
        else:
            val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(val_X, 0.2)
            val_X = masked_fill(val_X, 1 - val_X_missing_mask, torch.nan)
            val_set = DatasetForMIT(val_X)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self._train_model(training_loader, val_loader, val_X_intact, val_X_indicating_mask)

        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

    def assemble_input_data(self, data):
        """ Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """
        indices, X_intact, X, missing_mask, indicating_mask = data

        inputs = {
            'X': X,
            'X_intact': X_intact,
            'missing_mask': missing_mask,
            'indicating_mask': indicating_mask
        }

        return inputs

    def impute(self, X, k=-1):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = {'X': data[1], 'missing_mask': data[2]}
                imputed_data, _ = self.model.impute(inputs, k)
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
