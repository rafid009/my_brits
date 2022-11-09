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
import numpy as np
from pypots.data.base import BaseDataset
from pypots.data.dataset_for_mit import DatasetForMIT
from pypots.data.integration import mcar, masked_fill
from pypots.imputation.base import BaseNNImputer
from pypots.imputation.transformer import EncoderLayer, PositionalEncoding
from pypots.utils.metrics import cal_mae, cal_mse


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        print(f"diffusion_step: {diffusion_step.shape}\nemb: {self.embedding.shape}")
        x = self.embedding[diffusion_step]
        print(f"x after emb: {x.shape}")
        x = self.projection1(x)
        x = F.silu(x)
        print(f"x after proj1: {x.shape}")
        x = self.projection2(x)
        x = F.silu(x)
        print(f"x after proj2: {x.shape}")
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        # print(f"steps: {steps.shape}")
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        # print(f"frequency: {frequencies.shape}")
        table = steps * frequencies  # (T,dim)
        # print(f"table 1: {table.shape}")
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        # print(f"table 2: {table.shape}")
        return table

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffSAITS(nn.Module):
    def __init__(self, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 diagonal_attention_mask=True, ORT_weight=1, MIT_weight=1, diff_t_emb_dim=256, diff_steps=50, time_strategy='cat'):
        super().__init__()
        self.n_layers = n_layers
        self.time_strategy = time_strategy

        if self.time_strategy == 'cat':
            actual_d_feature = d_feature * 2 + int(diff_t_emb_dim)
        else:
            actual_d_feature = d_feature * 2

        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_t_emb_dim, projection_dim=int(diff_t_emb_dim))
        
        if self.time_strategy == 'cat':
            self.diffusion_projection1 = nn.Linear(1, d_time)
        else:
            self.diffusion_projection1 = nn.Linear(int(diff_t_emb_dim), d_time)
        
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

    def impute(self, inputs, time_step=None):
        X, masks = inputs['X'], inputs['missing_mask']
        diffusion_emb = self.diffusion_embedding(time_step)

        print(f"X: {X.shape}, masks: {masks.shape}, diffusion_emb: {diffusion_emb.shape}")
        diffusion_emb = self.diffusion_projection1(diffusion_emb)
        diffusion_emb = diffusion_emb.unsqueeze(-1)
        print(f"diffusion_emb after proj1: {diffusion_emb.shape}")
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2)
        print(f"X after mask concat: {input_X_for_first.shape}")
        input_X_for_first = self.embedding_1(input_X_for_first)
        print(f"input_X_for_first: {input_X_for_first.shape}")
        enc_output = self.dropout(self.position_enc(input_X_for_first))  # namely, term e in the math equation
        print(f"enc_output: {enc_output.shape}")
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output += diffusion_emb
            enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2)
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output += diffusion_emb
            enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]


    def impute1(self, inputs, time_step=None):
        X, masks = inputs['X'], inputs['missing_mask']
        diffusion_emb = self.diffusion_embedding(time_step)
        
        if self.time_strategy == 'cat':
            diffusion_emb = diffusion_emb.unsqueeze(-1)
            diffusion_emb = self.diffusion_projection1(diffusion_emb)
            # print(f"diff proj 1: {diffusion_emb.shape}")
            diffusion_emb = torch.transpose(diffusion_emb, -1, -2)
            # first DMSA block
            # print(f"X: {X.shape}, masks: {masks.shape}, diffusion_emb: {diffusion_emb.shape}")
            input_X_for_first = torch.cat([X, masks, diffusion_emb], dim=2)
        else:
            print(f"X: {X.shape}, masks: {masks.shape}, diffusion_emb: {diffusion_emb.shape}")
            diffusion_emb = self.diffusion_projection1(diffusion_emb)
            print(f"diffusion_emb: {diffusion_emb.shape}")
            diffusion_emb = diffusion_emb.unsqueeze(-1)
            input_X_for_first = torch.cat([X, masks], dim=2)
            # input_X_for_first += diffusion_emb
            print(f"input X first: {input_X_for_first.shape}")

        input_X_for_first = self.embedding_1(input_X_for_first)
        print(f"input_X_for_first: {input_X_for_first.shape}")
        enc_output = self.dropout(self.position_enc(input_X_for_first))  # namely, term e in the math equation
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, _ = encoder_layer(enc_output)
            enc_output = F.silu(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        if self.time_strategy == 'cat':
            input_X_for_second = torch.cat([X_prime, masks, diffusion_emb], dim=2)
        else:
            input_X_for_second = torch.cat([X_prime, masks], dim=2)
            # input_X_for_second += diffusion_emb
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, attn_weights = encoder_layer(enc_output)
            enc_output = F.silu(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = torch.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return X_tilde_3 # X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, time_step=None):
        # X, masks = inputs['X'], inputs['missing_mask']
        # reconstruction_loss = 0
        # imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs, time_emb)
        predicted_mean, X_finals = self.impute(inputs, time_step)

        # reconstruction_loss += cal_mae(X_tilde_1, X, masks)
        # reconstruction_loss += cal_mae(X_tilde_2, X, masks)
        # final_reconstruction_MAE = cal_mae(X_tilde_3, X, masks)
        # reconstruction_loss += final_reconstruction_MAE
        # reconstruction_loss /= 3

        # # have to cal imputation loss in the val stage; no need to cal imputation loss here in the tests stage
        # imputation_loss = cal_mae(X_tilde_3, inputs['X_intact'], inputs['indicating_mask'])

        # loss = self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_loss
        return predicted_mean, X_finals
        # return {
        #     'imputed_data': imputed_data,
        #     'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_loss,
        #     'loss': loss
        # }


# class SAITS(BaseNNImputer):
#     def __init__(self,
#                  n_steps,
#                  n_features,
#                  n_layers,
#                  d_model,
#                  d_inner,
#                  n_head,
#                  d_k,
#                  d_v,
#                  dropout,
#                  diagonal_attention_mask=True,
#                  ORT_weight=1,
#                  MIT_weight=1,
#                  learning_rate=1e-3,
#                  epochs=100,
#                  patience=10,
#                  batch_size=32,
#                  weight_decay=1e-5,
#                  device=None):
#         super().__init__(learning_rate, epochs, patience, batch_size, weight_decay, device)

#         self.n_steps = n_steps
#         self.n_features = n_features
#         # model hype-parameters
#         self.n_layers = n_layers
#         self.d_model = d_model
#         self.d_inner = d_inner
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#         self.dropout = dropout
#         self.diagonal_attention_mask = diagonal_attention_mask
#         self.ORT_weight = ORT_weight
#         self.MIT_weight = MIT_weight

#         self.model = _SAITS(self.n_layers, self.n_steps, self.n_features, self.d_model, self.d_inner, self.n_head,
#                             self.d_k, self.d_v, self.dropout, self.diagonal_attention_mask,
#                             self.ORT_weight, self.MIT_weight)
#         self.model = self.model.to(self.device)
#         self._print_model_size()

#     def _train_model(self, training_loader, val_loader=None, val_X_intact=None, val_indicating_mask=None):
#         self.optimizer = torch.optim.Adam(self.model.parameters(),
#                                           lr=self.lr,
#                                           weight_decay=self.weight_decay)

#         # each training starts from the very beginning, so reset the loss and model dict here
#         self.best_loss = float('inf')
#         self.best_model_dict = None

#         try:
#             for epoch in range(self.epochs):
#                 self.model.train()
#                 epoch_train_loss_collector = []
#                 for idx, data in enumerate(training_loader):
#                     inputs = self.assemble_input_data(data)
#                     self.optimizer.zero_grad()
#                     results = self.model.forward(inputs)
#                     results['loss'].backward()
#                     self.optimizer.step()
#                     epoch_train_loss_collector.append(results['loss'].item())

#                 mean_train_loss = np.mean(epoch_train_loss_collector)  # mean training loss of the current epoch
#                 self.logger['training_loss'].append(mean_train_loss)

#                 if val_loader is not None:
#                     self.model.eval()
#                     imputation_collector = []
#                     with torch.no_grad():
#                         for idx, data in enumerate(val_loader):
#                             inputs = self.assemble_input_data(data)
#                             results = self.model.forward(inputs)
#                             imputation_collector.append(results['imputed_data'])

#                     imputation_collector = torch.cat(imputation_collector)
#                     imputation_collector = imputation_collector

#                     mean_val_loss = cal_mae(imputation_collector, val_X_intact, val_indicating_mask)
#                     self.logger['validating_loss'].append(mean_val_loss)
#                     print(f'epoch {epoch}: training loss {mean_train_loss:.4f}, validating loss {mean_val_loss:.4f}')
#                     mean_loss = mean_val_loss
#                 else:
#                     print(f'epoch {epoch}: training loss {mean_train_loss:.4f}')
#                     mean_loss = mean_train_loss

#                 if mean_loss < self.best_loss:
#                     self.best_loss = mean_loss
#                     self.best_model_dict = self.model.state_dict()
#                     self.patience = self.original_patience
#                 else:
#                     self.patience -= 1

#                 # if os.getenv('enable_nni', False):
#                 #     nni.report_intermediate_result(mean_loss)
#                 #     if epoch == self.epochs - 1 or self.patience == 0:
#                 #         nni.report_final_result(self.best_loss)

#                 if self.patience == 0:
#                     print('Exceeded the training patience. Terminating the training procedure...')
#                     break

#         except Exception as e:
#             print(f'Exception: {e}')
#             if self.best_model_dict is None:
#                 raise RuntimeError('Training got interrupted. Model was not get trained. Please try fit() again.')
#             else:
#                 RuntimeWarning('Training got interrupted. '
#                                'Model will load the best parameters so far for testing. '
#                                "If you don't want it, please try fit() again.")

#         if np.equal(self.best_loss, float('inf')):
#             raise ValueError('Something is wrong. best_loss is Nan after training.')

#         print('Finished training.')

#     def check_input(self, expected_n_steps, expected_n_features, X, y=None, out_dtype='tensor'):
#         """ Check value type and shape of input X and y

#         Parameters
#         ----------
#         expected_n_steps : int
#             Number of time steps of input time series (X) that the model expects.
#             This value is the same with the argument `n_steps` used to initialize the model.

#         expected_n_features : int
#             Number of feature dimensions of input time series (X) that the model expects.
#             This value is the same with the argument `n_features` used to initialize the model.

#         X : array-like,
#             Time-series data that must have a shape like [n_samples, expected_n_steps, expected_n_features].

#         y : array-like, default=None
#             Labels of time-series samples (X) that must have a shape like [n_samples] or [n_samples, n_classes].

#         out_dtype : str, in ['tensor', 'ndarray'], default='tensor'
#             Data type of the output, should be np.ndarray or torch.Tensor

#         Returns
#         -------
#         X : tensor

#         y : tensor
#         """
#         assert out_dtype in ['tensor', 'ndarray'], f'out_dtype should be "tensor" or "ndarray", but got {out_dtype}'
#         is_list = isinstance(X, list)
#         is_array = isinstance(X, np.ndarray)
#         is_tensor = isinstance(X, torch.Tensor)
#         assert is_tensor or is_array or is_list, TypeError('X should be an instance of list/np.ndarray/torch.Tensor, '
#                                                            f'but got {type(X)}')

#         # convert the data type if in need
#         if out_dtype == 'tensor':
#             if is_list:
#                 X = torch.tensor(X).to(self.device)
#             elif is_array:
#                 X = torch.from_numpy(X).to(self.device)
#             else:  # is tensor
#                 X = X.to(self.device)
#         else:  # out_dtype is ndarray
#             # convert to np.ndarray first for shape check
#             if is_list:
#                 X = np.asarray(X)
#             elif is_tensor:
#                 X = X.numpy()
#             else:  # is ndarray
#                 pass

#         # check the shape of X here
#         X_shape = X.shape
#         assert len(X_shape) == 3, f'input should have 3 dimensions [n_samples, seq_len, n_features],' \
#                                   f'but got shape={X.shape}'
#         assert X_shape[1] == expected_n_steps, f'expect X.shape[1] to be {expected_n_steps}, but got {X_shape[1]}'
#         assert X_shape[2] == expected_n_features, f'expect X.shape[2] to be {expected_n_features}, but got {X_shape[2]}'

#         if y is not None:
#             assert len(X) == len(y), f'lengths of X and y must match, ' \
#                                      f'but got f{len(X)} and {len(y)}'
#             if isinstance(y, torch.Tensor):
#                 y = y.to(self.device) if out_dtype == 'tensor' else y.numpy()
#             elif isinstance(y, list):
#                 y = torch.tensor(y).to(self.device) if out_dtype == 'tensor' else np.asarray(y)
#             elif isinstance(y, np.ndarray):
#                 y = torch.from_numpy(y).to(self.device) if out_dtype == 'tensor' else y
#             else:
#                 raise TypeError('y should be an instance of list/np.ndarray/torch.Tensor, '
#                                 f'but got {type(y)}')
#             return X, y
#         else:
#             return X



#     def fit(self, train_X, val_X=None):
#         train_X = self.check_input(self.n_steps, self.n_features, train_X)
#         if val_X is not None:
#             val_X = self.check_input(self.n_steps, self.n_features, val_X)

#         training_set = DatasetForMIT(train_X)
#         training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
#         if val_X is None:
#             self._train_model(training_loader)
#         else:
#             val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(val_X, 0.2)
#             val_X = masked_fill(val_X, 1 - val_X_missing_mask, torch.nan)
#             val_set = DatasetForMIT(val_X)
#             val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
#             self._train_model(training_loader, val_loader, val_X_intact, val_X_indicating_mask)

#         self.model.load_state_dict(self.best_model_dict)
#         self.model.eval()  # set the model as eval status to freeze it.

#     def assemble_input_data(self, data):
#         """ Assemble the input data into a dictionary.

#         Parameters
#         ----------
#         data : list
#             A list containing data fetched from Dataset by Dataload.

#         Returns
#         -------
#         inputs : dict
#             A dictionary with data assembled.
#         """
#         indices, X_intact, X, missing_mask, indicating_mask = data

#         inputs = {
#             'X': X,
#             'X_intact': X_intact,
#             'missing_mask': missing_mask,
#             'indicating_mask': indicating_mask
#         }

#         return inputs

    # def impute(self, X):
    #     X = self.check_input(self.n_steps, self.n_features, X)
    #     self.model.eval()  # set the model as eval status to freeze it.
    #     test_set = BaseDataset(X)
    #     test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
    #     imputation_collector = []

    #     with torch.no_grad():
    #         for idx, data in enumerate(test_loader):
    #             inputs = {'X': data[1], 'missing_mask': data[2]}
    #             imputed_data, _ = self.model.impute(inputs)
    #             imputation_collector.append(imputed_data)

    #     imputation_collector = torch.cat(imputation_collector)
    #     return imputation_collector.cpu().detach().numpy()
