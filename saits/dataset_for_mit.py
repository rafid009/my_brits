import torch
import numpy as np
from pypots.data.base import BaseDataset





class DatasetForMIT(BaseDataset):
    """ Dataset for models that need MIT (masked imputation task) in their training, such as SAITS.

    For more information about MIT, please refer to :cite:`du2022SAITS`.

    Parameters
    ----------
    X : tensor, shape of [n_samples, n_steps, n_features]
        Time-series feature vector.

    y : tensor, shape of [n_samples], optional, default=None,
        Classification labels of according time-series samples.

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.

        Note that,
        `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.

    """

    def __init__(self, X, mean, std, y=None, rate=0.1, is_rand=False):
        super().__init__(X, y)
        self.rate = rate
        self.is_rand = is_rand
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        """ Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.

        Returns
        -------
        dict,
            A dict contains

            index : int tensor,
                The index of the sample.

            X_intact : tensor,
                Original time-series for calculating mask imputation loss.

            X : tensor,
                Time-series data with artificially missing values for model input.

            missing_mask : tensor,
                The mask records all missing values in X.

            indicating_mask : tensor.
                The mask indicates artificially missing values in X.
        """
        X = self.X[idx]
        X = torch.tensor(X)
        X = (X - self.mean) / self.std
        if self.is_rand:
            X_intact, X, missing_mask, indicating_mask = self.mcar(X, rate=self.rate, nan=-1)
        else:
            X_intact, X, missing_mask, indicating_mask = self.mcar(X, rate=self.rate)

        # X = ((X - self.mean) / self.std) * indicating_mask
        # X_intact = ((X_intact - self.mean) / self.std) * missing_mask

        sample = [
            torch.tensor(idx),
            X_intact.to(torch.float32),
            X.to(torch.float32),
            missing_mask.to(torch.float32),
            indicating_mask.to(torch.float32),
        ]

        if self.y is not None:
            sample.append(
                self.y[idx].to(torch.long)
            )

        return sample

    def mcar(self, X, rate, nan=0):
        """ Create completely random missing values (MCAR case).

        Parameters
        ----------
        X : array,
            Data vector. If X has any missing values, they should be numpy.nan.

        rate : float, in (0,1),
            Artificially missing rate, rate of the observed values which will be artificially masked as missing.

            Note that,
            `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
            not (number of artificially missing values) / np.product(self.data.shape),
            considering that the given data may already contain missing values,
            the latter way may be confusing because if the original missing rate >= `rate`,
            the function will do nothing, i.e. it won't play the role it has to be.

        nan : int/float, optional, default=0
            Value used to fill NaN values.

        Returns
        -------
        X_intact : array,
            Original data with missing values (nan) filled with given parameter `nan`, with observed values intact.
            X_intact is for loss calculation in the masked imputation task.

        X : array,
            Original X with artificial missing values. X is for model input.
            Both originally-missing and artificially-missing values are filled with given parameter `nan`.

        missing_mask : array,
            The mask indicates all missing values in X.
            In it, 1 indicates observed values, and 0 indicates missing values.

        indicating_mask : array,
            The mask indicates the artificially-missing values in X, namely missing parts different from X_intact.
            In it, 1 indicates artificially missing values, and other values are indicated as 0.
        """
        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(X, np.ndarray):
            return self._mcar_numpy(X, rate, nan)
        elif isinstance(X, torch.Tensor):
            return self._mcar_torch(X, rate, nan)
        else:
            raise TypeError('X must be type of list/numpy.ndarray/torch.Tensor, '
                            f'but got {type(X)}')


    def _mcar_numpy(self, X, rate, nan=0):
        original_shape = X.shape
        X = X.flatten()
        X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
        # select random indices for artificial mask
        indices = np.where(~np.isnan(X))[0].tolist()  # get the indices of observed values
        print(f"orig miss: {len(indices)}")
        indices = np.random.choice(indices, int(len(indices) * rate), replace=False)
        # create artificially-missing values by selected indices
        X[indices] = np.nan  # mask values selected by indices
        indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
        missing_mask = (~np.isnan(X_intact)).astype(np.float32)
        if nan == -1:
            intact_mask = np.isnan(X_intact)
            X_mask = np.isnan(X)
            rand_nums = np.random.randn(*X.shape)
            X_intact[intact_mask] = rand_nums[intact_mask] #np.nan_to_num(X_intact, nan=nan)
            X[X_mask] = rand_nums[X_mask] #np.nan_to_num(X, nan=nan)
        else:
            X_intact = np.nan_to_num(X_intact, nan=nan)
            X = np.nan_to_num(X, nan=nan)
        # reshape into time-series data
        X_intact = X_intact.reshape(original_shape)
        X = X.reshape(original_shape)
        missing_mask = missing_mask.reshape(original_shape)
        indicating_mask = indicating_mask.reshape(original_shape)
        return X_intact, X, missing_mask, indicating_mask


    def _mcar_torch(self, X, rate, nan=0):
        X = X.clone()  # clone X to ensure values of X out of this function not being affected
        original_shape = X.shape
        X = X.flatten()
        X_intact = torch.clone(X)  # keep a copy of originally observed values in X_intact
        # select random indices for artificial mask
        indices = torch.where(~torch.isnan(X))[0].tolist()  # get the indices of observed values
        # print(f"orig miss: {len(indices)}\nsupposed: {int(len(indices) * rate)}")
        indices = np.random.choice(indices, int(len(indices) * rate), replace=False)
        # print(f"indicate indices: {indices}\nlength: {len(indices)}")
        # create artificially-missing values by selected indices
        X[indices] = torch.nan  # mask values selected by indices
        indicating_mask = (~torch.isnan(X)).type(torch.float32)#((~torch.isnan(X_intact)) ^ (~torch.isnan(X))).type(torch.float32)
        missing_mask = (~torch.isnan(X_intact)).type(torch.float32)
        if nan == -1:
            intact_mask = torch.isnan(X_intact)
            X_mask = torch.isnan(X)
            rand_nums = torch.randn_like(X)
            X_intact[intact_mask] = rand_nums[intact_mask] #np.nan_to_num(X_intact, nan=nan)
            X[X_mask] = rand_nums[X_mask] #np.nan_to_num(X, nan=nan)
        else:
            X_intact = torch.nan_to_num(X_intact, nan=nan)
            X = torch.nan_to_num(X, nan=nan)
        # reshape into time-series data
        X_intact = X_intact.reshape(original_shape)
        X = X.reshape(original_shape)
        missing_mask = missing_mask.reshape(original_shape)
        indicating_mask = indicating_mask.reshape(original_shape)
        return X_intact, X, missing_mask, indicating_mask

