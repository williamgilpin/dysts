import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint


class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class ODEFunc(nn.Module):

    def __init__(self, input_shape, n_units=30):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, n_units),
            nn.Tanh(),
            ResNet(
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.SiLU(),
                    nn.Linear(n_units, n_units),
                    nn.SiLU(),
                )
            ),
            nn.Linear(n_units, input_shape),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
    

class BatchLoader:
    """
    Class for loading batches of data in a format for the neural ODE

    Parameters:
        dataset (np.ndarray): The dataset to load, with shape (nseries, maxt, dim)
        tlen (int): The length of the time series to load
        tpts (np.ndarray): The time points to load
        replace (bool): Whether to sample batches with replacement
        batch_size (int): The number of time series to load into each batch

    """
    def __init__(
        self, dataset, tlen, tpts=None, replace=False, batch_size=64,
        standardize=True,
        random_state=None
    ):

        if len(dataset.shape) == 1:
            self.data = dataset[None, :, None]
        elif len(dataset.shape) == 2:
            self.data = dataset[None, ...]
        else:
            self.data = dataset
            

        (self.npts, self.maxt, self.dim) = self.data.shape

        if tpts is not None:
            self.tpts = tpts
        else:
            self.tpts = np.arange(self.maxt)

        self.tlen = tlen
        self.batch_size = batch_size
        self.random_state = random_state
        self.replace = replace
        np.random.seed(self.random_state)

        ## Prevent batches that are too large
        if self.batch_size > self.maxt - self.tlen:
            self.batch_size = self.maxt - self.tlen

        # self.device = torch.device(
        #     "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
        # )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def get_batch(self):
        """
        Randomly sample a batch of time series data
        """
        # pick M random timepoints for the batch
        # pick M random ic points for the batch
        ic_inds = torch.from_numpy(
            np.random.choice(
                np.arange(self.npts, dtype=np.int64), self.batch_size, replace=True,
            )
        )

        time_inds = torch.from_numpy(
            np.random.choice(
                np.arange(self.maxt - self.tlen, dtype=np.int64),
                self.batch_size,
                replace=self.replace,
            )
        )

        batch_y0 = torch.from_numpy(
            np.vstack(
                [
                    self.data[ic_val, time_val].astype(np.float32)
                    for ic_val, time_val in zip(ic_inds, time_inds)
                ]
            )
        )

        batch_t = torch.from_numpy(self.tpts[:self.tlen].astype(np.float32))  # (T) ##?
        sol_batch = self.data[np.array(ic_inds)]

        batch_full = list()
        for ic_val, time_val in zip(ic_inds, time_inds):
            sub_batch = list()
            for i in range(self.tlen):
                sub_batch.append(
                    torch.from_numpy(self.data[ic_val, time_val + i].astype(np.float64))
                )
            batch_full.append(torch.stack(sub_batch, dim=0))
        batch_y = torch.transpose(torch.stack(batch_full, dim=0), 0, 1)

        return (
            batch_y0.to(self.device),
            batch_t.to(self.device),
            batch_y.to(self.device),
        )




class NODEForecast:
    """
    A wrapper class for a neural ODE forecasting model. The current version does not
    use timepoint values for anything; if needed, append time values to the input as an 
    additional dimension.

    Parameters:
        ndim (int): the dimensionality of the time series
        time_lookback (int): the number of timepoints to use for the ODE
        func (torch.nn.Module): the neural ODE function
        random_state (int): the random seed

    Methods:
        fit(X, niters=500, learning_rate=1e-2, batch_size=128): fit the model to a dataset
        predict(nt, ic=None, t=None): predict the future time series values

    Example:
        model = NODEForecast(train_data.shape[-1], 30)
        model.fit(train_data)
        sol_pred = model.predict(100)
    """

    def __init__(self, ndim, time_lookback, random_state=None):
        self.time_lookback = time_lookback
        self.ndim = ndim
        self.func = ODEFunc(self.ndim)
        self.random_state = random_state
        
    def fit(self, X, niters=500, learning_rate=1e-2, batch_size=128):
        """
        Fit the model to a dataset

        Args:
            X (np.ndarray): the time series data to fit the model to
            niters (int): the number of iterations to train for
            learning_rate (float): the learning rate for the optimizer
            batch_size (int): the batch size to use for training

        """
        self.batch_size = batch_size
        nt = X.shape[0]
        t_train = np.arange(nt)
        bt = BatchLoader(
            X, 
            self.time_lookback, 
            tpts=t_train, 
            batch_size=self.batch_size, 
            random_state=self.random_state
        )
        self.func = self.func.to(bt.device)
        optimizer = optim.Adam(self.func.parameters(), lr=learning_rate)
        optimizer.zero_grad()

        batch_y0, batch_t, batch_y = bt.get_batch()
        pred_y = odeint(self.func, batch_y0, batch_t).to(bt.device)
        
        loss_history = list()
        for itr in range(1, niters + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = bt.get_batch()
            pred_y = odeint(self.func, batch_y0, batch_t).to(bt.device)
            loss = torch.mean(torch.square(pred_y - batch_y))  # MSE
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
        self.loss_history = loss_history
        self.ic = X[-1]
        self.bt = bt

        return self

    def predict(self, nt, ic=None, t=None):
        """
        Predict future time series values

        Args:
            nt (int): the number of timepoints to predict
            ic (np.ndarray): the initial condition for the ODE
            t (np.ndarray): the timepoints to use for the ODE

        Returns:
            sol_pred (np.ndarray): the predicted time series values
        """
        if ic is None:
            ic = self.ic

        t_test = np.arange(nt)
        sol_pred = odeint(self.func, 
            torch.from_numpy(ic.astype(np.float32)).to(self.bt.device),
            torch.from_numpy(t_test.astype(np.float32)).to(self.bt.device)
            ).to(self.bt.device)
        sol_pred = sol_pred.detach().cpu().numpy()
        return sol_pred