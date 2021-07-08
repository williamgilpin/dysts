#!/usr/bin/python

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .causal_cnn import CausalCNNEncoder, CausalConvolutionBlock


class UnivariateTimeSeries(Dataset):
    """
    A dataset representation for univariate time series
    
    Args:
        ts (np.ndarray): A time series dataset of length T
    """

    def __init__(self, ts, seqlen):
        self.ts = torch.tensor(ts[None, :], dtype=torch.float32, requires_grad=True)
        self.seqlen = seqlen

    def __len__(self):
        return self.ts.shape[-1] - self.seqlen

    def __getitem__(self, idx):
        input_val = self.ts[:, idx : idx + self.seqlen]
        output_val = self.ts[:, idx : idx + self.seqlen]

        return input_val, output_val


class TimeSeriesCollection(Dataset):
    """
    A dataset representation for a collection time series
    
    Args:
        ts (np.ndarray): A time series dataset of shape T x D
    """

    def __init__(self, ts, seqlen):
        self.nt, self.nb = ts.shape
        self.ts = torch.tensor(ts[None, ...], dtype=torch.float32, requires_grad=True)
        self.seqlen = seqlen

    def __len__(self):
        return self.nb * (self.nt - self.seqlen)

    def __getitem__(self, idx):

        i, j = np.unravel_index(idx, (self.nb, (self.nt - self.seqlen)))

        input_val = self.ts[:, j : j + self.seqlen, i]
        output_val = self.ts[:, j : j + self.seqlen, i]

        return input_val, output_val


class Autoencoder(nn.Module):
    """
    A causal dilated autoencoder for time series
    """

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=2),  # b, 16, 10, 10
            CausalConvolutionBlock(1, 16, 3, 2, final=False),
            #             CausalConvolutionBlock(16, 8, 3, 2, final=False),
            nn.Conv1d(16, 8, kernel_size=3, stride=3, padding=1),  # b, 8, 3, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                8, 16, kernel_size=3, stride=1, padding=1
            ),  # b, 16, 5, 5
            nn.ELU(True),
            nn.ConvTranspose1d(
                16, 8, kernel_size=3, stride=3, padding=1
            ),  # b, 8, 15, 15
            nn.ELU(True),
            nn.ConvTranspose1d(
                8, 1, kernel_size=3, stride=1, padding=1
            ),  # b, 1, 28, 28
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x