#!/usr/bin/python

import sys
import os
import json

import dysts
from dysts.datasets import *

import pandas as pd

import darts
from darts.models import *
from darts import TimeSeries
import darts.models

import echotorch.nn as etnn
import echotorch.utils
from echotorch.utils.matrix_generation import *

from models import models
from models.LiESN import LiESNFitter
from rc_chaos.Methods.RUN import new_args_dict

cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()


# TODO: use full data for better fitting? (may overfit hyperparams)
input_path = os.path.dirname(cwd) + "/dysts/data/train_univariate__pts_per_period_15__periods_12.json"
pts_per_period = 15
network_inputs = [5, 10, int(0.5 * pts_per_period), pts_per_period]  # can't have kernel less than 5

# input_path = os.path.dirname(cwd)  + "/dysts/data/train_univariate__pts_per_period_100__periods_12.json"
# pts_per_period = 100
# network_inputs = [5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period]

SKIP_EXISTING = True
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE,
                 darts.utils.utils.SeasonalityMode.NONE,
                 darts.utils.utils.SeasonalityMode.MULTIPLICATIVE]
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE,
                 darts.utils.utils.SeasonalityMode.NONE
                 ]
time_delays = [3, 5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period,
               int(1.5 * pts_per_period)]
time_delays = [3, 5, 10, int(0.25 * pts_per_period)]
network_outputs = [1, 4]
network_outputs = [1]

import torch

has_gpu = torch.cuda.is_available()
if not has_gpu:
    warnings.warn("No GPU found.")
else:
    warnings.warn("GPU working.")

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"

equation_data = load_file(input_path)

try:
    with open(output_path, "r") as file:
        all_hyperparameters = json.load(file)
except FileNotFoundError:
    all_hyperparameters = dict()

parameter_candidates = dict()

parameter_candidates["ARIMA"] = {"p": time_delays}
parameter_candidates["LinearRegressionModel"] = {"lags": time_delays}
parameter_candidates["RandomForest"] = {"lags": time_delays, "lags_exog": [None]}
parameter_candidates["NBEATSModel"] = {"input_chunk_length": network_inputs, "output_chunk_length": network_outputs}
parameter_candidates["TCNModel"] = {"input_chunk_length": network_inputs, "output_chunk_length": network_outputs}
parameter_candidates["TransformerModel"] = {"input_chunk_length": network_inputs,
                                            "output_chunk_length": network_outputs}
parameter_candidates["RNNModel"] = {
    "input_chunk_length": network_inputs,
    "output_chunk_length": network_outputs,
    "model": ["LSTM"],
    "n_rnn_layers": [2],
    "n_epochs": [200]
}
parameter_candidates["ExponentialSmoothing"] = {"seasonal": season_values}
parameter_candidates["FourTheta"] = {"season_mode": season_values}
parameter_candidates["Theta"] = {"season_mode": season_values}
for model_name in ["AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", "NaiveSeasonal", "Prophet"]:
    parameter_candidates[model_name] = {}

# TODO: find and try out new values
# TODO: run with SKIP_EXISTING FALSE to rerun experiments that were conducted during debugging
# first entries are default
parameter_candidates['esn'] = {
    "reservoir_size": [1000, 2000, 5000],
    "sparsity": [0.01, 0.1], # 0.001 does not work/converge
    "radius": [0.6, 0.2, 1.0],
    "sigma_input": [1],
    "dynamics_fit_ratio": [2 / 7, 0.05, 0.1, 0.5],
    "regularization": [0.0],
    "scaler_tt": ['Standard', 'MinMaxZeroOne'],
    "solver": ['auto']  # "pinv", "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"
}

# TODO find good values
# TODO also try other ESN models
n_train_samples = [1]
n_test_samples = [1]
spectral_radius = [0.9]
leaky_rate = [1.0]
input_dim = [1]
n_hidden = [100]
learning_algos = ['inv']  # , 'pinv']

# TODO
'''
parameter_candidates['LiESN'] = {
    "lags": time_delays,
    "input_dim": input_dim,
    "hidden_dim": n_hidden,
    "output_dim": [1],
    "spectral_radius": spectral_radius,
    "learning_algo": learning_algos,
    "leaky_rate": leaky_rate,
    "w_generator": [MatrixGenerator()],  # TODO: check these
    "win_generator": [MatrixGenerator()],
    "wbias_generator": [MatrixGenerator()]
}
'''

for equation_name in equation_data.dataset:

    print(equation_name, flush=True)

    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    if equation_name not in all_hyperparameters.keys():
        all_hyperparameters[equation_name] = dict()

    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

    for model_name in parameter_candidates.keys():
        print(equation_name + " " + model_name)
        if SKIP_EXISTING and model_name in all_hyperparameters[equation_name].keys():
            print(f"Entry for {equation_name} - {model_name} found, skipping it.")
            continue

        if model_name in models:
            model = models[model_name]()
        else:
            model = getattr(darts.models, model_name)
        model_best = model.gridsearch(parameter_candidates[model_name], y_train_ts, val_series=y_test_ts,
                                      metric=darts.metrics.mse)

        best_hyperparameters = model_best[1].copy()

        # Write season object to string
        for hyperparameter_name in best_hyperparameters:
            if "season" in hyperparameter_name:
                best_hyperparameters[hyperparameter_name] = best_hyperparameters[hyperparameter_name].name

        all_hyperparameters[equation_name][model_name] = best_hyperparameters

    with open(output_path + 'DEBUG', 'w') as f:
        json.dump(all_hyperparameters, f, indent=4)
