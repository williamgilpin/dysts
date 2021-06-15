#!/usr/bin/python

import sys
import os
import json

import dysts
from dysts.datasets import *

import pandas as pd

from darts.models import *
from darts import TimeSeries
import darts.models

pts_per_period = 100
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE, darts.utils.utils.SeasonalityMode.NONE]
time_delays = [3, 5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period, int(1.5 * pts_per_period)]
time_delays = [3, 5, int(0.25 * pts_per_period)]

time_delays = [1, 5, 10, 25]
network_inputs = [5, 10, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period]
SKIP_EXISTING = False


time_models = {
    'ARIMA': {"p": 10},
    'LinearRegressionModel' : {"lags": 10},
    'RandomForest' : {"lags": 10, "lags_exog": None},
}

network_models = {
    'NBEATSModel' : {"input_chunk_length": 40, "output_chunk_length": 1},
    'TCNModel' : {"input_chunk_length": 40, "output_chunk_length": 1},
    "TransformerModel" :  {"input_chunk_length": 100, "output_chunk_length": 1},
}

seasonality_models = {
    'ExponentialSmoothing' : {"seasonal": darts.utils.utils.SeasonalityMode.ADDITIVE},
    'FourTheta' : {"season_mode": darts.utils.utils.SeasonalityMode.ADDITIVE},
    'Theta' : {"season_mode": darts.utils.utils.SeasonalityMode.ADDITIVE},
}

null_models = ["AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", "NaiveSeasonal", "Prophet"]


cwd = os.path.dirname(os.path.realpath(__file__))
input_path = os.path.dirname(cwd)  + "/dysts/data/train_univariate__pts_per_period_100__periods_12.json"
dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"

equation_data = load_file(input_path)


try:
    with open(output_path, "r") as file:
        all_hyperparameters = json.load(file)
except FileNotFoundError:
    all_hyperparameters = dict()
    
    
for equation_name in equation_data.dataset:
    
    if SKIP_EXISTING and equation_name in all_hyperparameters.keys():
        print(f"Entry for {equation_name} found, skipping it.")
        continue
    
    print(equation_name)
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))
    all_hyperparameters[equation_name] = dict()
    
    split_point = int(5/6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    
    try:
        for model_name in time_models:
            print("\t" + model_name)
            all_scores = list()
            for tau in time_delays:
                kwarg_vals = time_models[model_name].copy()
                kwarg_vals[list(kwarg_vals.keys())[0]] = tau
                model = getattr(darts.models, model_name)(**kwarg_vals)

                y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
                model.fit(y_train_ts)
                y_val_pred = model.predict(len(y_val))

                pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
                true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

                all_scores.append(darts.metrics.mse(true_y, pred_y))
            best_tau = time_delays[np.argmin(all_scores)]
            kwarg_vals = time_models[model_name].copy()
            kwarg_vals[list(kwarg_vals.keys())[0]] = best_tau
            all_hyperparameters[equation_name][model_name] = kwarg_vals.copy()
    except: 
        print(model_name, "1")
        
    try:
        for model_name in network_models:
            print("\t" + model_name)
            all_scores = list()
            for tau in network_inputs:
                kwarg_vals = network_models[model_name].copy()
                kwarg_vals[list(kwarg_vals.keys())[0]] = tau
                model = getattr(darts.models, model_name)(**kwarg_vals)

                y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
                model.fit(y_train_ts)
                y_val_pred = model.predict(len(y_val))

                pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
                true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

                all_scores.append(darts.metrics.mse(true_y, pred_y))
            best_tau = network_inputs[np.argmin(all_scores)]
            kwarg_vals = network_models[model_name].copy()
            kwarg_vals[list(kwarg_vals.keys())[0]] = best_tau
            all_hyperparameters[equation_name][model_name] = kwarg_vals.copy()
    except: 
        print(model_name, "2")
        
    ## These models don't require hyperparameter tuning
    for model_name in null_models:
        all_hyperparameters[equation_name][model_name] = {}


    ## Tune seasonality parameter
    try:
        for model_name in seasonality_models:
            all_scores = list()
            for season in season_values:
                kwarg_vals = seasonality_models[model_name].copy()
                kwarg_vals[list(kwarg_vals.keys())[0]] = season
                model = getattr(darts.models, model_name)(**kwarg_vals)

                y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
                model.fit(y_train_ts)
                y_val_pred = model.predict(len(y_val))

                pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
                true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

                all_scores.append(darts.metrics.mse(true_y, pred_y))
            best_season = season_values[np.argmin(all_scores)]
            kwarg_vals = seasonality_models[model_name].copy()
            kwarg_vals[list(kwarg_vals.keys())[0]] = best_season.name
            all_hyperparameters[equation_name][model_name] = kwarg_vals.copy()
    except: 
        print(model_name, "3")
    
    with open(output_path, 'w') as f:
        json.dump(all_hyperparameters, f, indent=4)   
    