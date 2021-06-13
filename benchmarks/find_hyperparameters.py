#!/usr/bin/python

import sys
import os
import json

# import dysts
from dysts.datasets import *

import pandas as pd

from darts.models import *
from darts import TimeSeries
import darts.models

pts_per_period = 100
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE, darts.utils.utils.SeasonalityMode.NONE]
time_delays = [1, 5, int(0.25 * pts_per_period), int(0.5 * pts_per_period), pts_per_period, int(1.5 * pts_per_period)]
time_delays = [3, 5, int(0.25 * pts_per_period)]

time_models = {
    'ARIMA': {"p": 10},
#     'LinearRegressionModel' : {"lags": 10},
#     'NBEATSModel' : {"input_chunk_length": 40, "output_chunk_length": 1},
    'RandomForest' : {"lags": 10, "lags_exog": None},
#     'TCNModel' : {"input_chunk_length": 40, "output_chunk_length": 1},
#     "TransformerModel" :  {"input_chunk_length": 100, "output_chunk_length": 1},
}

seasonality_models = {
    'ExponentialSmoothing' : {"seasonal": darts.utils.utils.SeasonalityMode.ADDITIVE},
    'FourTheta' : {"season_mode": darts.utils.utils.SeasonalityMode.ADDITIVE},
    'Theta' : {"season_mode": darts.utils.utils.SeasonalityMode.ADDITIVE},
}

null_models = ["AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", "NaiveSeasonal", "Prophet"]


cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/train_univariate__pts_per_period_100__periods_12.json"
dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = "./hyperparameters/hyperparameters_" + dataname + ".json"

equation_data = load_file(input_path)



all_hyperparameters = dict()
for equation_name in equation_data.dataset:
    print(equation_name)
    train_data = np.array(equation_data.dataset[equation_name]["values"])
    all_hyperparameters[equation_name] = dict()
    
    split_point = int(5/6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    
    for model_name in time_models:
        print("\t" + model_name)
        all_scores = list()
        for tau in time_delays:
            kwarg_vals = time_models[model_name]
            kwarg_vals[list(kwarg_vals.keys())[0]] = tau
            model = getattr(darts.models, model_name)(**kwarg_vals)
            
            y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
            model.fit(y_train_ts)
            y_val_pred = model.predict(len(y_val))

            pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
            true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

            all_scores.append(darts.metrics.mse(true_y, pred_y))
        best_tau = time_delays[np.argmin(all_scores)]
        kwarg_vals = time_models[model_name]
        kwarg_vals[list(kwarg_vals.keys())[0]] = best_tau
        all_hyperparameters[equation_name][model_name] = kwarg_vals
        
    ## These models don't require hyperparameter tuning
    for model_name in null_models:
        all_hyperparameters[equation_name][model_name] = {}


    ## Tune seasonality parameter      
    for model_name in seasonality_models:
        all_scores = list()
        for season in season_values:
            kwarg_vals = seasonality_models[model_name]
            kwarg_vals[list(kwarg_vals.keys())[0]] = season
            model = getattr(darts.models, model_name)(**kwarg_vals)
            
            y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
            model.fit(y_train_ts)
            y_val_pred = model.predict(len(y_val))

            pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
            true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

            all_scores.append(darts.metrics.mse(true_y, pred_y))
        best_season = season_values[np.argmin(all_scores)]
        kwarg_vals = seasonality_models[model_name]
        kwarg_vals[list(kwarg_vals.keys())[0]] = best_season.name
        all_hyperparameters[equation_name][model_name] = kwarg_vals
        
    with open(output_path, 'w') as f:
        json.dump(all_hyperparameters, f, indent=4)   
    