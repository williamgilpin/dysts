## Sweep granularity


#!/usr/bin/python

import sys
import os
import json

import dysts
import dysts.flows
from dysts.datasets import *

import pandas as pd
import numpy as np

import darts
from darts.models import *
from darts import TimeSeries
import darts.models


cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
## link to TEST data

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/results_" + dataname + "_granularity_sweep.json"
dataname = dataname.replace("test", "train" )
hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"

metric_list = [
    'coefficient_of_variation',
    'mae',
    'mape',
    'marre',
    #'mase', # requires scaling with train partition; difficult to report accurately
    'mse',
    #'ope', # runs into issues with zero handling
    'r2_score',
    'rmse',
    #'rmsle', # requires positive only time series
    'smape'
]

equation_data = load_file(input_path)

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
    
granularity_vals = np.logspace(np.log10(5), np.log10(300), 10).astype(int)

for equation_name in list(equation_data.dataset.keys()):
    
    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    
    for granularity in granularity_vals:
        granularity_name = str(granularity)
        equation = getattr(dysts.flows, equation_name)()
        tpts, sol = equation.make_trajectory(12 * granularity, 
                                         resample=True, 
                                         pts_per_period=granularity,
                                         return_times=True
                                        )
        train_data = sol[:, 0]

        if equation_name not in all_results.keys():
            all_results[equation_name][granularity_name] = dict()

        split_point = int(5/6 * len(train_data))
        y_train, y_val = train_data[:split_point], train_data[split_point:]
        y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
        
        if granularity_name not in all_results[equation_name].keys():
            all_results[equation_name][granularity_name] = dict()
            
        all_results[equation_name][granularity_name]["values"] = np.squeeze(y_val)[:-1].tolist()

        for model_name in all_hyperparameters[equation_name].keys():
            if model_name in all_results[equation_name][granularity_name].keys():
                continue
            all_results[equation_name][granularity_name][model_name] = dict()

            print(equation_name + " " + model_name, flush=True)
        
            # look up season object from string
            for hyperparameter_name in all_hyperparameters[equation_name][model_name]:
                if "season" in hyperparameter_name:
                    old_val = all_hyperparameters[equation_name][model_name][hyperparameter_name]
                    old_val = str(old_val).split(".")[-1]
                    all_hyperparameters[equation_name][model_name][hyperparameter_name] = getattr(darts.utils.utils.SeasonalityMode, old_val)
            # Rescale hyperparameter to time
                if hyperparameter_name in ["lags", "input_chunk_length", "p"]:
                    old_val = all_hyperparameters[equation_name][model_name][hyperparameter_name]
                    new_val = int(old_val * granularity/100)
                    new_val = max(new_val, 5)
                    all_hyperparameters[equation_name][model_name][hyperparameter_name] = new_val            
            
            #model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])
            model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])
            if hasattr(model, "force_reset"):
                all_hyperparameters[equation_name][model_name]["force_reset"] = True
                model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])

            try:
                model.fit(y_train_ts)
                y_val_pred = model.predict(len(y_val))
                pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
                true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

                all_results[equation_name][granularity_name][model_name]["prediction"] = np.squeeze(y_val_pred.values()).tolist()
                for metric_name in metric_list:
                    metric_func = getattr(darts.metrics.metrics, metric_name)
                    all_results[equation_name][granularity_name][model_name][metric_name] = metric_func(true_y, pred_y)
            except:
                all_results[equation_name][granularity_name][model_name]["prediction"] = float("nan")
                for metric_name in metric_list:
                    metric_func = getattr(darts.metrics.metrics, metric_name)
                    all_results[equation_name][granularity_name][model_name][metric_name] = float("nan")

            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=4)   
            
#             break
#         break
#     break
        



