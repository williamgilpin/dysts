#!/usr/bin/python

import sys
import os
import json

import dysts
from dysts.datasets import *

import numpy as np

import darts
from darts.models import *
from darts import TimeSeries
import darts.models

import dysts.metrics


cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"
# input_path = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_60.json.gz"

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/data_sweep_" + dataname + ".json"
dataname = dataname.replace("test", "train" )
hyperparameter_path = cwd + "/hyperparameters/hyperparameters_multivariate_" + dataname + ".json"
hyperparameter_path = cwd + "/hyperparameters/hyperparameters_multivariate_train_multivariate__pts_per_period_100__periods_12.json"

import torch
has_gpu = torch.cuda.is_available()
print("has gpu: ", torch.cuda.is_available(), flush=True)
n = torch.cuda.device_count()
print(f"{n} devices found.", flush=True)
if not has_gpu:
    warnings.warn("No GPU found.")
    gpu_params = {
        "accelerator": "cpu",                                                                             
    }
else:
    warnings.warn("GPU working.")
    gpu_params = {
        "accelerator": "gpu",
        "devices": n,
    }

pl_trainer_kwargs = [gpu_params]
model_static_dict = {"pl_trainer_kwargs" : pl_trainer_kwargs}


equation_data = load_file(input_path)

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

model_names = list(all_hyperparameters["Aizawa"].keys())
equation_names = list(equation_data.dataset.keys())

# print("model_names: ", model_names, flush=True)
# model_names = ["RandomForest"]
# model_names = ["LinearRegressionModel"]

## load file and clean out dictionary of null keys
try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
for equation_name in equation_data.dataset:
    for model_name in model_names:
        if model_name not in all_results[equation_name].keys():
            continue
        if len(all_results[equation_name][model_name]) == 0:
            all_results[equation_name].pop(model_name, None)

for equation_name in equation_names:
    
    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    else:
        print(f"{equation_name} seen before.", flush=True) 
    
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))
    split_point = int(5 / 6 * len(train_data)) # long horizon
    max_lag_val = 100 # need at least one full lookback window to fit a model
    truncate_inds = np.unique(np.linspace(0, split_point - max_lag_val - 1, 25).astype(int))
        
    for model_name in model_names:
        
        ## re-load file and clean out dictionary of null keys
#         try:
#             with open(output_path, "r") as file:
#                 all_results = json.load(file)
#         except FileNotFoundError:
#             all_results = dict()
#         for equation_name in equation_data.dataset:
#             for model_name in model_names:
#                 if model_name not in all_results[equation_name].keys():
#                     continue
#                 if len(all_results[equation_name][model_name]) == 0:
#                     all_results[equation_name].pop(model_name, None)
        
    
        if model_name not in all_results[equation_name].keys():
            all_results[equation_name][model_name] = list()
        elif len(all_results[equation_name][model_name]) >= len(truncate_inds):
            print(f"Skipping {model_name} for {equation_name}", flush=True)
            continue
        else:
            ## reset count
            all_results[equation_name][model_name] = list()

        for i in truncate_inds:
            y_train, y_val = train_data[i:split_point], train_data[split_point:]
            # print("y_train.shape: ", y_train.shape, flush=True)
            y_train_ts, y_test_ts = (
                TimeSeries.from_values(y_train), 
                TimeSeries.from_values(y_val)
            )
            all_hyperparameters[equation_name][model_name].pop("pl_trainer_kwargs", None)
            # if model_name in all_results[equation_name].keys():
            #     continue
            # all_results[equation_name][model_name] = dict()
            
            
            
            # The seasonality hyperparameter is a string, but needs to be a SeasonalityMode 
            # object for the darts models
            for hyperparameter_name in all_hyperparameters[equation_name][model_name]:
                if "season" in hyperparameter_name:
                    old_val = all_hyperparameters[equation_name][model_name][hyperparameter_name]
                    all_hyperparameters[equation_name][model_name][hyperparameter_name] = getattr(darts.utils.utils.SeasonalityMode, old_val)
            try:
                pc = dict()
                pc.update(all_hyperparameters[equation_name][model_name])
                pc.update({"pl_trainer_kwargs": [gpu_params]})
                model = getattr(darts.models, model_name)(**pc)
            except:
                model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])
            
            ## Fit the forecasting model on the given data
            model.fit(y_train_ts)

            ## Attempt to predict the validation data. If it fails, then the model returns
            ## a None value for the prediction.
            try:
                y_val_pred = model.predict(len(y_val)).values().squeeze()
            except:
                print(f"Failed to predict {equation_name} {model_name}")
                y_val_pred = np.array([None] * len(y_val))
            #all_results[equation_name][model_name]["prediction"] = y_val_pred.tolist()

            ## Attempt to compute several time series distance metrics. If it fails due
            ## to a ValueError, then all metrics are not defined for the given time series.
            try:
                score = dysts.metrics.smape(y_val, y_val_pred)
            except ValueError:
                score = None

            all_results[equation_name][model_name].append(score)
        print(equation_name + " " + model_name, flush=True)
        ## Update results dictionary with metrics, then save to JSON file.
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)   