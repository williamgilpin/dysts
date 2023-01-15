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
input_path = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json"
input_path = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_60.json"

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/results_" + dataname + ".json"
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
        #    "gpus": [0],  # use "devices" instead of "gpus" for PyTorch Lightning >= 1.7.                                    #    "auto_select_gpu": True,                                                                                 
    }

pl_trainer_kwargs = [gpu_params]
model_static_dict = {"pl_trainer_kwargs" : pl_trainer_kwargs}


equation_data = load_file(input_path)

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
    

for equation_name in equation_data.dataset:
    
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    
    # split_point = int(5 / 6 * len(train_data))
    split_point = int(1 / 6 * len(train_data)) # long horizon
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_values(train_data).split_before(split_point)
    
    all_results[equation_name]["values"] = np.squeeze(y_val)[:-1].tolist()
    
    for model_name in all_hyperparameters[equation_name].keys():
        all_hyperparameters[equation_name][model_name].pop("pl_trainer_kwargs", None)
        if model_name in all_results[equation_name].keys():
            continue
        all_results[equation_name][model_name] = dict()
        
        print(equation_name + " " + model_name, flush=True)
        
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
        
        ## Fit the forecasting model on the given data, and then predict the validation
        ## set and store the predictions in the results dictionary.
        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val)).values().squeeze()
        all_results[equation_name][model_name]["prediction"] = y_val_pred.tolist()

        ## Attempt to compute several time series distance metrics. If it fails due
        ## to a ValueError, then all metrics are not defined for the given time series.
        try:
	        all_metrics = dysts.metrics.compute_metrics(y_val, y_val_pred)
        except ValueError:
            all_metrics = dysts.metrics.compute_metrics(y_val, y_val)
            for	key in all_metrics:
                all_metrics[key] = None

        ## Update results dictionary with metrics, then save to JSON file.
        all_results[equation_name][model_name].update(all_metrics)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)   