#!/usr/bin/python

import sys
import os
import json

import dysts
from dysts.datasets import *
import time

import numpy as np

import darts
from darts.models import *
from darts import TimeSeries
import darts.models

import dysts.metrics

from resources.node import NODEForecast
from resources.esn import ESNForecast

cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/results_" + dataname + "_timing.json"
dataname = dataname.replace("test", "train" )
hyperparameter_path = cwd + "/hyperparameters/hyperparameters_multivariate__pts_per_period_100__periods_12.json"
equation_data = load_file(input_path)

SEED = 0

## No GPU for fair comparison
# import torch
# has_gpu = torch.cuda.is_available()
# print("has gpu: ", torch.cuda.is_available(), flush=True)
# n = torch.cuda.device_count()
# print(f"{n} devices found.", flush=True)
# if not has_gpu:
#     warnings.warn("No GPU found.")
#     gpu_params = {
#         "accelerator": "cpu",                                                                                    
#     }
# else:
#     warnings.warn("GPU working.")
#     gpu_params = {
#         "accelerator": "gpu",
#         "devices": n,
#         #    "gpus": [0],  # use "devices" instead of "gpus" for PyTorch Lightning >= 1.7.                                    #    "auto_select_gpu": True,                                                                                 
#     }
# pl_trainer_kwargs = [gpu_params]
# model_static_dict = {"pl_trainer_kwargs" : pl_trainer_kwargs}
model_static_dict = {}

input_path_train = os.path.dirname(cwd)  + "/dysts/data/train_multivariate__pts_per_period_100__periods_12.json.gz"
input_path_test = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"

equation_data_train = load_file(input_path_train)
equation_data_test = load_file(input_path_test)

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
    
from dysts.metrics import smape
score_func = smape

for equation_name in equation_data.dataset:
    
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    
    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_values(train_data).split_before(split_point)
    
    all_results[equation_name]["values"] = np.squeeze(y_val)[:-1].tolist()
    

    ## Run darts timing benchmarks
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
            model = getattr(darts.models, model_name)(**pc)
        except:
            model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])
        
        ## Fit the forecasting model on the given data        
        time_start = time.perf_counter()
        model.fit(y_train_ts)
        time_end = time.perf_counter()
        fit_time = str(time_end - time_start)

        ## Attempt to predict the validation data. If it fails, then the model returns
        ## a None value for the prediction.
        try:    
            time_start = time.perf_counter()
            y_val_pred = model.predict(len(y_val)).values().squeeze()
            time_end = time.perf_counter()
            predict_time = str(time_end - time_start)
        except:
            print(f"Failed to predict {equation_name} {model_name}")
            y_val_pred = np.array([None] * len(y_val))
            predict_time = None

        ## Update results dictionary with metrics, then save to JSON file.
        all_results[equation_name][model_name]["Train time"] = fit_time
        all_results[equation_name][model_name]["Inference time"] = predict_time
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)   



    train_data = np.copy(np.array(equation_data_train.dataset[equation_name]["values"]))
    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]

    t_train, sol_train = np.arange(len(y_train)), y_train
    t_test, sol_test = np.arange(len(y_val)), y_val

    ## Run node timing benchmarks
    tau_vals = np.array([2, 5, 15, 30, 60, 120])
    t_train = np.arange(len(y_train))
    t_test = np.arange(len(y_val))

    all_scores = list()
    for tau in tau_vals:
        try:
            model = NODEForecast(train_data.shape[-1], tau, random_state=0)
            model.fit(y_train)
            sol_pred = model.predict(200)
        except AssertionError:
            print("Integration error encountered, skipping this entry for now")
            continue
        score_val = score_func(sol_test, sol_pred)
        all_scores.append(score_val)
    tau_opt = tau_vals[np.argmin(all_scores)]

    test_data = np.copy(np.array(equation_data_test.dataset[equation_name]["values"]))
    #split_point = int(5 / 6 * len(test_data))
    split_point = int(1 / 6 * len(test_data))
    y_test, y_test_val = test_data[:split_point], test_data[split_point:]
    model = NODEForecast(test_data.shape[-1], tau_opt)
    time_start = time.perf_counter()
    model.fit(y_test, niters=500)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    #y_test_pred_val = model.predict(200)

    time_start = time.perf_counter()
    y_test_pred_val = model.predict(len(y_test_val))
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)

    all_results[equation_name]["NODE"] = dict()
    all_results[equation_name]["NODE"]["Train time"] = fit_time
    all_results[equation_name]["NODE"]["Inference time"] = predict_time




    ## run esn benchmarks
    leak_rates = np.arange(1, 12) * 0.1
    all_scores = list()
    for leak_rate in leak_rates:
        try:
            model = ESNForecast(train_data.shape[-1], leak_rate=leak_rate, random_state=0)
            model.fit(y_train)
            sol_pred = model.predict(200)
        except AssertionError:
            print("Integration error encountered, skipping this entry for now")
            continue
        score_val = score_func(sol_test, sol_pred)
        all_scores.append(score_val)

    leak_rate_opt = leak_rates[np.argmin(all_scores)]
    print(leak_rate_opt)

    test_data = np.copy(np.array(equation_data_test.dataset[equation_name]["values"]))
    split_point = int(5 / 6 * len(test_data))
    y_test, y_test_val = test_data[:split_point], test_data[split_point:]

    model = ESNForecast(test_data.shape[-1], leak_rate=leak_rate_opt, random_state=SEED)
    model.fit(y_test)
    y_test_pred_val = model.predict(200)
    score_val = score_func(y_test_val, y_test_pred_val)