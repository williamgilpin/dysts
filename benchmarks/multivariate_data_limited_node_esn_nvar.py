#!/usr/bin/python

import sys
import os
import json

import dysts
from dysts.datasets import *

import numpy as np

from resources.esn import ESNForecast, NVARForecast
from resources.node import NODEForecast

import dysts.metrics
from dysts.metrics import smape
score_func = smape

SEED = 0
LONG_TEST = False
cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"
# input_path = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_60.json.gz"

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/data_sweep_" + dataname + "_node_esn_nvar.json"
dataname = dataname.replace("test", "train" )

equation_data = load_file(input_path)

equation_names = list(equation_data.dataset.keys())

cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path_train = os.path.dirname(cwd)  + "/dysts/data/train_multivariate__pts_per_period_100__periods_12.json.gz"
input_path_test = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"
if LONG_TEST:
    input_path_test = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_60.json.gz"

equation_data_train = load_file(input_path_train)
equation_data_test = load_file(input_path_test)


try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
# for equation_name in equation_data.dataset:
#     for model_name in ["ESN", "NVAR"]:
#         if model_name not in all_results[equation_name].keys():
#             continue
#         if len(all_results[equation_name][model_name]) == 0:
#             all_results[equation_name].pop(model_name, None)



for equation_name in equation_names:


    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    else:
        print(f"{equation_name} seen before.", flush=True) 
    
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))
    split_point = int(5 / 6 * len(train_data)) # long horizon
    max_lag_val = 100 # need at least one full lookback window to fit a model
    truncate_inds = np.unique(np.linspace(0, split_point - max_lag_val - 1, 25).astype(int))
    

    ### ESN
    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]

    t_train, sol_train = np.arange(len(y_train)), y_train
    t_test, sol_test = np.arange(len(y_val)), y_val

    ## Inference
    test_data = np.copy(np.array(equation_data_test.dataset[equation_name]["values"]))
    split_point = int(5 / 6 * len(test_data))
    if LONG_TEST:
        split_point = int(1 / 6 * len(test_data))
    y_test, y_test_val = test_data[:split_point], test_data[split_point:]



    # model_name = "ESN"
    # leak_rates = np.arange(1, 12) * 0.1
    # if model_name not in all_results[equation_name].keys():
    #     all_results[equation_name][model_name] = list()
    # elif len(all_results[equation_name][model_name]) >= len(truncate_inds):
    #     print(f"Skipping {model_name} for {equation_name}", flush=True)
    #     continue
    # else:
    #     ## reset count
    #     all_results[equation_name][model_name] = list()

    # ## Validation
    # all_scores = list()
    # for leak_rate in leak_rates:
    #     try:
    #         model = ESNForecast(train_data.shape[-1], leak_rate=leak_rate, random_state=0)
    #         model.fit(y_train)
    #         sol_pred = model.predict(200)
    #     except AssertionError:
    #         print("Integration error encountered, skipping this entry for now")
    #         continue
    #     score_val = score_func(sol_test, sol_pred)
    #     all_scores.append(score_val)
    # leak_rate_opt = leak_rates[np.argmin(all_scores)]
    # print(leak_rate_opt)

    # for i in truncate_inds:
    #     y_train, y_val = test_data[i:split_point], test_data[split_point:]
    #     model = ESNForecast(test_data.shape[-1], leak_rate=leak_rate_opt, random_state=SEED)
    #     model.fit(y_train)

    #     # print(y_val.shape, test_data.shape, flush=True)
    #     ## Attempt to predict the validation data. If it fails, then the model returns
    #     ## a None value for the prediction.
    #     try:
    #         y_val_pred = model.predict(len(y_val)).squeeze()
    #     except:
    #         print(f"Failed to predict {equation_name} {model_name}")
    #         y_val_pred = np.array([None] * len(y_val))
    #     #all_results[equation_name][model_name]["prediction"] = y_val_pred.tolist()

    #     ## Attempt to compute several time series distance metrics. If it fails due
    #     ## to a ValueError, then all metrics are not defined for the given time series.
    #     try:
    #         score = dysts.metrics.smape(y_val, y_val_pred)
    #     except ValueError:
    #         score = None

    #     all_results[equation_name][model_name].append(score)

    
    #     print(equation_name + " " + model_name, flush=True)
    #     ## Update results dictionary with metrics, then save to JSON file.
    #     with open(output_path, 'w') as f:
    #         json.dump(all_results, f, indent=4)   



    
    # model_name = "NVAR"
    # tau_vals = np.array([2, 5, 15, 30, 60, 120])
    # if model_name not in all_results[equation_name].keys():
    #     all_results[equation_name][model_name] = list()
    # elif len(all_results[equation_name][model_name]) >= len(truncate_inds):
    #     print(f"Skipping {model_name} for {equation_name}", flush=True)
    #     continue
    # else:
    #     ## reset count
    #     all_results[equation_name][model_name] = list()
    # ## Validation
    # all_scores = list()
    # for tau in tau_vals:
    #     try:
    #         model = NVARForecast(train_data.shape[-1], delay=tau, random_state=0)
    #         model.fit(y_train)
    #         sol_pred = model.predict(200)
    #     except AssertionError:
    #         print("Integration error encountered, skipping this entry for now")
    #         continue
    #     score_val = score_func(sol_test, sol_pred)
    #     all_scores.append(score_val)
    # tau_opt = tau_vals[np.argmin(all_scores)]
    # print(tau_opt)

    # for i in truncate_inds:
    #     y_train, y_val = test_data[i:split_point], test_data[split_point:]
    #     model = NVARForecast(test_data.shape[-1], delay=tau_opt, random_state=SEED)
    #     model.fit(y_train)

    #     # print(y_val.shape, test_data.shape, flush=True)
    #     ## Attempt to predict the validation data. If it fails, then the model returns
    #     ## a None value for the prediction.
    #     try:
    #         y_val_pred = model.predict(len(y_val)).squeeze()
    #     except:
    #         print(f"Failed to predict {equation_name} {model_name}")
    #         y_val_pred = np.array([None] * len(y_val))
    #     #all_results[equation_name][model_name]["prediction"] = y_val_pred.tolist()

    #     ## Attempt to compute several time series distance metrics. If it fails due
    #     ## to a ValueError, then all metrics are not defined for the given time series.
    #     try:
    #         score = dysts.metrics.smape(y_val, y_val_pred)
    #     except ValueError:
    #         score = None

    #     all_results[equation_name][model_name].append(score)

    
    #     print(equation_name + " " + model_name, flush=True)
    #     ## Update results dictionary with metrics, then save to JSON file.
    #     with open(output_path, 'w') as f:
    #         json.dump(all_results, f, indent=4)   


    model_name = "NODE"
    tau_vals = np.array([2, 5, 15, 30, 60, 120])
    if model_name not in all_results[equation_name].keys():
        all_results[equation_name][model_name] = list()
    elif len(all_results[equation_name][model_name]) >= len(truncate_inds):
        print(f"Skipping {model_name} for {equation_name}", flush=True)
        continue
    else:
        ## reset count
        all_results[equation_name][model_name] = list()
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
    print(tau_opt)

    for i in truncate_inds:
        y_train, y_val = test_data[i:split_point], test_data[split_point:]
        model = NODEForecast(test_data.shape[-1], tau_opt, random_state=SEED)
        model.fit(y_train, niters=500)

        ## Attempt to predict the validation data. If it fails, then the model returns
        ## a None value for the prediction.
        try:
            y_val_pred = model.predict(len(y_val)).squeeze()
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