#!/usr/bin/python

import os
import numpy as np
# import torch
import json
import time

# import torch.optim as optim
# from torchdiffeq import odeint

import pandas as pd

import dysts
import dysts.flows
from dysts.base import get_attractor_list
from dysts.datasets import load_file, convert_json_to_gzip

import darts
from darts.models import *
from darts import TimeSeries
import darts.models

from resources.univariate_strided import MultivariateForecast

SEED = 0
LONG_TEST = True
niters = 500

cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path_train = os.path.dirname(cwd)  + "/dysts/data/train_multivariate__pts_per_period_100__periods_12.json.gz"
input_path_test = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json.gz"
if LONG_TEST:
    input_path_test = os.path.dirname(cwd)  + "/dysts/data/test_multivariate__pts_per_period_100__periods_60.json.gz"
output_path = cwd + "/results/results_classical_multivariate2.json"

equation_data_train = load_file(input_path_train)
equation_data_test = load_file(input_path_test)



## Load results
try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
        print("Existing database found")
except FileNotFoundError:
    all_results = dict()

from dysts.metrics import smape
score_func = smape


time_delays = [3, 5, 10, 25]
season_values = [darts.utils.utils.SeasonalityMode.ADDITIVE, 
                 darts.utils.utils.SeasonalityMode.NONE
                ]
all_best_hparam = dict()
parameter_candidates = dict()
parameter_candidates["ARIMA"] = {"p": time_delays}
parameter_candidates["ExponentialSmoothing"] = {"seasonal": season_values}
parameter_candidates["FourTheta"] = {"season_mode": season_values}
parameter_candidates["Theta"] = {"season_mode": season_values}
for model_name in ["AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", "NaiveSeasonal", "TBATS"]:
    parameter_candidates[model_name] = {}

for equation_name in get_attractor_list():

    print(equation_name, flush=True)
    if equation_name in all_results.keys():
        print(f"Skipped {equation_name}")
        continue
    
    all_results[equation_name] = dict()

    for model_name in [ "ExponentialSmoothing", "ARIMA","FourTheta", 
                    "Theta", "AutoARIMA", "FFT", "NaiveDrift", "NaiveMean", 
                    "NaiveSeasonal", "TBATS"]:
        model_instance = getattr(darts.models, model_name)

        print(model_name, flush=True)
        if model_name in all_results[equation_name].keys():
            print(f"Skipped {model_name}")
            continue
        all_results[equation_name][model_name] = dict()

        train_data = np.copy(np.array(equation_data_train.dataset[equation_name]["values"]))

        if equation_name not in all_results.keys():
            all_results[equation_name] = dict()
        
        split_point = int(5 / 6 * len(train_data))
        y_train, y_val = train_data[:split_point], train_data[split_point:]

        t_train, sol_train = np.arange(len(y_train)), y_train
        t_test, sol_test = np.arange(len(y_val)), y_val

        train_ts = TimeSeries.from_values(y_train)
        test_ts = TimeSeries.from_values(y_val)

        if len(parameter_candidates[model_name]) > 0:
            all_scores = list()
            hkey = list(parameter_candidates[model_name].keys())[0]
            print(hkey, flush=True)
            hparam_list = parameter_candidates[model_name][hkey]
            for hparam in hparam_list:
                try:
                    model = MultivariateForecast(model_instance)
                    model.fit(train_ts, n_jobs=1, **{hkey: hparam})
                    pred_ts = model.predict(len(y_val))
                    sol_pred = pred_ts.values()
                    score_val = score_func(sol_test, sol_pred)
                    all_scores.append(score_val)
                except:
                    print("Error encountered, skipping this entry")
                    all_scores.append(np.inf)
                    continue
            hparam_opt = hparam_list[np.argmin(all_scores)]
            hyperparams = {hkey: hparam_opt}
            print(hparam_opt, flush=True)
        else:
            hyperparams = {}
            hparam_opt = None
        

        test_data = np.copy(np.array(equation_data_test.dataset[equation_name]["values"]))
        split_point = int(5 / 6 * len(test_data))
        if LONG_TEST:
            split_point = int(1 / 6 * len(test_data))
        y_test, y_test_val = test_data[:split_point], test_data[split_point:]
        y_test_ts = TimeSeries.from_values(y_test)

        model = MultivariateForecast(model_instance)
        try:
            time_start = time.perf_counter()
            model.fit(y_test_ts, n_jobs=1, **hyperparams)
            time_end = time.perf_counter()
            fit_time = str(time_end - time_start)
            
            time_start = time.perf_counter()
            _ = model.predict(200)
            time_end = time.perf_counter()
            predict_time = str(time_end - time_start)

            y_test_pred_val = model.predict(len(y_test_val))
            y_test_pred_val = y_test_pred_val.values()
            score_val = score_func(y_test_val, y_test_pred_val)
        except:
            score_val = None
            y_test_pred_val = np.array([None] * len(y_test_val))
            fit_time = None
            predict_time = None

        ## For SeasonalityModes, convert to a serializable string
        if hasattr(hparam_opt, "name"):
            hparam_opt = hparam_opt.name
        
        all_results[equation_name][model_name]["hparam_val"] = hparam_opt
        all_results[equation_name][model_name]["smape"] = score_val
        all_results[equation_name][model_name]["traj_true"] =  y_test_val.tolist()
        all_results[equation_name][model_name]["traj_pred"] =  y_test_pred_val.tolist()
        all_results[equation_name][model_name]["Train time"] = fit_time
        all_results[equation_name][model_name]["Inference time"] = predict_time

        print(equation_name, score_val, flush=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4, sort_keys=True)   
            

convert_json_to_gzip(output_path)
