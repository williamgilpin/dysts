#!/usr/bin/python3

import json
import pandas as pd
import numpy as np

import pysindy as ps
from dsr import DeepSymbolicRegressor
from pysr import pysr, best_callable

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
metric_calc = lambda x0, x1 : mean_absolute_percentage_error(x0, x1, symmetric=True)

import dysts
from dysts.flows import *
from dysts.base import *
from dysts.utils import *
# from dysts.analysis import *
from dysts.analysis import sample_initial_conditions


cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = "."
# dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/symbolic_scores_sweep.json"


operator_list = [
            "cos",
            "exp",
            "sin",
            "log",
            "tanh",
            "inv(x) = 1/x",
        ]

pysr_opts = {
    "populations" : 3,  # number of workers, defaults to 20
    "niterations" : 5,
    "procs": 1, # number of threads
    "delete_tempfiles" : True,
    "verbosity": 0,
    "unary_operators" : operator_list,
    "binary_operators" : ["+", "*", "÷", "-"]
}


# get data
with open(cwd + "/resources/symb_train_test_data.json", "r") as file:
    all_train_test = json.load(file)

try:
    with open(output_path, "r") as file:
        sym_scores = json.load(file)
except FileNotFoundError:
    sym_scores = dict()

for i, equation_name in enumerate(get_attractor_list()):    
    
    if equation_name in sym_scores.keys():
        print(f"Entry for {equation_name} found, skipping it.", flush=True)
        continue
    print(equation_name, flush=True)
    
    sym_scores[equation_name] = dict()
    
    X_train = np.array(all_train_test[equation_name]["X_train"])
    y_train = np.array(all_train_test[equation_name]["y_train"])
    X_test = np.array(all_train_test[equation_name]["X_test"])
    y_test = np.array(all_train_test[equation_name]["y_test"])
    t_train = np.array(all_train_test[equation_name]["t_train"])
    t_test = np.array(all_train_test[equation_name]["t_test"])
    
#     n_train = 150
#     np.random.seed(0)
#     ic_train, ic_test = sample_initial_conditions(model, 2, traj_length=1000, pts_per_period=30)

#     model.ic = ic_train
#     tvals, sol = model.make_trajectory(n_train, pts_per_period=15, resample=True, return_times=True, standardize=False)
#     dt = np.median(np.diff(tvals))
#     dsol = np.vstack([model.rhs(val, 0) for val in sol])# * dt
#     X_train, y_train = sol, dsol

#     model.ic = ic_test
#     tvals, sol = model.make_trajectory(n_train, pts_per_period=15, resample=True, return_times=True, standardize=False)
#     dt = np.median(np.diff(tvals))
#     dsol = np.vstack([model.rhs(val, 0) for val in sol])# * dt
#     X_test, y_test = sol, dsol
    
    ndim = min([X_test.shape[-1], y_test.shape[-1]])
    
    ## SINDY-poly
    sym_model = ps.SINDy()
    sym_model.fit(X_train, t=t_train)
    y_test_pred = sym_model.predict(X_test)
    all_scores = list()
    for i in range(ndim):
        all_scores.append(metric_calc(y_test[:, i], y_test_pred[:, i]))
    sym_scores[equation_name]["SINDY-poly"] = np.median(all_scores)
    
    ## SINDY-fourier basis
    sym_model = ps.SINDy(feature_library=ps.FourierLibrary(n_frequencies=10))
    sym_model.fit(X_train, t=t_train)
    y_test_pred = sym_model.predict(X_test)
    all_scores = list()
    for i in range(ndim):
        all_scores.append(metric_calc(y_test[:, i], y_test_pred[:, i]))
    sym_scores[equation_name]["SINDY-fourier"] = np.median(all_scores)
    
    
    all_scores = list()
    for i in range(ndim):
        model = DeepSymbolicRegressor(cwd +"/resources/config.json")
        model.fit(X_train, y_train[:, i]) # Should solve in ~10 seconds
        y_test_pred = model.predict(X_test)
        all_scores.append(metric_calc(y_test[:, i], y_test_pred))
    final_score = np.median(all_scores)
    sym_scores[equation_name]["DSR"] = final_score
    
    
    all_scores = list()
    for i in range(ndim):
        models = pysr(X_train, y_train[:, i], **pysr_opts)
        y_test_pred = best_callable(models)(X_test)
        all_scores.append(metric_calc(y_test[:, i], y_test_pred))
    final_score = np.median(all_scores)
    sym_scores[equation_name]["pySR"] = final_score
    
    
    print(equation_name, final_score, flush=True)
    
    with open(output_path, 'w') as f:
        json.dump(sym_scores, f, indent=4)   
    
    

