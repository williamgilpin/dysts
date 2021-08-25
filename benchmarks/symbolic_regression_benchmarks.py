#!/usr/bin/python3

import json
import pandas as pd
import numpy as np
import time

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
output_path = cwd + "/results/symbolic_scores_sweep2.json"

operator_list = [
            "cos",
            "exp",
            "sin",
            "log",
            "tanh",
            "inv(x) = 1/x",
        ]

NPROCS = 24
pysr_opts = {
    "populations" : 2 * NPROCS,
    "niterations" : 5,
#    "runtests" : False,
    "procs": NPROCS, # number of threads
#    "procs": 0,
    "delete_tempfiles" : True,
    "verbosity": 0,
    "unary_operators" : operator_list,
#    "multithreading" : True,
    "binary_operators" : ["+", "*"]
}

# get data
with open(cwd + "/resources/symb_train_test_data.json", "r") as file:
    all_train_test = json.load(file)

try:
    with open(output_path, "r") as file:
        sym_scores = json.load(file)
except FileNotFoundError:
    sym_scores = dict()
    with open(output_path, 'w') as f:
        json.dump(sym_scores, f, indent=4)

# 0, 50, 80, 100, -1
for i, equation_name in enumerate(get_attractor_list()[::-1][67:]):

#    with open(output_path, "r") as file:
#        sym_scores = json.load(file)
        
    with open(output_path, "r") as file:
        sym_scores.update(json.load(file))

    if equation_name in sym_scores.keys():
        print(f"Entry for {equation_name} found, skipping it.", flush=True)
        continue
    print(str(i) + " ", equation_name, flush=True)

    curr_scores = dict()
    curr_scores[equation_name] = dict()
    
    X_train = np.array(all_train_test[equation_name]["X_train"])
    y_train = np.array(all_train_test[equation_name]["y_train"])
    X_test = np.array(all_train_test[equation_name]["X_test"])
    y_test = np.array(all_train_test[equation_name]["y_test"])
    t_train = np.array(all_train_test[equation_name]["t_train"])
    t_test = np.array(all_train_test[equation_name]["t_test"])
    
    ndim = min([X_test.shape[-1], y_test.shape[-1]])
    
    ## SINDY-poly
   t0 = time.perf_counter()
   sym_model = ps.SINDy()
   sym_model.fit(X_train, t=t_train)
   y_test_pred = sym_model.predict(X_test)
   all_scores = list()
   for i in range(ndim):
       all_scores.append(metric_calc(y_test[:, i], y_test_pred[:, i]))
   curr_scores[equation_name]["SINDY-poly"] = np.median(all_scores)
   t1 = time.perf_counter()
   elapsed = t1 - t0
   curr_scores[equation_name]["SINDY-poly-time"] = elapsed

    ## SINDY-fourier basis
   t0 = time.perf_counter()
   sym_model = ps.SINDy(feature_library=ps.FourierLibrary(n_frequencies=10))
   sym_model.fit(X_train, t=t_train)
   y_test_pred = sym_model.predict(X_test)
   all_scores = list()
   for i in range(ndim):
       all_scores.append(metric_calc(y_test[:, i], y_test_pred[:, i]))
   curr_scores[equation_name]["SINDY-fourier"] = np.median(all_scores)
   t1 = time.perf_counter()
   elapsed = t1 - t0
   curr_scores[equation_name]["SINDY-fourier-time"] = elapsed
    
    ## DSR
    t0 = time.perf_counter()
    all_scores = list()
    for i in range(ndim):
        try:
            model = DeepSymbolicRegressor(cwd + "/resources/config.json")
            model.fit(X_train, y_train[:, i])
            y_test_pred = model.predict(X_test)
            all_scores.append(metric_calc(y_test[:, i], y_test_pred))
            print("iter complete", flush=True)
        except:
            print("bad iteration", flush=True)
    try:
        final_score = np.median(all_scores)
    except:
        final_score = None
    curr_scores[equation_name]["DSR"] = final_score
    t1 = time.perf_counter()
    elapsed = t1 - t0
    curr_scores[equation_name]["DSR-time"] = elapsed

    ## pySR
   t0 = time.perf_counter() 
   all_scores = list()
   for i in range(ndim):
       try:
           models = pysr(X_train, y_train[:, i], **pysr_opts)
           y_test_pred = best_callable(models)(X_test)
           all_scores.append(metric_calc(y_test[:, i], y_test_pred))
           print("iter complete", flush=True)
       except:
           print("bad iteration", flush=True)
   try:
       final_score = np.median(all_scores)
   except:
       final_score = None
   curr_scores[equation_name]["pySR"] = final_score
   t1 = time.perf_counter()
   elapsed = t1 - t0
   curr_scores[equation_name]["pySR-time"] = elapsed 

    print(equation_name, final_score, flush=True)

    with open(output_path, "r") as file:
        sym_scores.update(json.load(file))

    sym_scores.update(curr_scores)

    with open(output_path, 'w') as f:
        json.dump(sym_scores, f, indent=4, sort_keys=True)   
    
    

