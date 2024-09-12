#!/usr/bin/python

# import torch
import json
import os

import numpy as np
from resources.node import NODEForecast

from dysts.datasets import load_file

# import torch.optim as optim
# from torchdiffeq import odeint
from dysts.systems import get_attractor_list

SEED = 0
LONG_TEST = False
niters = 500

cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path_train = (
    os.path.dirname(cwd)
    + "/dysts/data/train_multivariate__pts_per_period_100__periods_12.json"
)
input_path_test = (
    os.path.dirname(cwd)
    + "/dysts/data/test_multivariate__pts_per_period_100__periods_12.json"
)
output_path = cwd + "/results/results_neural_ode_multivariate.json"

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

tau_vals = np.array([2, 5, 15, 30, 60, 120])
all_best_tau = dict()
for equation_name in get_attractor_list():
    print(equation_name, flush=True)
    if equation_name in all_results.keys():
        print("Skipped")
        continue

    train_data = np.copy(np.array(equation_data_train.dataset[equation_name]["values"]))

    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()

    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]

    t_train, sol_train = np.arange(len(y_train)), y_train
    t_test, sol_test = np.arange(len(y_val)), y_val

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

    test_data = np.copy(np.array(equation_data_test.dataset[equation_name]["values"]))
    split_point = int(5 / 6 * len(test_data))
    if LONG_TEST:
        split_point = int(1 / 6 * len(test_data))
    y_test, y_test_val = test_data[:split_point], test_data[split_point:]

    model = NODEForecast(test_data.shape[-1], tau_opt, random_state=SEED)
    model.fit(y_test, niters=500)
    y_test_pred_val = model.predict(200)
    score_val = score_func(y_test_val, y_test_pred_val)

    all_results[equation_name] = dict()
    all_results[equation_name]["tau_val"] = float(tau_opt)
    all_results[equation_name]["smape"] = score_val
    all_results[equation_name]["traj_true"] = y_test_val.tolist()
    all_results[equation_name]["traj_pred"] = y_test_pred_val.tolist()

    print(equation_name, score_val, flush=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4, sort_keys=True)
