#!/usr/bin/python

## Define a model that predicts a time series, given its previous values


import pandas as pd
from scipy.signal import savgol_filter
import json
import time

import darts 
from darts import TimeSeries
from darts.models import RNNModel

from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

import dysts
from dysts.flows import *
from dysts.base import *
from dysts.utils import *
from dysts.analysis import *




hyperparams = {
    "input_chunk_length": 50, 
    "output_chunk_length": 1, 
    "model": "LSTM", 
    "n_rnn_layers": 2, 
    "random_state": 0
}

results = dict()

# cwd = os.getcwd()
cwd = os.path.dirname(os.path.realpath(__file__))
output_path = cwd + "/results/importance_sampling.json"
print("Saving data to: ", output_path)


full_epoch_count = 400
forecast_length = 200
transient_length = 2
n_iters = 5
epoch_count = 30
n_ic = 10  # model retraining is not currently working in darts
traj_len =  150

show_progress = False

print(f"{n_ic} points sampled per iteration, with trajectory length {traj_len}, for a total of {n_iters} iterations of length {epoch_count}")
print(n_ic * traj_len * n_iters* epoch_count)
print(full_epoch_count * 1000)


for equation_ind, equation_name in enumerate(get_attractor_list()):
#     if equation_ind < 30 + 1:
#         continue
    np.random.seed(0)
    
    print(f"{equation_name} {equation_ind}", flush=True)
    results[equation_name] = dict()

    equation = getattr(dysts.flows, equation_name)()
    if hasattr(equation, "delay"):
        if equation.delay: 
            continue
    sol = equation.make_trajectory(1200, resample=True)
    y_train, y_test = sol[:-forecast_length, 0], sol[-forecast_length:, 0]
    y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))
    results[equation_name]["true_value"] = y_test.tolist()

    try:
        del model
    except:
        pass
    model = RNNModel(**hyperparams)
    
    base_t0 = time.perf_counter()
    model.fit(y_train_ts, epochs=full_epoch_count)
    base_t1 = time.perf_counter()

    y_val_pred = model.predict(forecast_length)
    y_val_pred = np.squeeze(y_val_pred.values())
    score = mean_absolute_percentage_error(y_test, y_val_pred, symmetric=True)
    base_elapsed = base_t1 - base_t0
    print(score, base_elapsed)
    plt.figure()
    plt.plot(y_test)
    plt.plot(y_val_pred)
    
    results[equation_name]["base_value"] = y_val_pred.tolist()
    results[equation_name]["base_time"] = base_elapsed
    results[equation_name]["base_score"] = score

    pred_backtest = model.historical_forecasts(y_train_ts, retrain=False, start=(1 + model.input_chunk_length)).values()
    model = RNNModel(**hyperparams)
    comp_t0 = time.perf_counter()
    for i in range(n_iters):
        print(i)
        
        if i == 0:
            ic_indices = np.random.choice(np.arange(len(y_train_ts) - (forecast_length + transient_length)), n_ic)
        else:
            pred_backtest = model.historical_forecasts(y_train_ts, retrain=False, start=(1 + model.input_chunk_length)).values()

            y_train_backtest = y_train_ts.values()[(1 + model.input_chunk_length):]
            mse_back = np.squeeze((pred_backtest - y_train_backtest)**2)
            mse_back = savgol_filter(mse_back, 51, 3)
            mse_back[mse_back<0] = 0.0
            sample_probs = mse_back**4
            sample_probs /= np.sum(sample_probs)
            ic_indices = np.random.choice(np.arange(len(y_train_backtest)), n_ic, p=sample_probs, replace=True)
            
            ## Random sampling as a control condition
            # ic_indices = np.random.choice(np.arange(len(y_train_backtest)), n_ic)
            # ic_indices = np.argsort(mse_back)[-n_ic:]

        ic_vals = sol[ic_indices] + 1e-2 * (np.random.random(sol[ic_indices].shape) - 0.5)
        equation.ic = ic_vals
        new_sol = equation.make_trajectory(traj_len + transient_length, resample=True)[:, transient_length:, :]
        y_train_list = list(new_sol[..., 0])
        y_train_list = [TimeSeries.from_dataframe(pd.DataFrame(item)) for item in y_train_list]
        model.fit(y_train_list, epochs=epoch_count)
    comp_t1 = time.perf_counter()  
    comp_elapsed = comp_t1 - comp_t0

    y_val_pred = model.predict(forecast_length, series=y_train_ts)
    y_val_pred = np.squeeze(y_val_pred.values())
    score = mean_absolute_percentage_error(y_test, y_val_pred, symmetric=True)
    print(score, comp_elapsed)
    plt.plot(y_val_pred)
    plt.show()

    results[equation_name]["importance_values"] = y_val_pred.tolist()
    results[equation_name]["importance_time"] = comp_elapsed
    results[equation_name]["importance_score"] = score

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)



for equation_ind, equation_name in enumerate(get_attractor_list()):
    np.random.seed(0)
    
    print(f"{equation_name} {equation_ind}", flush=True)

    equation = getattr(dysts.flows, equation_name)()
    if hasattr(equation, "delay"):
        if equation.delay: 
            continue
    sol = equation.make_trajectory(1200, resample=True)
    y_train, y_test = sol[:-forecast_length, 0], sol[-forecast_length:, 0]
    y_train_ts = TimeSeries.from_dataframe(pd.DataFrame(y_train))

    model = RNNModel(**hyperparams)

    comp_t0 = time.perf_counter()
    for i in range(n_iters):
        print(i)
        ic_indices = np.random.choice(np.arange(len(y_train_ts) - (forecast_length + transient_length)), n_ic)
        ic_vals = sol[ic_indices] + 1e-2 * (np.random.random(sol[ic_indices].shape) - 0.5)
        equation.ic = ic_vals
        new_sol = equation.make_trajectory(traj_len + transient_length, resample=True)[:, transient_length:, :]
        y_train_list = list(new_sol[..., 0])
        y_train_list = [TimeSeries.from_dataframe(pd.DataFrame(item)) for item in y_train_list]
        model.fit(y_train_list, epochs=epoch_count)
    comp_t1 = time.perf_counter()  
    comp_elapsed = comp_t1 - comp_t0

    y_val_pred = model.predict(forecast_length, series=y_train_ts)
    y_val_pred = np.squeeze(y_val_pred.values())
    score = mean_absolute_percentage_error(y_test, y_val_pred, symmetric=True)
    print(score, comp_elapsed)
#     plt.plot(y_val_pred)
#     plt.show()

    results[equation_name]["random_values"] = y_val_pred.tolist()
    results[equation_name]["random_time"] = comp_elapsed
    results[equation_name]["random_score"] = score
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)   