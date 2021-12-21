import collections
import json
import os
import warnings

import darts
import numpy as np
import pandas as pd
from darts import TimeSeries

from dysts.datasets import load_file


def eval_simple(model):
    train_data = np.arange(1200)
    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
    print('-----Evaluating on simple sequence 0 1 2 3 ...', y_train_ts.values().shape)

    try:
        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val))
    except Exception as e:
        raise e
    pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
    true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))


    metric_func = getattr(darts.metrics.metrics, 'mse')
    score = metric_func(true_y, pred_y)
    print('MSE', score)
    if score > 10:
        warnings.warn(f'Predicting very simple sequence, check if training/predicting is correct. '
                      f'MSE is {score}, anything above 100 is a likely error for the sequence 1000 ... 1200')


def eval_all_dyn_syst(model):
    cwd = os.path.dirname(os.path.realpath(__file__))
    # cwd = os.getcwd()
    input_path = os.path.dirname(cwd) + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
    dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
    output_path = cwd + "/results/results_" + dataname + ".json"
    dataname = dataname.replace("test", "train")
    hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"
    metric_list = [
        'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        # 'mase', # requires scaling with train partition; difficult to report accurately
        'mse',
        # 'ope', # runs into issues with zero handling
        'r2_score',
        'rmse',
        # 'rmsle', # requires positive only time series
        'smape'
    ]
    equation_data = load_file(input_path)
    model_name = 'LiESN_DEBUG_DEFAULT'
    failed_combinations = collections.defaultdict(list)
    for equation_name in equation_data.dataset:

        train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

        split_point = int(5 / 6 * len(train_data))
        y_train, y_val = train_data[:split_point], train_data[split_point:]
        y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

        try:
            model.fit(y_train_ts)
            y_val_pred = model.predict(len(y_val))
        except Exception as e:
            warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
            failed_combinations[model_name].append(equation_name)
            continue
        pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

        print('-----', equation_name, y_train_ts.values().shape)
        for metric_name in metric_list:
            metric_func = getattr(darts.metrics.metrics, metric_name)
            score = metric_func(true_y, pred_y)
            print(metric_name, score)
        # TODO: print ranking relative to others for that dynamical system
        print()
    print('Failed combinations', failed_combinations)