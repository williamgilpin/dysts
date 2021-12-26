import collections
import os
import warnings

import darts
import numpy as np
import pandas as pd
from darts import TimeSeries

from benchmarks.results.read_results import ResultsObject
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
    print('MSE on simple sequence 0 ... 1000 ', score)
    if score > 10:
        warnings.warn(f'Predicting very simple sequence, check if training/predicting is correct. '
                      f'MSE is {score}, anything above 100 is a likely error for the sequence 1000 ... 1200')


def eval_single_dyn_syst(model, dataset):
    cwd = os.path.dirname(os.path.realpath(__file__))
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
        'mse',
        'r2_score',
        'rmse',
        'smape'
    ]
    equation_name = load_file(input_path).dataset[dataset]
    model_name = 'RC-CHAOS-ESN_DEBUG_DEFAULT'
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)

    train_data = np.copy(np.array(equation_name["values"]))

    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

    try:
        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val))
    except Exception as e:
        warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
        return np.inf
        failed_combinations[model_name].append(equation_name)
    pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
    true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

    print('-----', dataset, y_train_ts.values().shape)
    value = None
    for metric_name in metric_list:
        metric_func = getattr(darts.metrics.metrics, metric_name)
        score = metric_func(true_y, pred_y)
        print(metric_name, score)
        if metric_name == METRIC:
            value = score
            results.update_results(dataset, model_name, score)
    return value

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
    model_name = 'RC-CHAOS-ESN_DEBUG_DEFAULT'
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)
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
            if metric_name == METRIC:
                results.update_results(equation_name, model_name, score)

        # TODO: print ranking relative to others for that dynamical system
    print('Failed combinations', failed_combinations)
    results.get_average_rank(model_name, print_out=True)
