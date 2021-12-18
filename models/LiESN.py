import collections
import json
import os
from typing import Union, Sequence, Optional
import warnings

import darts
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from echotorch.utils import MatrixGenerator

from dysts.datasets import load_file
from echotorch.nn import LiESN


# TODO: inherit from base classes of RNNModel
# currently only using GlobalForecastingModel as we need gridsearch for it
class LiESNFitter(LiESN, GlobalForecastingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
            ) -> None:
        # input: TimeSeries DataArray
        super().fit(series)
        data = torch.Tensor(np.array(series.all_values()))
        input = data[:-1]
        target = data[1:]
        self(input, target)

        # TODO: more sohpisticated ways of lag, e.g. see
        ''' RegressionModel()
        training_samples, training_labels = self._create_lagged_data(
            series, past_covariates, future_covariates, None)

        # if training_labels is of shape (n_samples, 1) we flatten it to have shape (n_samples,)
        if len(training_labels.shape) == 2 and training_labels.shape[1] == 1:
            training_labels = training_labels.ravel()
        self(training_samples, training_labels)
        '''

        self.finalize()

    # TODO: enable predicting on trained and new series
    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                num_samples: int = 1,
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # only passing n (len of y_val) and continuing on from trained vals (in compute_benchmarks.py)
        if series is None:
            series = self.training_series

        data = torch.Tensor(np.array(series.all_values()))
        return self(data)


def get_default():
    # Hyperparameters
    n_train_samples = 1
    n_test_samples = 1
    spectral_radius = 0.9
    leaky_rate = 1.0
    input_dim = 1
    n_hidden = 100

    return LiESNFitter(
        input_dim=input_dim,
        hidden_dim=n_hidden,
        output_dim=1,
        spectral_radius=spectral_radius,
        learning_algo='inv',
        leaky_rate=leaky_rate,
        w_generator=MatrixGenerator(),
        win_generator=MatrixGenerator(),
        wbias_generator=MatrixGenerator()
    )


def main():
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

    try:
        with open(output_path, "r") as file:
            all_results = json.load(file)
    except FileNotFoundError:
        all_results = dict()

    model_name = 'LiESN_DEBUG_DEFAULT'
    failed_combinations = collections.defaultdict(list)
    for equation_name in equation_data.dataset:

        train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

        if equation_name not in all_results.keys():
            all_results[equation_name] = dict()
        all_results[equation_name][model_name] = dict()

        split_point = int(5 / 6 * len(train_data))
        y_train, y_val = train_data[:split_point], train_data[split_point:]
        y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
        print('-----', equation_name, y_train_ts.values().shape)

        model = get_default()

        try:
            model.fit(y_train_ts)
            y_val_pred = model.predict(len(y_val))
        except Exception as e:
            warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
            failed_combinations[model_name].append(equation_name)
            continue
        # TODO  if on GPU: y_val_pred.values()
        # pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
        # true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

        # all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred.values()).tolist()

        pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred)))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

        all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred).tolist()

        for metric_name in metric_list:
            metric_func = getattr(darts.metrics.metrics, metric_name)
            score = metric_func(true_y, pred_y)
            print(metric_name, score)
            all_results[equation_name][model_name][metric_name] = score
        print()

    print(failed_combinations)


if __name__ == '__main__':
    main()
