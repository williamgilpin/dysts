from typing import Union, Sequence, Optional

import echotorch
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import RegressionModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from echotorch.utils import MatrixGenerator
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from echotorch.nn import LiESN

from models.utils import eval_simple, eval_all_dyn_syst
from rc_chaos.Methods.RUN import getModel, get_args_dict


class LiESNRegressor(LiESN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, u, y=None, **kwargs):
        super().forward(torch.Tensor(u), torch.Tensor([[y]]))
        #super().forward(u, y)
        self.finalize()

    def predict(self, u, **kwargs):
        return super().forward(u)

class LiESNRegressorFitter(RegressionModel):
    def __init__(self, lags=1, **kwargs):
        super().__init__(lags = lags)
        if len(kwargs) == 0:
            self.model = get_default('regressor')
        else:
            self.model = LiESNRegressor(**kwargs)

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
        #

        train_dataset = TensorDataset(input, target) # create your datset
        trainloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2) # create your dataloader
        for data in trainloader:

            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)

            #if use_cuda:
            # inputs, targets = inputs.cuda(), targets.cuda()
            #esn.cuda()

            self(inputs, targets)
        self.finalize()
        return
        #
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
        #
        input = data[:-1].squeeze(-1)
        target = data[1:].squeeze(-1)

        train_dataset = TensorDataset(input, target) # create your datset
        testloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2) # create your dataloader

        dataiter = iter(testloader)
        test_u, test_y = dataiter.next()
        test_u, test_y = Variable(test_u), Variable(test_y)
        y_predicted_test = self(test_u)
        #if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
        print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted_test.data, test_y.data)))
        return y_predicted_test
        #
        out = self(data)
        print('_____S', out.shape)
        ret =  TimeSeries.from_dataframe(pd.DataFrame(out))    # TODO gets casted to values and then TimeSeries again...
        print('___R', ret.values().shape)
        return ret#[:n] # only return first n values
        # TODO: likely error here, also get warnings


def get_default(type='normal'):
    # Hyperparameters
    n_train_samples = 1
    n_test_samples = 1
    spectral_radius = 0.9
    leaky_rate = 1.0
    input_dim = 1
    n_hidden = 100

    if type == 'normal':
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
    elif type == 'regressor':
        return LiESNRegressor(
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
    eval_simple(get_default())
    eval_all_dyn_syst(get_default())
    #eval_simple(getModel(get_args_dict()))
    #eval_all_dyn_syst(getModel(get_args_dict()))


if __name__ == '__main__':
    main()
