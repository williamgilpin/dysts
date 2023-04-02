import numpy as np

import darts
from darts import TimeSeries

class MultivariateForecast():
    """MultivariateForecast class.

    A wrapper for univariate forecasting models that can be used to forecast
    multivariate time series. This class trains a separate model for each
    component of the multivariate time series.

    Parameters:
        model: forecasting model to be used for each component.

    Attributes:
        models_ (list): list of models, one for each component.
        n_components (int): number of components of the multivariate time series.
    """
    def __init__(self, model):
        self.model = model

    def fit(self, ts, **kwargs):
        self.n_components = ts.width
        self.models_ = []
        for i in range(self.n_components):
            self.models_.append(self.model(**kwargs))
            self.models_[i].fit(ts.univariate_component(i))

    def predict(self, n):
        return TimeSeries.from_values(
            np.array([model.predict(n).values() for model in self.models_]).transpose(1, 0, 2),
        )
    
    def __str__(self):
        return f"MultivariateForecast({self.model.__name__})"
