import numpy as np


import darts
from darts import TimeSeries
# from darts.models.forecasting.forecasting_model import GlobalForecastingModel, ForecastingModel


## wrap forecast models that operate only on univariate time series to separately
## forecast each component of a multivariate time series. Train a separate model
## for each component of the multivariate time series.

class MultivariateForecast():
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
