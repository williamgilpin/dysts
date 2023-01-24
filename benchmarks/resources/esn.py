import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, NVAR

from reservoirpy import verbosity

verbosity(0)

class ESNForecast:
    """
    A wrapper class for an echo-state network time series forecasting model.

    Parameters:
        n_dim (int): the dimensionality of the time series
        n_units (int): the number of units in the reservoir
        spectral_radius (float): the spectral radius of the reservoir
        leak_rate (float): the leak rate of the reservoir
        connectivity (float): the connectivity of the reservoir
        input_scaling (float): the input scaling of the reservoir
        input_connectivity (float): the input connectivity of the reservoir
        regularization (float): the regularization parameter for the ridge regression of
            the readout layer
        random_state (int): the random seed

    Methods:
        fit(X, niters=500, learning_rate=1e-2, batch_size=128): fit the model to a dataset
        predict(nt, ic=None, t=None): predict the future time series values

    Example:
        model = ESNForecast(train_data.shape[-1], 30)
        model.fit(train_data)
        sol_pred = model.predict(100)
    """
    def __init__(
        self,
        n_dim,
        n_units=500, 
        spectral_radius=0.99,
        leak_rate=0.3,
        connectivity=0.1,
        input_scaling=1.0, 
        input_connectivity=0.2,
        regularization=1e-4,
        random_state=None
    ):  
        self.n_dim = n_dim
        self.n_units = n_units
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.connectivity = connectivity
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.regularization = regularization
        self.random_state = random_state

        self._init_model()

    def _init_model(self):
        """Initialize the internal reservoir model"""
        reservoir = Reservoir(
            self.n_units, 
            input_scaling=self.input_scaling,
            sr=self.spectral_radius,
            lr=self.leak_rate,
            rc_connectivity=self.connectivity,
            input_connectivity=self.input_connectivity,
            seed=self.random_state
        )
        readout = Ridge(self.n_dim, ridge=self.regularization)
        self.model = reservoir >> readout

        
        
    def fit(self, X):
        """
        Fit the model to a dataset

        Args:
            X (np.ndarray): the time series data to fit the model to

        Returns:
            self (ESNForecast): the fitted model
        """
        X_train, y_train = X[:-1], X[1:]
        warmup = min(100, int(0.1 * X_train.shape[0]))
        self.model.fit(X_train, y_train, warmup=warmup)
        self.ic = X[-1]
        return self

    def predict(self, nt, ic=None, t=None):
        """
        Predict future time series values, not including the initial condition, using
        1-step autoregressive prediction.

        Args:
            nt (int): the number of timepoints to predict
            ic (np.ndarray): the initial condition for the ODE
            t (np.ndarray): the timepoints to use for the ODE

        Returns:
            sol_pred (np.ndarray): the predicted time series values
        """
        if ic is None:
            ic = self.ic

        curr = ic
        sol_pred = []
        for i in range(nt):
            curr = self.model.run(curr)
            sol_pred.append(curr)
        sol_pred = np.array(sol_pred).squeeze()
        return sol_pred


class NVARForecast(ESNForecast):
    """
    A wrapper class for a non-linear vector autoregressive time series forecasting model.

    Parameters:
        n_dim (int): the dimensionality of the time series
        delay (int): the delay of the model
        order (int): the order of the model
        strides (int): the strides of the model
        regularization (float): the regularization parameter for the ridge regression of
            the readout layer
        random_state (int): the random seed

    Methods:
        fit(X, niters=500, learning_rate=1e-2, batch_size=128): fit the model to a dataset
        predict(nt, ic=None, t=None): predict the future time series values

    Example:
        model = NVARForecast(train_data.shape[-1], 30)
        model.fit(train_data)
        sol_pred = model.predict(100)
    """

    def __init__(self, ndim, delay=100, order=1, strides=1, regularization=1e-4, random_state=None):
        self.ndim = ndim
        self.delay = delay
        self.order = order
        self.strides = strides
        self.regularization = regularization
        self.random_state = random_state

        super().__init__(ndim, regularization=regularization, random_state=random_state)
        self._init_model()
    
    def _init_model(self):
        nvar = NVAR(delay=self.delay, order=self.order, strides=self.strides, seed=self.random_state)
        readout = Ridge(self.n_dim, ridge=self.regularization)
        self.model = nvar >> readout