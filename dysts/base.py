"""Dynamical systems in Python

(M, T, D) or (T, D) convention for outputs

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)

"""

import gzip
import json
import os
import warnings
from dataclasses import dataclass, field
from functools import partial
from itertools import starmap
from typing import Callable, Optional

import pkg_resources
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

## Check for optional datasets
try:
    from dysts_data.dataloader import get_datapath
except ImportError:
    _has_data = False
else:
    _has_data = True

import numpy as np

from dysts.utils import ddeint, integrate_dyn, standardize_ts

try:
    from numba import njit
except ModuleNotFoundError:
    warnings.warn("Numba not installed. Falling back to no JIT compilation.")
    import numpy as np

    def njit(func):
        return func


data_default = {
    "bifurcation_parameter": None,
    "citation": None,
    "correlation_dimension": None,
    "delay": False,
    "description": None,
    "dt": 0.001,
    "embedding_dimension": 3,
    "hamiltonian": False,
    "initial_conditions": [0.1, 0.1, 0.1],
    "kaplan_yorke_dimension": None,
    "lyapunov_spectrum_estimated": None,
    "maximum_lyapunov_estimated": None,
    "multiscale_entropy": None,
    "nonautonomous": False,
    "parameters": {},
    "period": 10,
    "pesin_entropy": None,
    "unbounded_indices": [],
}

DATAPATH_CONTINUOUS = pkg_resources.resource_filename(
    "dysts", "data/chaotic_attractors.json"
)
DATAPATH_DISCRETE = pkg_resources.resource_filename("dysts", "data/discrete_maps.json")


def staticjit(func: Callable) -> Callable:
    return staticmethod(njit(func))


@dataclass(init=False)
class BaseDyn:
    """A base class for dynamical systems

    Attributes:
        name (str): The name of the system
        params (dict): The parameters of the system.
        random_state (int): The seed for the random number generator. Defaults to None

    Development:
        Add a function to look up additional metadata, if requested
    """

    name: Optional[str] = None
    params: dict = field(default_factory=dict)
    random_state: Optional[int] = None

    def __init__(self, **entries):
        self.name = self.__class__.__name__
        self.data = self._load_data()

        self.params = self.data["parameters"]
        self.params.update(entries)
        self.params = {
            k: v if np.isscalar(v) else np.array(v) for k, v in self.params.items()
        }
        self.__dict__.update(self.params)
        self.param_list = [getattr(self, param) for param in sorted(self.params.keys())]
        ic_val = self.data["initial_conditions"]
        ic_val = np.array(ic_val) if not np.isscalar(ic_val) else np.array([ic_val])
        self.ic = ic_val
        np.random.seed(self.random_state)

        for key in self.data:
            setattr(self, key, self.data[key])

        self.mean = np.asarray(getattr(self, "mean", np.zeros_like(self.ic)))
        self.std = np.asarray(getattr(self, "std", np.ones_like(self.ic)))

    def transform_ic(
        self,
        transform_fn: Callable[[np.ndarray], np.ndarray],
        equation_name: Optional[str] = None,
    ) -> None:
        """Updates the initial condition via a transform function"""
        self.ic = transform_fn(self.ic, equation_name=equation_name)

        warnings.warn(
            """Changing the initial condition makes other estimated parameters
            such as `period`, `mean`, etc invalid!"""
        )

    def transform_params(
        self,
        transform_fn: Callable[[str, np.ndarray], np.ndarray],
        equation_name: Optional[str] = None,
    ) -> None:
        """Updates the current parameter list via a transform function"""
        self.param_list = list(
            starmap(
                partial(transform_fn, equation_name=equation_name),
                zip(sorted(self.params.keys()), self.param_list),
            )
        )

        warnings.warn(
            """Changing the systems parameters makes all other estimated parameters
            such as `period`, `maximum_lyapunov_estimated`, `mean`, etc invalid!"""
        )

    def _load_data(self):
        """Load data from a JSON file"""
        with open(self.data_path, "r") as read_file:
            data = json.load(read_file)
        try:
            return data[self.name]
        except KeyError:
            print(f"No metadata available for {self.name}")
            # return {"parameters": None}
            return data_default

    @staticmethod
    def _rhs(X, t):
        """The right-hand side of the dynamical system. Overwritten by the subclass"""
        raise NotImplementedError

    @staticmethod
    def _jac(X, t, *args):
        """The Jacobian of the dynamical system. Overwritten by the subclass"""
        return None

    @staticmethod
    def bound_trajectory(traj):
        """Bound a trajectory within a periodic domain"""
        return np.mod(traj, 2 * np.pi)

    def load_trajectory(
        self,
        subsets="train",
        granularity="fine",
        return_times=False,
        standardize=False,
        noise=False,
    ):
        """
        Load a precomputed trajectory for the dynamical system

        Args:
            subsets ("train" |  "test"): Which dataset (initial conditions) to load
            granularity ("course" | "fine"): Whether to load fine or coarsely-spaced samples
            noise (bool): Whether to include stochastic forcing
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution
                was computed

        Returns:
            sol (ndarray): A T x D trajectory
            tpts, sol (ndarray): T x 1 timepoint array, and T x D trajectory

        """
        period = 12
        granval = {"coarse": 15, "fine": 100}[granularity]
        dataset_name = subsets.split("_")[0]
        data_path = f"{dataset_name}_multivariate__pts_per_period_{granval}__periods_{period}.json.gz"
        if noise:
            name_parts = list(os.path.splitext(data_path))
            data_path = "".join(name_parts[:-1] + ["_noise"] + [name_parts[-1]])

        if not _has_data:
            warnings.warn(
                "Data module not found. To use precomputed datasets, "
                + "please install the external data repository "
                + "\npip install git+https://github.com/williamgilpin/dysts_data"
            )

        base_path = get_datapath()
        data_path = os.path.join(base_path, data_path)

        with gzip.open(data_path, "rt", encoding="utf-8") as file:
            dataset = json.load(file)

        tpts, sol = (
            np.array(dataset[self.name]["time"]),
            np.array(dataset[self.name]["values"]),
        )

        if standardize:
            print(f"Standardizing {self.name}... ", sol.shape)
            try:
                sol = standardize_ts(sol)
            except Exception as err:
                print(f"Error {err=}, {type(err)=}")
                warnings.warn("Standardization failed")
                sol = None
                raise

        return (tpts, sol) if return_times else sol

    def make_trajectory(self, *args, **kwargs):
        """Make a trajectory for the dynamical system"""
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Sample a trajectory for the dynamical system via numerical integration"""
        return self.make_trajectory(*args, **kwargs)


class DynSys(BaseDyn):
    """
    A continuous dynamical system base class, which loads and assigns parameter
    values from a file

    Attributes:
        kwargs (dict): A dictionary of keyword arguments passed to the base dynamical
            model class
    """

    def __init__(self, **kwargs):
        self.data_path = DATAPATH_CONTINUOUS
        super().__init__(**kwargs)

    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        return self._rhs(*X.T, t, *self.param_list)

    def jac(self, X, t):
        """The Jacobian of the dynamical system"""
        return self._jac(*X.T, t, *self.param_list)

    def __call__(self, X, t):
        """Wrapper around right hand side"""
        return self.rhs(X, t)

    def make_trajectory(
        self,
        n,
        resample=True,
        pts_per_period=100,
        return_times=False,
        standardize=False,
        postprocess=True,
        noise=0.0,
        timescale="Fourier",
        method="Radau",
        rtol=1e-12,
        atol=1e-12,
        **kwargs,
    ):
        """
        Generate a fixed-length trajectory with default timestep, parameters, and initial conditions

        Args:
            n (int): the total number of trajectory points
            resample (bool): whether to resample trajectories to have matching dominant
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution
                was computed
            postprocess (bool): Whether to apply coordinate conversions and other domain-specific
                rescalings to the integration coordinates
            noise (float): The amount of stochasticity in the integrated dynamics. This would correspond
                to Brownian motion in the absence of any forcing.
            timescale (str): The timescale to use for resampling. "Fourier" (default) uses
                the dominant significant Fourier timescale, estimated using the periodogram
                of the system and surrogates. "Lyapunov" uses the Lypunov timescale of
                the system.
            method (str): the integration method
            rtol (float): relative tolerance for the integration routine
            atol (float): absolute tolerance for the integration routine
            **kwargs: Additional keyword arguments passed to the integration routine

        Returns:
            sol (ndarray): A T x D trajectory
            tpts, sol (ndarray): T x 1 timepoint array, and T x D trajectory

        """
        tpts = np.arange(n) * self.dt
        np.random.seed(self.random_state)  # set random seed

        if resample:
            if timescale == "Fourier":
                tlim = (self.period) * (n / pts_per_period)
            elif timescale == "Lyapunov":
                tlim = (1 / self.maximum_lyapunov_estimated) * (n / pts_per_period)
            else:
                tlim = (self.period) * (n / pts_per_period)

            upscale_factor = (tlim / self.dt) / n
            if upscale_factor > 1e3:
                warnings.warn(
                    f"Expect slowdown due to excessive integration required; scale factor {upscale_factor}"
                )
            tpts = np.linspace(0, tlim, n)

        mu = self.mean if standardize else np.zeros_like(self.ic)
        std = self.std if standardize else np.ones_like(self.ic)

        # check for analytical Jacobian, with condition of ic being a ndim array
        if (self.ic.ndim > 1 and self.jac(self.ic[0], 0)) or self.jac(
            self.ic, 0
        ) is not None:
            jac = lambda t, x: self.jac(std * x + mu, t) / std
        else:
            jac = None

        m = self.ic.ndim
        ics = np.expand_dims(self.ic, axis=0) if m < 2 else self.ic

        def standard_rhs(X, t):
            return self(X * std + mu, t) / std

        # compute trajectories
        sol = list()
        for ic in ics:
            traj = integrate_dyn(
                standard_rhs,
                (ic - mu) / std,
                tpts,
                dtval=self.dt,
                method=method,
                noise=noise,
                jac=jac,
                rtol=rtol,
                atol=atol,
                **kwargs,
            )
            # check completeness of trajectory to kill off incomplete trajectories
            if traj.shape[-1] == len(tpts):  # full trajectory should have n points
                sol.append(traj)
            else:
                warnings.warn(
                    f"{self.name}: Integration did not complete for initial condition {ic}, only got {traj.shape[-1]} points. Skipping this point"
                )

        if len(sol) == 0:  # if no complete trajectories, return None
            return (tpts, None) if return_times else None

        sol = np.transpose(np.array(sol), (0, 2, 1))

        # postprocess the trajectory, if necessary
        if hasattr(self, "_postprocessing") and postprocess:
            warnings.warn(
                "This system has at least one unbounded variable, which has been mapped to a bounded domain. Pass argument postprocess=False in order to generate trajectories from the raw system."
            )
            sol2 = np.moveaxis(sol, (-1, 0), (0, -1))
            sol = np.moveaxis(np.dstack(self._postprocessing(*sol2)), (0, 1), (1, 0))
        sol = np.squeeze(sol)

        return (tpts, sol) if return_times else sol


class DynMap(BaseDyn):
    """
    A dynamical system base class, which loads and assigns parameter
    values from a file

    Args:
        params (list): parameter values for the differential equations
        kwargs (dict): A dictionary of keyword arguments passed to the base dynamical
            model class

    Todo:
        A function to look up additional metadata, if requested
    """

    def __init__(self, **kwargs):
        self.data_path = DATAPATH_DISCRETE
        super().__init__(**kwargs)

    def rhs(self, X):
        """The right hand side of a dynamical map"""
        out = self._rhs(*X.T, *self.param_list)
        return np.vstack(out).T

    def rhs_inv(self, Xp):
        """The inverse of the right hand side of a dynamical map"""
        out = self._rhs_inv(*Xp.T, *self.param_list)
        return np.vstack(out).T

    def __call__(self, X):
        """Wrapper around right hand side"""
        return self.rhs(X)

    def make_trajectory(
        self, n, inverse=False, return_times=False, standardize=False, **kwargs
    ):
        """
        Generate a fixed-length trajectory with default timestep,
        parameters, and initial condition(s)

        Args:
            n (int): the length of each trajectory
            inverse (bool): whether to reverse a trajectory
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution
                was computed
        """

        m = self.ic.ndim
        # shape (M, D)
        curr = np.expand_dims(self.ic, axis=0) if m < 2 else self.ic

        if inverse:
            propagator = self.rhs_inv
        else:
            propagator = self.rhs

        traj = np.zeros((curr.shape[0], n, curr.shape[-1]))

        for i in range(n):
            curr = propagator(curr)
            traj[:, i, :] = curr

        sol = np.squeeze(traj)

        if standardize:
            sol = standardize_ts(sol)

        if return_times:
            return np.arange(len(sol)), sol
        else:
            return sol


class DynSysDelay(BaseDyn):
    """
    A delayed differential equation object. Uses a exposed fork of ddeint
    The delay timescale is assumed to be the "tau" field. The embedding dimensions are set to a
    default value, but delay equations are infinite dimensional.

    Attributes:
        kwargs (dict): A dictionary of keyword arguments passed to the dynamical
            system parent class

    Todo:
        Currently, only univariate delay equations and single initial conditons
        are supported
    """

    def __init__(self, **kwargs):
        self.data_path = DATAPATH_CONTINUOUS
        super().__init__(**kwargs)

    def rhs(self, X: Callable[[float], ArrayLike], t: float) -> ArrayLike:
        """The right hand side of a dynamical equation"""
        xt, xt_delayed = X(t), X(t - self.tau)
        return self._rhs(xt, xt_delayed, t, *self.param_list)

    def make_trajectory(
        self,
        n: int,
        method: str = "Euler",
        noise: float = 0.0,
        resample: bool = False,
        pts_per_period: int = 100,
        standardize: bool = False,
        timescale: str = "Fourier",
        return_times: bool = False,
        postprocess: bool = True,
        embedding_dim: Optional[int] = None,
        history_function: Optional[Callable[[float], ArrayLike]] = None,
        **kwargs,
    ):
        """
        Generate a fixed-length trajectory with default timestep, parameters, and
        initial conditions.

        Args:
            n (int): the total number of trajectory points
            method (str): Not used. Currently Euler is the only option here
            noise (float): The amplitude of brownian forcing
            resample (bool): whether to resample trajectories to have matching dominant
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            standardize (bool): Standardize the output time series.
            timescale (str): The timescale to use for resampling. "Fourier" (default) uses
                the dominant significant Fourier timescale, estimated using the periodogram
                of the system and surrogates. "Lyapunov" uses the Lypunov timescale of
                the system.
            return_times (bool): Whether to return the timepoints at which the solution
                was computed
            postprocess (bool): Whether to apply coordinate conversions and other domain-specific
                rescalings to the integration coordinates
            embedding_dim (int): Optionally augment solution with delay embedded trajectories.
                If equation is multi-dimensional with dimension d, then this will return a flattened
                delay embedding of shape (n, d*embedding_dim) where each consecutive (n, d) block
                will be a trajectory for a different delay parameter.
            history_function (callable): Function for specifying past conditions i.e.
                points for t < 0

        Todo:
            Support for multiple initial conditions
        """
        # add delay time increment for potential interpolation later on
        interp_pad = self.tau

        # if not provided, fallback to default or 1 if a default doesnt exist
        emb_dim = (
            getattr(self, "embedding_dimension", 1)
            if embedding_dim is None
            else embedding_dim
        )

        tpts = np.linspace(0, n * self.dt + interp_pad, n)
        np.random.seed(self.random_state)  # set random seed

        if resample:
            if timescale == "Fourier":
                tlim = (self.period) * (n / pts_per_period)
            elif timescale == "Lyapunov":
                tlim = (1 / self.maximum_lyapunov_estimated) * (n / pts_per_period)
            else:
                tlim = (self.period) * (n / pts_per_period)

            upscale_factor = (tlim / self.dt) / n
            if upscale_factor > 1e3:
                warnings.warn(
                    f"Expect slowdown due to excessive integration required; scale factor {upscale_factor}"
                )

            tpts = np.linspace(0, tlim + interp_pad, n)

        # assume constant past points, overridable behavior
        # need to change initial conditions to NOT be the same size as the default embedding
        # dimensions, since the delay embedding happens as a postprocessing step
        history_fn = history_function or (lambda t: self.ic[0])
        sol = ddeint(self.rhs, history_fn, tpts)

        # augment the trajectory with per-dimension delay embeddings
        interp_fns = [
            interp1d(
                tpts,
                sol[:, dim],
                axis=0,
                kind=kwargs.pop("kind", "linear"),
            )
            for dim in range(sol.shape[-1])
        ]

        sample_ts = np.linspace(self.tau, tpts[-1], n)
        sol = (
            np.stack(
                [
                    [fn(sample_ts - tau) for fn in interp_fns]
                    for tau in np.linspace(0, self.tau, emb_dim)
                ],
                axis=1,
            )
            .reshape(-1, len(tpts))
            .T
        )

        if standardize:
            sol = standardize_ts(sol)

        if return_times:
            return tpts, sol
        else:
            return sol
