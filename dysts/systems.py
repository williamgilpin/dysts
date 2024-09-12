"""Utilities for the implemented systems"""

import inspect
import json
from functools import partial
from multiprocessing import Pool
from os import PathLike
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import numpy.typing as npt

import dysts.flows as dfl
import dysts.maps as dmp
from dysts.base import DATAPATH_CONTINUOUS, DATAPATH_DISCRETE

Array = npt.NDArray[np.float64]


def get_attractor_list(sys_class: str = "continuous") -> List[str]:
    """Get names of implemented dynamical systems

    Args:
        sys_class: class of systems to get the name of - must
            be one of ['continuous', 'continuous_no_delay', 'delay', 'discrete']

    Returns:
        Sorted list of systems belonging to sys_class
    """
    if sys_class in ["continuous", "continuous_no_delay", "delay"]:
        module = dfl
    elif sys_class == "discrete":
        module = dmp
    else:
        raise Exception(
            "sys_class must be in ['continuous', 'continuous_no_delay', 'delay', 'discrete']"
        )

    systems = (
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    )

    if sys_class == "continuous_no_delay":
        systems = filter(lambda name: "delay" not in name.lower(), systems)
    elif sys_class == "delay":
        systems = filter(lambda name: "delay" in name.lower(), systems)

    return sorted(systems)


def get_system_data(sys_class: str = "continuous") -> Dict[str, Any]:
    """Get system data from the dedicated system class json files

    Arguments:
        sys_class: class of systems to get the name of - must
            be one of ['continuous', 'continuous_no_delay', 'delay', 'discrete']

    Returns:
        Data from json file filtered by sys_class as a dict
    """
    if sys_class in ["continuous", "continuous_no_delay", "delay"]:
        datapath = DATAPATH_CONTINUOUS
    elif sys_class == "discrete":
        datapath = DATAPATH_DISCRETE
    else:
        raise Exception(
            "sys_class must be in ['continuous', 'continuous_no_delay', 'delay', 'discrete']"
        )

    systems = get_attractor_list(sys_class)
    with open(datapath, "r") as file:
        data = json.load(file)

    # filter out systems from the data
    return {k: v for k, v in data.items() if k in systems}


def _compute_trajectory(
    equation_name, n, kwargs, init_cond=None, param_transform_fn=None
):
    """A helper function for multiprocessing"""
    eq = getattr(dfl, equation_name)()

    if init_cond is not None:
        eq.ic = init_cond

    if param_transform_fn is not None:
        eq.transform_params(param_transform_fn)

    traj = eq.make_trajectory(n, **kwargs)
    return traj


def make_trajectory_ensemble(
    n: int,
    init_conds: Dict[str, Array] = {},
    use_tqdm: bool = False,
    use_multiprocessing: bool = False,
    param_transform: Optional[Callable] = None,
    subset: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict[str, Array]:
    """
    Integrate multiple dynamical systems with identical settings

    Args:
        n (int): The number of timepoints to integrate
        subset (list): A list of system names. Defaults to all systems
        use_multiprocessing (bool): Not yet implemented.
        init_cond (dict): Optional user input initial conditions mapping string system name to array
        param_transform (callable): function that transforms individual system parameters
        use_tqdm (bool): Whether to use a progress bar
        kwargs (dict): Integration options passed to each system's make_trajectory() method

    Returns:
        all_sols (dict): A dictionary containing trajectories for each system

    """
    if subset is None:
        # subset = get_attractor_list()
        subset = []

    if len(init_conds) > 0:
        assert all(
            sys in init_conds.keys() for sys in subset
        ), "given initial conditions must at least contain the subset"

    if use_tqdm and not use_multiprocessing:
        from tqdm import tqdm

        subset = tqdm(subset)

    all_sols = dict()
    if use_multiprocessing:
        with Pool() as pool:
            results = pool.starmap(
                partial(_compute_trajectory, param_transform_fn=param_transform),
                [
                    (equation_name, n, kwargs, init_conds.get(equation_name))
                    for equation_name in subset
                ],
            )
        all_sols = dict(zip(subset, results))

    else:
        for equation_name in subset:
            sol = _compute_trajectory(
                equation_name, n, kwargs, init_conds.get(equation_name), param_transform
            )
            all_sols[equation_name] = sol

    return all_sols


def gaussian_init_cond_sampler(
    random_seed: Optional[int] = 0,
    subset: Optional[Iterable] = None,
    dynsys_class: str = "continuous",
) -> Callable:
    """Sample gaussian perturbations for each initial condition in a given system list

    Args:
        random_seed: for random sampling
        subset: A list of system names. Defaults to all systems

    Returns:
        a function which samples a random perturbation of the init conditions
    """
    if subset is None:
        subset = get_attractor_list()

    rng = np.random.default_rng(random_seed)
    ic_dict = {sys: np.array(getattr(dfl, sys)().ic) for sys in subset}

    def _sampler(scale: float = 1e-4) -> Dict[str, Array]:
        return {
            sys: rng.normal(loc=ic, scale=scale, size=ic.shape)
            for sys, ic in ic_dict.items()
        }

    return _sampler


def gaussian_parameter_sampler(random_seed: int = 0, scale: float = 1e-3) -> Callable:
    """Sample gaussian perturbations for system parameters

    Args:
        random_seed: for random sampling
        scale: std (isotropic) of gaussian used for sampling

    Returns:
        a function which samples a random perturbation of given parameters
    """
    rng = np.random.default_rng(random_seed)

    def _sampler(name: str, param: Array) -> Array:
        size = None if np.isscalar(param) else param.shape
        return rng.normal(loc=param, scale=scale, size=size)

    return _sampler


def compute_trajectory_statistics(
    n: int,
    subset: Optional[Iterable[str]] = None,
    datapath: Optional[PathLike] = None,
    **kwargs,
) -> Dict[str, Dict[str, Array]]:
    """Compute mean and std for given trajectory list"""
    sols = make_trajectory_ensemble(n, subset=subset, **kwargs)
    return {
        name: {"mean": sol.mean(axis=0), "std": sol.std(axis=0)}
        for name, sol in sols.items()
    }
