"""Utilities for the implemented systems"""

import inspect
import json
from multiprocessing import Pool
from os import PathLike
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from . import flows as dfl
from . import maps as dmp
from .base import (
    DATAPATH_CONTINUOUS,
    DATAPATH_DISCRETE,
    BaseDyn,
    DynMap,
    DynSys,
    DynSysDelay,
)
from .sampling import BaseSampler

Array = npt.NDArray[np.float64]

DEFAULT_RNG = np.random.default_rng()


def get_attractor_list(
    sys_class: str = "continuous", exclude: list[str] = []
) -> list[str]:
    """Get names of implemented dynamical systems

    Args:
        sys_class: class of systems to get the name of - must
            be one of ['continuous', 'continuous_no_delay', 'delay', 'discrete']
        exclude: list of systems to exclude, optional

    Returns:
        Sorted list of systems belonging to sys_class
    """
    module: ModuleType
    if sys_class in ["continuous", "continuous_no_delay", "delay"]:
        module = dfl
        parent_class = {
            "continuous": (DynSys, DynSysDelay),
            "continuous_no_delay": DynSys,
            "delay": DynSysDelay,
        }[sys_class]
    elif sys_class == "discrete":
        module = dmp
        parent_class = DynMap
    else:
        raise Exception(
            "sys_class must be in ['continuous', 'continuous_no_delay', 'delay', 'discrete']"
        )

    systems: list[tuple[str, BaseDyn]] = inspect.getmembers(
        module,
        lambda obj: inspect.isclass(obj)  # type: ignore
        and issubclass(obj, parent_class)  # type: ignore
        and obj.__module__ == module.__name__,
    )

    systems = list(filter(lambda x: x[0] not in exclude, systems))

    return sorted([name for name, _ in systems])


def get_system_data(
    sys_class: str = "continuous", exclude: list[str] = []
) -> dict[str, Any]:
    """Get system data from the dedicated system class json files

    Arguments:
        sys_class: class of systems to get the name of - must
            be one of ['continuous', 'continuous_no_delay', 'delay', 'discrete']
        exclude: list of systems to exclude, optional

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

    systems = get_attractor_list(sys_class, exclude)
    with open(datapath, "r") as file:
        data = json.load(file)

    # filter out systems from the data
    return {k: v for k, v in data.items() if k in systems}


def _compute_trajectory(
    system: str | BaseDyn,
    n: int,
    kwargs: dict[str, Any],
    ic_transform: BaseSampler | None = None,
    param_transform: BaseSampler | None = None,
    ic_rng: np.random.Generator | None = None,
    param_rng: np.random.Generator | None = None,
    _silent_errors: bool = False,
) -> Array | None:
    """Helper function to compute a single trajectory for a dynamical system.

    Args:
        system (Union[str, BaseDyn]): Either a string name of a system or a system instance
        n (int): Number of timepoints to integrate
        kwargs (Dict[str, Any]): Additional arguments passed to make_trajectory
        ic_transform (Optional[BaseSampler]): Transform to apply to initial conditions
        param_transform (Optional[BaseSampler]): Transform to apply to parameters
        ic_rng (Optional[np.random.Generator]): Random number generator for IC sampling
        param_rng (Optional[np.random.Generator]): Random number generator for param sampling
        _silent_errors (bool): Whether to silence errors and return None instead

    Returns:
        Optional[Array]: The computed trajectory, or None if error occurs and _silent_errors=True
    """
    if isinstance(system, str):
        sys = getattr(dfl, system)()
    else:
        sys = system

    if param_transform is not None:
        if param_rng is not None and hasattr(param_transform, "set_rng"):
            param_transform.set_rng(param_rng)
        sys.transform_params(param_transform)  # type: ignore

    # the initial condition transform must come after the parameter transform
    # because suitable initial conditions may depend on the parameters
    if ic_transform is not None:
        if ic_rng is not None and hasattr(ic_transform, "set_rng"):
            ic_transform.set_rng(ic_rng)
        sys.transform_ic(ic_transform)  # type: ignore

    try:
        traj = sys.make_trajectory(n, **kwargs)
    except Exception as e:
        print(f"Error in {sys.name}: {e}")
        if _silent_errors:
            return None
        raise e

    return traj


def make_trajectory_ensemble(
    n: int,
    use_tqdm: bool = True,
    use_multiprocessing: bool = False,
    ic_transform: BaseSampler | None = None,
    param_transform: BaseSampler | None = None,
    subset: Sequence[str] | Sequence[BaseDyn] | None = None,
    ic_rng: np.random.Generator | None = None,
    param_rng: np.random.Generator | None = None,
    **kwargs,
) -> dict[str, Array | None]:
    """
    Integrate multiple dynamical systems with identical settings

    Args:
        n (int): The number of timepoints to integrate
        use_tqdm (bool): Whether to use a progress bar
        use_multiprocessing (bool): Not yet implemented.
        ic_transform (callable): function that transforms individual system initial conditions
        param_transform (callable): function that transforms individual system parameters
        subset (list): A list of system names or BaseDyn (e.g. custom dynamical systems). Defaults to all continuous systems.
            Can also pass in `sys_class` as a kwarg to specify other system classes.
        kwargs (dict): Integration options passed to each system's make_trajectory() method

    Returns:
        all_sols (dict): A dictionary containing trajectories for each system

    """
    if subset is None:
        sys_class = kwargs.pop("sys_class", "continuous")
        exclude = kwargs.pop("exclude", [])
        subset = get_attractor_list(sys_class, exclude)

    if use_tqdm and not use_multiprocessing:
        subset = tqdm(subset, desc="Integrating systems")  # type: ignore

    all_sols = dict()
    if use_multiprocessing:
        all_sols = _multiprocessed_compute_trajectory(
            n, subset or [], ic_transform, param_transform, ic_rng, param_rng, **kwargs
        )
    else:
        # stupid lint error fix for subset being possibly None
        for system in subset or []:
            sol = _compute_trajectory(system, n, kwargs, ic_transform, param_transform)
            equation_name = system if isinstance(system, str) else type(system).__name__
            all_sols[equation_name] = sol

    return all_sols


def _multiprocessed_compute_trajectory(
    n: int,
    subset: Sequence[str] | Sequence[BaseDyn],
    ic_transform: BaseSampler | None = None,
    param_transform: BaseSampler | None = None,
    ic_rng: np.random.Generator | None = None,
    param_rng: np.random.Generator | None = None,
    **kwargs,
) -> dict[str, Array | None]:
    """
    Helper for handling multiprocessed integration
    with _compute_trajectory with proper RNG seeding

    NOTE: If rngs are provided, every child process will receive a new rng, this is
    necessary for proper sampling as per: https://numpy.org/doc/stable/reference/random/parallel.html

    otherwise, the default rng will be used for each process and the results will be deterministic
    """
    if ic_rng is not None:
        ic_rng_stream = ic_rng.spawn(len(subset))
    else:
        ic_rng_stream = [None for _ in subset]

    if param_rng is not None:
        param_rng_stream = param_rng.spawn(len(subset))
    else:
        param_rng_stream = [None for _ in subset]

    with Pool() as pool:
        results = pool.starmap(
            _compute_trajectory,
            [
                (
                    equation_name,
                    n,
                    kwargs,
                    ic_transform,
                    param_transform,
                    ic_rng,
                    param_rng,
                )
                for equation_name, param_rng, ic_rng in zip(
                    subset, param_rng_stream, ic_rng_stream
                )
            ],
        )

    names = [name if isinstance(name, str) else type(name).__name__ for name in subset]
    return dict(zip(names, results))


def compute_trajectory_statistics(
    n: int,
    subset: Sequence[str],
    datapath: PathLike | None = None,
    **kwargs,
) -> dict[str, dict[str, Array]]:
    """
    Compute mean and std for given trajectory list
    Args:
        n (int): The number of timepoints to integrate
        subset (list): A list of system names. Defaults to all continuous systems.
            Can also pass in `sys_class` as a kwarg to specify other system classes.
        datapath (str): Path to save the computed statistics to a JSON file
        kwargs: Additional keyword arguments passed to the integration routine
    """
    sols = make_trajectory_ensemble(n, subset=subset, **kwargs)
    stats = {
        name: {"mean": sol.mean(axis=0), "std": sol.std(axis=0)}
        for name, sol in sols.items()
        if sol is not None
    }

    # Save the computed statistics to a JSON file
    if datapath is not None:
        with open(datapath) as f:
            data = json.load(f)

        for system in subset:
            data[system].update({k: v.tolist() for k, v in stats[system].items()})

        with open(datapath, "w") as f:
            json.dump(data, f, indent=2)

    return stats
