"""Utilities for the implemented systems"""

import inspect
import json
from multiprocessing import Pool
from os import PathLike
from types import ModuleType
from typing import Any, Callable, Sequence

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
from .utils import num_unspecified_params

Array = npt.NDArray[np.float64]


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


def _resolve_event_signature(
    system: BaseDyn,
    event: Callable[[BaseDyn], Callable[[float, Array], float]]
    | Callable[[float, Array], float],
) -> Callable[[float, Array], float]:
    """
    Hacky check if a given event function is system dependent or not

    If the event callable has a single argument, it is assumed to be system dependent
        and the event returns a solve_ivp compatible event function.
    If the event callable has more than one argument, it is assumed to be a solve_ivp
        compatible event function.

    Args:
        system: The system to pass to the possibly system dependent event function
        event: The event function to resolve

    Returns:
        The resolved event function (solve_ivp compatible)
    """
    if num_unspecified_params(event) == 1:
        return event(system)  # type: ignore
    return event  # type: ignore


def _compute_trajectory(
    n: int,
    system: str | BaseDyn,
    event_fns: Sequence[Callable] | None,
    kwargs: dict[str, Any],
) -> Array | None:
    """Helper function to compute a single trajectory for a dynamical system.

    Args:
        system (Union[str, BaseDyn]): Either a string name of a system or a system instance
        n (int): Number of timepoints to integrate
        event_fns (Sequence[Callable]): A list of functions that take a dynamical system and returns a
            solve_ivp compatible event function.
        kwargs (Dict[str, Any]): Additional arguments passed to make_trajectory

    Returns:
        Optional[Array]: The computed trajectory, or None if error occurs and _silent_errors=True
    """
    if isinstance(system, str):
        sys = getattr(dfl, system)()
    else:
        sys = system

    # if event functions are provided, resolve them and pack them into the kwargs
    if event_fns is not None:
        kwargs["events"] = [
            _resolve_event_signature(sys, event_fn) for event_fn in event_fns
        ]

    silent_errors = kwargs.pop("_silent_errors", False)
    logger = kwargs.pop("logger", None)

    try:
        traj = sys.make_trajectory(n, **kwargs)
    except Exception as exception:
        if logger is not None:
            logger.error(f"Error in {sys.name}: {exception}")

        # fail silently by returning None if silent_errors is True,
        # useful for large ensemble runs
        if silent_errors:
            return None
        raise exception

    return traj


def make_trajectory_ensemble(
    n: int,
    use_tqdm: bool = True,
    use_multiprocessing: bool = False,
    subset: Sequence[str] | Sequence[BaseDyn] | None = None,
    event_fns: Sequence[Callable] | None = None,
    **kwargs,
) -> dict[str, Array | None]:
    """
    Integrate multiple dynamical systems with identical settings

    Args:
        n (int): The number of timepoints to integrate
        use_tqdm (bool): Whether to use a progress bar
        use_multiprocessing (bool): Not yet implemented.
        subset (list): A list of system names or BaseDyn (e.g. custom dynamical systems). Defaults to all continuous systems.
            Can also pass in `sys_class` as a kwarg to specify other system classes.
        event_fns (list): A list of functions that either take the signature:
            (system: BaseDyn) -> (event_fn: Callable[[float, Array], float])
            or the signature:
            (event_fn: Callable[[float, Array], float])
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
            n, subset or [], event_fns, **kwargs
        )
    else:
        # stupid lint error fix for subset being possibly None
        for system in subset or []:
            sol = _compute_trajectory(n, system, event_fns, kwargs)
            equation_name = system if isinstance(system, str) else system.name
            all_sols[equation_name] = sol

    return all_sols


def _multiprocessed_compute_trajectory(
    n: int,
    subset: Sequence[str] | Sequence[BaseDyn],
    event_fns: Sequence[Callable] | None,
    **kwargs,
) -> dict[str, Array | None]:
    """Helper for handling multiprocessed integration

    Args:
        n: Number of timepoints to integrate
        subset: Systems to compute trajectories for
        event_fns: Event functions to pass to the systems
        **kwargs: Additional arguments passed to _compute_trajectory
    """
    with Pool() as pool:
        solutions = pool.starmap(
            _compute_trajectory,
            [(n, system, event_fns, kwargs) for system in subset],
        )

    names = [sys if isinstance(sys, str) else sys.name for sys in subset]
    return dict(zip(names, solutions))


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
