"""Utilities for simulating trajectories"""

from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import numpy.typing as npt

import dysts.flows as dfl
from dysts.base import get_attractor_list


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
    n,
    subset=None,
    use_multiprocessing=False,
    init_conds={},
    param_transform=None,
    use_tqdm=False,
    **kwargs,
):
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
    if not subset:
        subset = get_attractor_list()

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


def init_cond_sampler(
    random_seed: Optional[int] = 0, subset: Optional[Iterable] = None
) -> Callable:
    """Sample zero mean guassian perturbations for each initial condition in a given system list

    Args:
        random_seed: for random sampling
        subset: A list of system names. Defaults to all systems

    Returns:
        a function which samples a random perturbation of the init conditions
    """
    if not subset:
        subset = get_attractor_list()

    rng = np.random.default_rng(random_seed)
    ic_dict = {sys: np.array(getattr(dfl, sys)().ic) for sys in subset}

    def _sampler(scale: float = 1e-4) -> Dict[str, npt.NDArray[np.float64]]:
        return {
            sys: ic + rng.normal(scale=scale * np.linalg.norm(ic), size=ic.shape)
            for sys, ic in ic_dict.items()
        }

    return _sampler
