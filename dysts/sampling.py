"""Sampling functions for dysts"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class BaseSampler:
    """Base class for any sampling-based transformations"""

    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def set_rng(self, rng: np.random.Generator) -> None:
        """Required for multiprocessing"""
        self.rng = rng


def gaussian_init_cond_sampler(
    random_seed: Optional[int] = 0,
    subset: Optional[Iterable] = None,
    covariances: Optional[Dict[str, Array]] = None,
    sys_class: str = "continuous",
) -> Callable:
    """Sample gaussian perturbations for each initial condition in a given system list

    Args:
        random_seed: for random sampling
        subset: A list of system names. Defaults to all systems
        covariances: A dict of covariance matrices for each system for sampling
        sys_class: only used when subset is None

    Returns:
        a function which samples a random perturbation of the init conditions
    """
    if subset is None:
        subset = get_attractor_list(sys_class)

    rng = np.random.default_rng(random_seed)
    ic_dict = {sys: np.array(getattr(dfl, sys)().ic) for sys in subset}

    if covariances is not None:
        assert all(
            cm.ndim == 2 and cm.shape[0] == cm.shape[1] for cm in covariances.values()
        )
        assert all(k in subset for k in covariances.keys())

    def _sampler(scale: Union[float, Array] = 1e-4) -> Dict[str, Array]:
        if covariances is not None:
            return {
                sys: rng.multivariate_normal(ic, covariances[sys])
                for sys, ic in ic_dict.items()
            }
        else:
            return {
                sys: rng.normal(ic, scale=scale, size=ic.shape)
                for sys, ic in ic_dict.items()
            }

    return _sampler


@dataclass
class GaussianInitialConditionSampler(BaseSampler):
    """Sample gaussian perturbations for each initial condition

    Args:
        random_seed: for random sampling
        scale: std (isotropic) of gaussian used for sampling
    """

    scale: float = 1e-4

    def __call__(self, init_cond: Array) -> Array:
        return self.rng.normal(loc=init_cond, scale=self.scale, size=init_cond.shape)


def attractor_init_cond_sampler(
    random_seed: Optional[int] = 0,
    subset: Optional[Iterable] = None,
    sys_class: str = "continuous",
) -> Callable:
    """Sample points from the attractor of a system"""
    pass


@dataclass
class GaussianParamSampler(BaseSampler):
    """Sample gaussian perturbations for system parameters

    NOTE: This is a MWE of a parameter transform. Other examples should follow
    this dataclass template in order to be pickable e.g. for multiprocessing

    Args:
        random_seed: for random sampling
        scale: std (isotropic) of gaussian used for sampling
    """

    scale: float = 1e-2

    def __call__(self, name: str, param: Array) -> Array:
        size = None if np.isscalar(param) else param.shape

        # scale each parameter relatively
        scale = np.linalg.norm(param) * self.scale
        return self.rng.normal(loc=param, scale=scale, size=size)
