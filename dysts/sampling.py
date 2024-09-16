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


@dataclass
class GaussianInitialConditionSampler(BaseSampler):
    """Sample gaussian perturbations for each initial condition in a given system list
    """
    scale: float = 1e-4

    def __call__(self, ic: Array) -> Array:
        """
        Sample a new initial condition from a multivariate isotropic Gaussian.

        Args:
            ic (Array): The current initial condition.

        Returns:
            Array: A resampled version of the initial condition.
        """
        cov = np.eye(ic.shape[0]) * self.scale**2
        return self.rng.multivariate_normal(mean=ic, cov=cov)



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
