"""Sampling functions for dysts"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from dysts.base import BaseDyn

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
    """Sample gaussian perturbations for each initial condition in a given system list"""

    scale: float = 1e-4

    def __call__(self, ic: Array, system: Optional[BaseDyn] = None) -> Array:
        scaled_cov = np.diag(np.square(ic * self.scale))
        return self.rng.multivariate_normal(mean=ic, cov=scaled_cov)


@dataclass
class OnAttractorInitCondSampler(BaseSampler):
    """
    Sample points from the attractor of a system

    Subtleties:
        - This is slow, it requires integrating each system with its default
          parameters before sampling from the attractor.
        - The sampled initial conditions from this sampler are necessarily
          tied to the attractor defined by the default parameters.
    """

    reference_traj_length: int = 4096
    reference_traj_transient: int = 500
    trajectory_cache: Dict[str, Array] = field(default_factory=dict)

    def __call__(self, ic: Array, system: BaseDyn) -> Array:
        if system.name is None:
            raise ValueError("System must have a name")
        if system.name not in self.trajectory_cache:
            # Integrate the system with default parameters
            self.trajectory_cache[system.name] = system.make_trajectory(
                self.reference_traj_length
            )[self.reference_traj_transient :]

        trajectory = self.trajectory_cache[system.name]

        # Sample a new initial condition from the attractor
        return self.rng.choice(trajectory)


@dataclass
class GaussianParamSampler(BaseSampler):
    """Sample gaussian perturbations for system parameters

    NOTE:
        - This is a MWE of a parameter transform
        - Other parameter transforms should follow this dataclass template

    Args:
        random_seed: for random sampling
        scale: std (isotropic) of gaussian used for sampling
    """

    scale: float = 1e-2

    def __call__(
        self, name: str, param: Array, system: Optional[BaseDyn] = None
    ) -> Array:
        # scale each parameter relatively
        shape = 1 if np.isscalar(param) else param.shape

        # avoid shape errors
        param = np.array(param).flatten()
        scale = np.abs(param) * self.scale
        cov = np.diag(np.square(scale))
        return (
            self.rng.multivariate_normal(mean=param, cov=cov).reshape(shape).squeeze()
        )
