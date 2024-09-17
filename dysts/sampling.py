"""Sampling functions for dysts"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import dysts.flows as flows

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

    def __call__(self, ic: Array, equation_name: Optional[str] = None) -> Array:
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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.trajectory_cache = {}

    def __call__(self, ic: Array, equation_name: Optional[str] = None) -> Array:
        if equation_name is not None and equation_name not in self.trajectory_cache:
            # Integrate the system with default parameters
            eq = getattr(flows, equation_name)()
            self.trajectory_cache[equation_name] = eq.make_trajectory(
                self.reference_traj_length
            )[self.reference_traj_transient :]

        trajectory = self.trajectory_cache[equation_name]

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
        self, name: str, param: Array, equation_name: Optional[str] = None
    ) -> Array:
        size = None if np.isscalar(param) else param.shape

        # scale each parameter relatively
        scale = np.linalg.norm(param) * self.scale
        return self.rng.normal(loc=param, scale=scale, size=size)
