"""Sampling functions for dysts"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .base import BaseDyn

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
    """
    Sample gaussian perturbations for each initial condition in a given system list
    """

    scale: float = 1e-4
    verbose: bool = False  # for testing purposes

    def __call__(self, ic: Array, system: Optional[BaseDyn] = None) -> Array:
        # Scale the covariance relative to each dimension
        scaled_cov = np.diag(np.square(ic * self.scale))
        perturbed_ic = self.rng.multivariate_normal(mean=ic, cov=scaled_cov)

        if self.verbose:
            if system is not None:
                print(f"System: {system.name}")
            print(f"--> Original initial condition: {ic}")
            print(f"--> Perturbed initial condition: {perturbed_ic}")

        return perturbed_ic


@dataclass
class OnAttractorInitCondSampler(BaseSampler):
    """
    Sample points from the attractor of a system

    Subtleties:
        - This is slow, it requires integrating each system with its default
          parameters before sampling from the attractor.
        - The sampled initial conditions from this sampler are necessarily
          tied to the attractor defined by the default parameters.

    Args:
        reference_traj_length: Length of the reference trajectory to use for sampling ic on attractor.
        reference_traj_transient: Transient length to ignore for the reference trajectory
        trajectory_cache: Cache of reference trajectories for each system.
        events: events to pass to solve_ivp
    """

    reference_traj_length: int = 4096
    reference_traj_transient: int = 500
    trajectory_cache: Dict[str, Array] = field(default_factory=dict)
    verbose: bool = False  # for testing purposes
    events: Optional[List[Callable]] = None  # solve_ivp events

    def __call__(self, ic: Array, system: BaseDyn) -> Array:
        if system.name is None:
            raise ValueError("System must have a name")

        # make reference trajectory if not already cached
        if system.name not in self.trajectory_cache:
            if self.verbose:
                print(f"Adding {system.name} to trajectory cache...")
            # Integrate the system with default parameters
            reference_traj = system.make_trajectory(
                self.reference_traj_length,
                events=self.events,
            )[self.reference_traj_transient :]

            if (
                reference_traj is None
            ):  # if integrate fails, resulting in an incomplete trajectory
                raise ValueError(
                    f"Failed to integrate the system {system.name} with ic {system.ic} and params {system.params}"
                )

            self.trajectory_cache[system.name] = reference_traj

        trajectory = self.trajectory_cache[system.name]

        # Sample a new initial condition from the attractor
        new_ic = self.rng.choice(trajectory)

        if self.verbose:
            print(f"System: {system.name}")
            print(f"--> Original initial condition: {ic}")
            print(f"--> New initial condition: {new_ic}")

        return new_ic


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
    verbose: bool = False  # for testing purposes

    def __call__(
        self, name: str, param: Array, system: Optional[BaseDyn] = None
    ) -> Array | float:
        # scale each parameter relatively
        shape = (1,) if np.isscalar(param) else param.shape

        # avoid shape errors
        flat_param = np.array(param).flatten()
        scale = np.abs(flat_param) * self.scale
        cov = np.diag(np.square(scale))
        perturbed_param = (
            self.rng.multivariate_normal(mean=flat_param, cov=cov)
            .reshape(shape)
            .squeeze()
        )
        if isinstance(param, (float, int)):
            perturbed_param = float(perturbed_param)

        if self.verbose:
            if system is not None:
                print(f"System: {system.name}")
            print(f"Parameter name: {name}")
            print(f"--> Original parameter: {param}")
            print(f"--> Perturbed parameter: {perturbed_param}")

        return perturbed_param
