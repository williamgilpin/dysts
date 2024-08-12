import numpy as np

from dataclasses import dataclass
from dysts.base import init_cond_sampler, make_trajectory_ensemble


@dataclass
class ParamPerturb:

    scale: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, name: str, param: np.ndarray) -> np.ndarray:
        size = None if np.isscalar(param) else param.shape
        return param + self.rng.normal(scale=self.scale*np.linalg.norm(param), size=size)


def main():
    
    sampler = init_cond_sampler()
    pt = ParamPerturb(1e-4, random_seed=9999)
    # pt = None
    make_trajectory_ensemble(100, use_multiprocessing=True, use_tqdm=True, init_conds=sampler(), param_transform=pt)


if __name__ == '__main__':
    main()