import matplotlib.pyplot as plt

from dysts.sampling import GaussianInitialConditionSampler, GaussianParamSampler
from dysts.systems import (
    make_trajectory_ensemble,
)


def main():
    sampler = GaussianInitialConditionSampler(random_seed=9999, scale=1e-1)
    pt = GaussianParamSampler(random_seed=9999, scale=1e-4)
    sols = make_trajectory_ensemble(
        4096,
        resample=True,
        pts_per_period=64,
        use_multiprocessing=True,
        use_tqdm=True,
        ic_transform=sampler,
        param_transform=pt,
        sys_class="delay",
        rng=pt.rng,
        standardize=True,
        embedding_dim=2,
    )

    plt.figure()
    for sys, traj in sols.items():
        plt.plot(traj[:, 0], traj[:, 1], label=sys)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
