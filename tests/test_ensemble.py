import matplotlib.pyplot as plt

from dysts.sampling import GaussianInitialConditionSampler, GaussianParamSampler
from dysts.systems import (
    make_trajectory_ensemble,
)


def main():
    sampler = GaussianInitialConditionSampler(random_seed=9999, scale=1e-4)
    pt = GaussianParamSampler(random_seed=9999, scale=1e-2)
    sols = make_trajectory_ensemble(
        100,
        use_multiprocessing=False,
        use_tqdm=True,
        init_conds=sampler,
        param_transform=pt,
        sys_class="delay",
        rng=pt.rng,
    )
    print(sols)

    plt.figure()
    for sys, traj in sols.items():
        plt.plot(*traj.T, label=sys)
    plt.show()


if __name__ == "__main__":
    main()
