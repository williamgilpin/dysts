import matplotlib.pyplot as plt

from dysts.sampling import (
    GaussianInitialConditionSampler,
    GaussianParamSampler,
    OnAttractorInitCondSampler,
)
from dysts.systems import (
    make_trajectory_ensemble,
)


def test_ensemble(ic_sampler, param_sampler):
    sols = make_trajectory_ensemble(
        4096,
        resample=True,
        pts_per_period=64,
        use_multiprocessing=True,
        ic_transform=ic_sampler,
        param_transform=param_sampler,
        sys_class="continuous_no_delay",
        rng=ic_sampler.rng,
        standardize=True,
        embedding_dim=2,
    )

    plt.figure()
    for sys, traj in sols.items():
        plt.plot(*traj.T[:2], label=sys)
    plt.legend()
    plt.show()


def main():
    sampler = GaussianInitialConditionSampler(random_seed=9999, scale=1e-1)
    pt = GaussianParamSampler(random_seed=9999, scale=1e-4)
    onattractor = OnAttractorInitCondSampler(
        reference_traj_length=100, reference_traj_transient=10, random_seed=9999
    )

    test_ensemble(onattractor, pt)


if __name__ == "__main__":
    main()
