import matplotlib.pyplot as plt

from dysts.systems import (
    GaussianParamSampler,
    gaussian_init_cond_sampler,
    make_trajectory_ensemble,
)


def main():
    sampler = gaussian_init_cond_sampler()
    pt = GaussianParamSampler(random_seed=9999, scale=1e-2)
    sols = make_trajectory_ensemble(
        1024,
        use_multiprocessing=True,
        use_tqdm=True,
        init_conds=sampler(),
        param_transform=None,
        subset=["Lorenz"],
        rng=pt.rng,
        standardize=True,
    )
    traj = sols["Lorenz"]
    print(traj.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*traj.T[:3])
    plt.show()


if __name__ == "__main__":
    main()
