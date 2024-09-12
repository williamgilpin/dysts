from dysts.systems import (
    GaussianParamSampler,
    gaussian_init_cond_sampler,
    make_trajectory_ensemble,
)


def main():
    sampler = gaussian_init_cond_sampler()
    pt = GaussianParamSampler(random_seed=9999, scale=1e-2)
    sols = make_trajectory_ensemble(
        100,
        use_multiprocessing=True,
        use_tqdm=True,
        init_conds=sampler(),
        param_transform=pt,
        sys_class="delay",
        rng=pt.rng,
    )
    print(sols)


if __name__ == "__main__":
    main()
