from dysts.systems import (
    gaussian_init_cond_sampler,
    gaussian_parameter_sampler,
    make_trajectory_ensemble,
)


def main():
    sampler = gaussian_init_cond_sampler()
    pt = gaussian_parameter_sampler(1e-4, random_seed=9999)
    make_trajectory_ensemble(
        100,
        use_multiprocessing=True,
        use_tqdm=True,
        init_conds=sampler(),
        param_transform=pt,
    )


if __name__ == "__main__":
    main()
