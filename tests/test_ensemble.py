from dataclasses import dataclass

from dysts.systems import (
    CallableWorker,
    GaussianParamSampler,
    gaussian_init_cond_sampler,
    make_trajectory_ensemble,
)


@dataclass
class ParameterRNGWorker(CallableWorker):
    def __call__(self, rng, *args, **kwargs):
        param_sampler = args[-1]
        param_sampler.set_rng(rng)
        return self.fn([*args[:-1], param_sampler], **kwargs)


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
        worker_class=ParameterRNGWorker,
    )
    print(sols)


if __name__ == "__main__":
    main()
