import os
import warnings
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import dysts.flows as dfl
from dysts.sampling import BaseSampler, GaussianParamSampler, OnAttractorInitCondSampler
from dysts.systems import get_attractor_list
from dysts.utils.utils import standardize_ts

FIGS_DIR = "tests/figs"


def test_trajectory():
    """visual check for all continuous systems"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for sys_name in get_attractor_list():
        sys = getattr(dfl, sys_name)()
        sol = sys.make_trajectory(1024, pts_per_period=1024 // 10, standardize=True)
        ax.set_title(sys_name)
        ax.plot(*sol.T[:3])
        plt.pause(0.1)
        plt.cla()


def main(
    dyst_name: str = "Lorenz",
    num_points: int = 1024,
    num_periods: int = 5,
    num_ics: int = 3,
    num_param_perturbs: int = 3,
    ic_transform: Optional[BaseSampler] = None,
    param_transform: Optional[BaseSampler] = None,
    standardize: bool = True,
    use_ref_traj: bool = False,
):
    sys = getattr(dfl, dyst_name)()

    all_sols = defaultdict(list)

    for param_idx in range(num_param_perturbs):
        if param_transform is not None:
            sys.transform_params(param_transform)

            if standardize and use_ref_traj:
                # make a reference trajectory to get the mean and std
                print("Computing reference trajectory for standardization...")
                num_periods_ref = 40
                pts_per_period_ref = 1024
                n_ref = pts_per_period_ref * num_periods_ref
                ref_traj = sys.make_trajectory(
                    n_ref,
                    pts_per_period=pts_per_period_ref,
                    standardize=False,
                    resample=True,
                )
                new_mean = ref_traj.mean(axis=0)
                new_std = ref_traj.std(axis=0)
                print("Reference trajectory computed")
                print(f"New mean: {new_mean}")
                print(f"New std: {new_std}")
                sys.set_statistics(new_mean, new_std)
        else:
            print(f"No param_transform provided, using default params for {dyst_name}")
        # keep list of ics trajectories for each parameter perturbation
        pt_sols_all_ics = []
        for ic_idx in range(num_ics):
            if ic_transform is not None:
                sys.transform_ic(ic_transform)
            else:
                print(f"No ic_transform provided, using default ic for {dyst_name}")
            sol = sys.make_trajectory(
                num_points,
                pts_per_period=num_points // num_periods,
                standardize=True if use_ref_traj and standardize else False,
                resample=True,
            )
            if sol is None:
                warnings.warn(
                    f"Failed to generate complete trajectory for sample {sys.ic} and param {sys.params}"
                )
                continue
            print(f"Generated trajectory with shape {sol.shape}")

            pt_sols_all_ics.append(sol)

        if standardize and not use_ref_traj:
            pt_sols_numpy = np.array(pt_sols_all_ics)
            print(
                f"Standardizing trajectory with shape {pt_sols_numpy.shape} using standardize_ts"
            )
            pt_sols_numpy = standardize_ts(pt_sols_numpy)
            pt_sols_all_ics = list(pt_sols_numpy)

        all_sols[param_idx].extend(pt_sols_all_ics)

    # Plot all trajectories
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.rcParams["axes.prop_cycle"]
    colors = colors.by_key()["color"]
    for param_idx, pt_sols in all_sols.items():
        # sols_param is a list of trajectories for a given params choice
        for sol in pt_sols:
            # Plot X, Y, Z
            ax.plot(*sol.T[:3], alpha=0.5, linewidth=0.5, color=colors[param_idx])
            ax.scatter(
                *sol.T[:3, 0], marker="*", s=50, alpha=0.8, color=colors[param_idx]
            )  # type: ignore
    # plt.show()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    plt.title(dyst_name)
    plt.savefig(os.path.join(FIGS_DIR, "test_standardization.png"), dpi=300)
    plt.close()

    # # Compute all trajectory statistics and analysis quantities
    # all_means = []
    # all_stds = []
    # for sol in all_sols:
    #     mean = sol.mean(axis=0)
    #     std = sol.std(axis=0)
    #     all_means.append(mean)
    #     all_stds.append(std)

    # all_means = np.array(all_means)
    # all_stds = np.array(all_stds)
    # print(f"All means: {all_means}")
    # print(f"All stds: {all_stds}")

    # # # check if all values of all_means are all close to each other
    # # assert np.allclose(all_means, all_means[0], atol=1e-2)
    # # assert np.allclose(all_stds, all_stds[0], atol=1e-2)


if __name__ == "__main__":
    os.makedirs(FIGS_DIR, exist_ok=True)
    rseed = 9999
    onattractor_sampler = OnAttractorInitCondSampler(
        reference_traj_length=100,
        reference_traj_transient=10,
        random_seed=rseed,
        verbose=True,
    )
    pt = GaussianParamSampler(random_seed=rseed, scale=1e-1, verbose=True)

    main(
        dyst_name="Lorenz",
        num_points=1024,
        num_periods=5,
        num_ics=5,
        num_param_perturbs=2,
        ic_transform=onattractor_sampler,
        param_transform=pt,
        standardize=True,
        use_ref_traj=False,
    )
