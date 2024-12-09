import random
import unittest

import numpy as np

import dysts.flows as dfl
from dysts.sampling import GaussianInitialConditionSampler, GaussianParamSampler
from dysts.systems import get_attractor_list, make_trajectory_ensemble


class TestTrajectoryEnsemble(unittest.TestCase):
    def test_ensemble_generation_no_multiprocessing(self):
        sols = make_trajectory_ensemble(
            256,
            resample=True,
            pts_per_period=64,
            use_multiprocessing=False,
            subset=random.sample(
                get_attractor_list(sys_class="continuous_no_delay"), 4
            ),
        )
        self.assertTrue(len(sols) > 0)
        self.assertTrue(all(arr is not None for arr in sols.values()))

    def test_ensemble_generation_with_standardization(self):
        sols = make_trajectory_ensemble(
            256,
            resample=True,
            pts_per_period=64,
            use_multiprocessing=True,
            subset=random.sample(
                get_attractor_list(sys_class="continuous_no_delay"), 4
            ),
            standardize=True,
        )
        self.assertTrue(len(sols) > 0)
        self.assertTrue(all(arr is not None for arr in sols.values()))

    def test_ensemble_generation_without_standardization(self):
        sols = make_trajectory_ensemble(
            256,
            resample=True,
            pts_per_period=64,
            use_multiprocessing=True,
            subset=random.sample(
                get_attractor_list(sys_class="continuous_no_delay"), 4
            ),
            standardize=False,
        )
        self.assertTrue(len(sols) > 0)
        self.assertTrue(all(arr is not None for arr in sols.values()))

    def test_trajectories(self):
        num_trials = 2
        trajs = []
        attractors = random.sample(get_attractor_list("continuous"), 50)
        for _ in range(num_trials):
            sols = make_trajectory_ensemble(
                256,
                resample=True,
                pts_per_period=64,
                use_multiprocessing=True,
                subset=attractors,
                standardize=False,
            )
            for system_name in attractors:
                traj = sols[system_name]
                self.assertTrue(traj is None or isinstance(traj, np.ndarray))
                if traj is not None:
                    self.assertEqual(traj.shape[0], 256)
                    self.assertFalse(np.any(np.isnan(traj)))
            trajs.append(traj)

        self.assertEqual(len(trajs), num_trials)

    def test_ensemble_generation_initial_condition_sampling(self):
        ic_sampler = GaussianInitialConditionSampler(
            scale=1e-4, random_seed=random.randint(0, 1000000)
        )
        system_sample = random.sample(get_attractor_list(sys_class="continuous"), 4)
        systems = [getattr(dfl, sys)() for sys in system_sample]
        unperturbed_sols = make_trajectory_ensemble(
            256,
            pts_per_period=64,
            use_multiprocessing=True,
            subset=systems,
        )

        for sys in systems:
            sys.transform_ic(ic_sampler)

        perturbed_sols = make_trajectory_ensemble(
            256,
            pts_per_period=64,
            use_multiprocessing=True,
            subset=systems,
        )

        for system_name in system_sample:
            unperturbed_traj = unperturbed_sols[system_name]
            perturbed_traj = perturbed_sols[system_name]
            self.assertTrue(unperturbed_traj is not None)
            self.assertTrue(perturbed_traj is not None)
            self.assertEqual(unperturbed_traj.shape, perturbed_traj.shape)
            self.assertFalse(np.allclose(unperturbed_traj, perturbed_traj))

    def test_ensemble_generation_parameter_sampling(self):
        param_sampler = GaussianParamSampler(
            scale=1e-4, random_seed=random.randint(0, 1000000)
        )
        system_sample = random.sample(get_attractor_list(sys_class="continuous"), 4)
        system_sample += ["InteriorSquirmer"]
        systems = [getattr(dfl, sys)() for sys in system_sample]
        unperturbed_sols = make_trajectory_ensemble(
            256,
            pts_per_period=64,
            use_multiprocessing=True,
            subset=systems,
        )

        for sys in systems:
            sys.transform_params(param_sampler)

        perturbed_sols = make_trajectory_ensemble(
            256,
            pts_per_period=64,
            use_multiprocessing=True,
            subset=systems,
        )

        for system_name in system_sample:
            unperturbed_traj = unperturbed_sols[system_name]
            perturbed_traj = perturbed_sols[system_name]
            self.assertTrue(unperturbed_traj is not None)
            self.assertTrue(perturbed_traj is not None)
            self.assertEqual(unperturbed_traj.shape, perturbed_traj.shape)
            self.assertFalse(np.allclose(unperturbed_traj, perturbed_traj))


if __name__ == "__main__":
    unittest.main()
