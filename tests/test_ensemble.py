import random
import unittest

import numpy as np

from dysts.sampling import GaussianParamSampler, OnAttractorInitCondSampler
from dysts.systems import get_attractor_list, make_trajectory_ensemble


class TestTrajectoryEnsemble(unittest.TestCase):
    def setUp(self):
        self.ic_sampler = OnAttractorInitCondSampler(
            reference_traj_length=100,
            reference_traj_transient=10,
            random_seed=9999,
            verbose=True,
        )
        self.pt_sampler = GaussianParamSampler(
            random_seed=9999, scale=1e-1, verbose=True
        )

    def test_ensemble_generation_with_standardization(self):
        sols = make_trajectory_ensemble(
            1024,
            resample=True,
            pts_per_period=128,
            use_multiprocessing=True,
            ic_transform=self.ic_sampler,
            subset=random.sample(
                get_attractor_list(sys_class="continuous_no_delay"), 4
            ),
            rng=self.ic_sampler.rng,
            standardize=True,
            embedding_dim=2,
        )
        self.assertTrue(len(sols) > 0)
        self.assertTrue(all(arr is not None for arr in sols.values()))

    def test_ensemble_generation_without_standardization(self):
        sols = make_trajectory_ensemble(
            1024,
            resample=True,
            pts_per_period=128,
            use_multiprocessing=True,
            ic_transform=self.ic_sampler,
            subset=random.sample(
                get_attractor_list(sys_class="continuous_no_delay"), 4
            ),
            rng=self.ic_sampler.rng,
            standardize=False,
            embedding_dim=2,
        )
        self.assertTrue(len(sols) > 0)
        self.assertTrue(all(arr is not None for arr in sols.values()))

    def test_initial_conditions(self):
        num_ic_trials = 2
        trajs = []
        attractors = random.sample(get_attractor_list("continuous"), 50)
        for _ in range(num_ic_trials):
            sols = make_trajectory_ensemble(
                1024,
                resample=True,
                pts_per_period=128,
                use_multiprocessing=False,
                ic_transform=self.ic_sampler,  # TODO: some NaNs seen in trajectories, even though this should be on attractor
                subset=attractors,
                rng=self.ic_sampler.rng,
                standardize=False,
            )
            for system_name in attractors:
                traj = sols[system_name]
                self.assertIsInstance(traj, np.ndarray)
                self.assertEqual(traj.shape[0], 1024)
                self.assertFalse(np.any(np.isnan(traj)))
            trajs.append(traj)

        self.assertEqual(len(trajs), num_ic_trials)

    # TODO: some param perturbations will yield NaNs in the trajectory, need to add tests
    def test_parameter_perturbations(self):
        num_ic_trials = 2
        trajs = []
        attractors = random.sample(get_attractor_list("continuous"), 50)
        for _ in range(num_ic_trials):
            sols = make_trajectory_ensemble(
                1024,
                resample=True,
                pts_per_period=128,
                use_multiprocessing=False,
                param_transform=self.pt_sampler,
                subset=attractors,
                rng=self.pt_sampler.rng,
                standardize=False,
            )
            for system_name in attractors:
                traj = sols[system_name]
                self.assertIsInstance(traj, np.ndarray)
                self.assertEqual(traj.shape[0], 1024)
                self.assertFalse(np.any(np.isnan(traj)))
            trajs.append(traj)

        self.assertEqual(len(trajs), num_ic_trials)


if __name__ == "__main__":
    unittest.main()
