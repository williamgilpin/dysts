import unittest

import numpy as np

from dysts.sampling import GaussianParamSampler, OnAttractorInitCondSampler
from dysts.systems import make_trajectory_ensemble


class TestTrajectoryEnsemble(unittest.TestCase):
    def setUp(self):
        self.ic_sampler = OnAttractorInitCondSampler(
            reference_traj_length=100,
            reference_traj_transient=10,
            random_seed=9999,
            verbose=True,
        )
        self.pt_sampler = GaussianParamSampler(
            random_seed=9999, scale=1e-4, verbose=True
        )

    def test_ensemble_generation(self):
        sols = make_trajectory_ensemble(
            4096,
            resample=True,
            pts_per_period=64,
            use_multiprocessing=True,
            ic_transform=self.ic_sampler,
            param_transform=self.pt_sampler,
            sys_class="continuous_no_delay",
            rng=self.ic_sampler.rng,
            standardize=True,
            embedding_dim=2,
        )
        self.assertTrue(len(sols) > 0)

    def test_initial_conditions(self):
        num_ic_trials = 4
        trajs = []
        for _ in range(num_ic_trials):
            sols = make_trajectory_ensemble(
                1024,
                resample=True,
                pts_per_period=128,
                use_multiprocessing=False,
                ic_transform=self.ic_sampler,
                subset=["Lorenz"],
                rng=self.ic_sampler.rng,
                standardize=True,
            )
            trajs.append(sols["Lorenz"])

        self.assertEqual(len(trajs), num_ic_trials)

        for traj in trajs:
            self.assertIsInstance(traj, np.ndarray)
            self.assertEqual(traj.shape[0], 1024)
            self.assertFalse(np.any(np.isnan(traj)))
            self.assertTrue(np.all(traj >= -1) and np.all(traj <= 1))

    def test_parameter_perturbations(self):
        num_ic_trials = 4
        trajs = []
        for _ in range(num_ic_trials):
            sols = make_trajectory_ensemble(
                1024,
                resample=True,
                pts_per_period=128,
                use_multiprocessing=False,
                param_transform=self.pt_sampler,
                subset=["Lorenz"],
                rng=self.pt_sampler.rng,
                standardize=True,
            )
            trajs.append(sols["Lorenz"])

        self.assertEqual(len(trajs), num_ic_trials)

        for traj in trajs:
            self.assertIsInstance(traj, np.ndarray)
            self.assertEqual(traj.shape[0], 1024)
            self.assertFalse(np.any(np.isnan(traj)))
            self.assertTrue(np.all(traj >= -1) and np.all(traj <= 1))


if __name__ == "__main__":
    unittest.main()
