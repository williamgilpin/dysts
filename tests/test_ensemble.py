import random
import unittest

import numpy as np

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
        """
        TODO: add tests for initial condition sampling
        """
        pass

    def test_ensemble_generation_parameter_sampling(self):
        """
        TODO: add tests for parameter sampling
        """
        pass


if __name__ == "__main__":
    unittest.main()
