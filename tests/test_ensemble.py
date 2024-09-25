# import matplotlib.pyplot as plt

# from dysts.sampling import (
#     GaussianParamSampler,
#     OnAttractorInitCondSampler,
# )
# from dysts.systems import (
#     make_trajectory_ensemble,
# )

# import unittest

# class TestEnsemble(unittest.TestCase):
#     def test_ensemble(self):
#         ic_sampler = OnAttractorInitCondSampler(
#             reference_traj_length=100,
#             reference_traj_transient=10,
#             random_seed=9999,
#             verbose=True,
#         )
#         pt_sampler = GaussianParamSampler(random_seed=9999, scale=1e-4, verbose=True)
#         sols = make_trajectory_ensemble(
#             4096,
#             resample=True,
#             pts_per_period=64,
#             use_multiprocessing=True,
#             ic_transform=ic_sampler,
#             param_transform=pt_sampler,
#             sys_class="continuous_no_delay",
#             rng=ic_sampler.rng,
#             standardize=True,
#             embedding_dim=2,
#         )
#         self.assertTrue(len(sols) > 0)

#     def test_init_conds(self):
#         ic_sampler = OnAttractorInitCondSampler(
#             reference_traj_length=100,
#             reference_traj_transient=10,
#             random_seed=9999,
#             verbose=True,
#         )
#         test_init_conds(ic_sampler)



# # def test_ensemble(ic_sampler, param_sampler):
# #     sols = make_trajectory_ensemble(
# #         4096,
# #         resample=True,
# #         pts_per_period=64,
# #         use_multiprocessing=True,
# #         ic_transform=ic_sampler,
# #         param_transform=param_sampler,
# #         sys_class="continuous_no_delay",
# #         rng=ic_sampler.rng,
# #         standardize=True,
# #         embedding_dim=2,
# #     )

# #     plt.figure()
# #     for sys, traj in sols.items():
# #         plt.plot(*traj.T[:2], label=sys)
# #     plt.legend()
# #     plt.show()


# def test_init_conds(ic_sampler, num_ic_trials=4):
#     print("Testing initial condition sampling")
#     trajs = []
#     for _ in range(num_ic_trials):
#         sols = make_trajectory_ensemble(
#             1024,
#             resample=True,
#             pts_per_period=128,
#             use_multiprocessing=False,
#             ic_transform=ic_sampler,
#             subset=["Lorenz"],
#             rng=ic_sampler.rng,
#             standardize=True,
#         )
#         trajs.append(sols["Lorenz"])

#     # Assertions
#     assert len(trajs) == num_ic_trials  # Check number of trajectories
    
#     for traj in trajs:
#         assert isinstance(traj, np.ndarray)  # Ensure trajectories are numpy arrays
#         assert traj.shape[0] == 1024  # Ensure each trajectory has 1024 points
        
#         # Optional: check if the trajectories meet expected statistical properties
#         assert ~np.any(np.isnan(traj)), "Trajectory contains NaN values"
#         assert np.all(traj >= -1) and np.all(traj <= 1), "Values not in expected range after standardization"
    

# def test_param_perturbs(pt_sampler, num_ic_trials=4):
#     print("Testing parameter perturbations")
#     trajs = []
#     for _ in range(num_ic_trials):
#         sols = make_trajectory_ensemble(
#             1024,
#             resample=True,
#             pts_per_period=128,
#             use_multiprocessing=False,
#             param_transform=pt_sampler,
#             subset=["Lorenz"],
#             rng=pt_sampler.rng,
#             standardize=True,
#         )
#         trajs.append(sols["Lorenz"])

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     for traj in trajs:
#         ax.plot(*traj.T[:3], alpha=0.1)
#         ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker="*", s=100)  # type: ignore
#     plt.show()


# def main():
#     # sampler = GaussianInitialConditionSampler(random_seed=9999, scale=1e-1)
#     pt = GaussianParamSampler(random_seed=9999, scale=1e-4, verbose=True)
#     onattractor = OnAttractorInitCondSampler(
#         reference_traj_length=100,
#         reference_traj_transient=10,
#         random_seed=9999,
#         verbose=True,
#     )

#     # test_ensemble(onattractor, pt)
#     test_param_perturbs(pt)
#     test_init_conds(onattractor)


# if __name__ == "__main__":
#     main()

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
        self.pt_sampler = GaussianParamSampler(random_seed=9999, scale=1e-4, verbose=True)

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