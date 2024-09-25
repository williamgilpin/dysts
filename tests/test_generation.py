"""
Test the models and regularizer
> python test_generation.py
"""

#!/usr/bin/env python
import os
import sys
import unittest

import numpy as np

import dysts.flows as dfl
from dysts.flows import Lorenz
from dysts.systems import get_attractor_list, make_trajectory_ensemble

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKING_DIR, "tests", "test_data")
print(WORKING_DIR)

sys.path.insert(1, os.path.join(WORKING_DIR, "dysts"))


DEFAULT_RNG = np.random.default_rng(99)


class TestModels(unittest.TestCase):
    """
    Tests integration and models
    """

    def test_trajectory(self):
        """
        Test generating a trajectory
        """
        model = Lorenz()
        sol = model.make_trajectory(100)
        assert sol is not None, "Generated trajectory is None"
        assert sol.shape == (100, 3), "Generated time series has the wrong shape"  # type: ignore

    def test_trajectory_noise(self):
        """
        Test generating a trajectory with stochasticity
        """
        # check if sdeint is installed
        try:
            import sdeint  # type: ignore
        except ImportError:
            self.skipTest("sdeint is not installed")

        model = Lorenz()
        sol = model.make_trajectory(100, noise=0.01)
        assert sol is not None, "Generated trajectory is None"
        assert sol.shape == (100, 3), "Generated time series has the wrong shape"  # type: ignore

    ## Test removed due to the need to re-generate the reference data every time
    ## a new system is added to the database
    # def test_ensemble(self):
    #     """
    #     Test all systems in the database
    #     """
    #     all_trajectories = make_trajectory_ensemble(5, method="Radau", resample=True)
    #     assert len(all_trajectories.keys()) >= 131

    #     xvals = np.array([all_trajectories[key][:, 0] for key in all_trajectories.keys()])
    #     xvals_reference = np.load(os.path.join(DATA_PATH, "all_trajectories.npy"), allow_pickle=True)
    #     diff_names = np.array(list(all_trajectories.keys()))[np.sum(np.abs(xvals - xvals_reference), axis=1) > 0]
    #     assert np.allclose(xvals, xvals_reference), "Generated trajectories do not match reference values for system {}".format(diff_names)

    # # TODO: make sure a data file exists in the data folder that is referenced
    # def test_precomputed(self):
    #     """
    #     Test loading a precomputed time series for a single system
    #     """
    #     dyst_name = "Lorenz"
    #     eq = getattr(dfl, dyst_name)()
    #     dyst_data_path = os.path.join(DATA_PATH, f"{dyst_name}.json.gz")
    #     if not os.path.exists(dyst_data_path):
    #         raise FileNotFoundError(f"File {dyst_data_path} does not exist")

    #     tpts, sol = eq.load_trajectory(
    #         data_path=DATA_PATH,
    #         standardize=True,
    #         return_times=True,
    #     )
    #     assert sol.shape == (1200, 3), "Generated time series has the wrong shape"
    #     assert tpts.shape == (1200,), "Time indices have the wrong shape"


class TestMakeTrajectoryEnsemble(unittest.TestCase):
    def test_ensemble(self):
        # Test that the function returns a dictionary with the correct keys
        n = 100
        subset = ["Lorenz", "Rossler"]
        kwargs = {"method": "Radau"}
        ensemble = make_trajectory_ensemble(
            n,
            subset=subset,
            rng=DEFAULT_RNG,
            **kwargs,  # type: ignore
        )
        self.assertIsInstance(ensemble, dict)
        self.assertEqual(set(ensemble.keys()), set(subset))

        # Test that the function returns the correct number of timepoints
        for key in ensemble:
            self.assertEqual(ensemble[key].shape[0], n)

        # Test that the function returns the correct shape of the solution array
        for key in ensemble:
            self.assertEqual(ensemble[key].shape[1], len(getattr(dfl, key)().ic))

        # Test that the function returns the correct shape of the solution array
        for key in ensemble:
            self.assertEqual(ensemble[key].shape[0], n)

    def test_multiprocessing(self):
        # Test that the function returns a warning when multiprocessing is set to True
        n = 100
        subset = ["Lorenz", "Rossler"]
        kwargs = {"method": "Radau"}
        with self.assertRaises(Exception):
            with self.assertWarns(UserWarning):
                make_trajectory_ensemble(
                    n,
                    subset=subset,
                    use_multiprocessing=True,
                    rng=DEFAULT_RNG,
                    **kwargs,  # type: ignore
                )


class TestJacobian(unittest.TestCase):
    """Perform a grad check to ensure that the Jacobian is implemented correctly"""

    def test_all_jacobians(self):
        eps = 1e-8

        equation_names = get_attractor_list()
        for name in equation_names:
            eq = getattr(dfl, name)()

            # skip if no analytic Jacobian implemented
            if not hasattr(eq, "jac"):
                continue
            if eq.jac(eq.ic, 0) is None:
                continue

            jac_analytic = eq.jac(eq.ic, 0)

            d = len(eq.ic)
            jac_fd = np.zeros((d, d))
            for i in range(d):
                ei = np.zeros(d)
                ei[i] = 1
                jac_fd[:, i] = (
                    np.array(eq.rhs(eq.ic + eps * ei, 0))
                    - np.array(eq.rhs(eq.ic - eps * ei, 0))
                ) / (2 * eps)

            self.assertTrue(
                np.allclose(jac_analytic, jac_fd, atol=1e-5),
                f"Jacobian for {name} is incorrect",
            )


if __name__ == "__main__":
    unittest.main()
