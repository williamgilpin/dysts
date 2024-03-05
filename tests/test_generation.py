"""
Test the models and regularizer
> python test_generation.py
"""
#!/usr/bin/env python
import os
import numpy as np
import unittest

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKING_DIR, 'tests', 'test_data')
print(WORKING_DIR)

import sys

sys.path.insert(1, os.path.join(WORKING_DIR, "dysts"))
from dysts.flows import Lorenz
from dysts.base import make_trajectory_ensemble, get_attractor_list
import dysts.flows as dfl


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
        assert sol.shape == (100, 3), "Generated time series has the wrong shape"
        
    def test_trajectory_noise(self):
        """
        Test generating a trajectory with stochasticity
        """
        model = Lorenz()
        sol = model.make_trajectory(100, noise=0.01)
        assert sol.shape == (100, 3), "Generated time series has the wrong shape"
    
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

        
    def test_precomputed(self):
        """
        Test loading a precomputed time series for a single system
        """
        eq = Lorenz()
        tpts, sol = eq.load_trajectory(
            subsets="test", 
            noise=False, 
            granularity="fine", 
            standardize=True, 
            return_times=True
        )
        assert sol.shape == (1200, 3), "Generated time series has the wrong shape"
        assert tpts.shape == (1200,), "Time indices have the wrong shape"

class TestMakeTrajectoryEnsemble(unittest.TestCase):
    def test_ensemble(self):
        # Test that the function returns a dictionary with the correct keys
        n = 100
        subset = ["Lorenz", "Rossler"]
        random_state = 42
        kwargs = {"method": "Radau"}
        ensemble = make_trajectory_ensemble(n, subset=subset, random_state=random_state, **kwargs)
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
        random_state = 42
        kwargs = {"method": "Radau"}
        with self.assertWarns(UserWarning):
            make_trajectory_ensemble(n, subset=subset, use_multiprocessing=True, random_state=random_state, **kwargs)


class TestJacobian(unittest.TestCase):
    """Perform a grad check to ensure that the Jacobian is implemented correctly"""
    def test_all_jacobians(self):

        eps = 1e-8

        equation_names = get_attractor_list()
        for name in equation_names:
            eq = getattr(dfl, name)()
            if eq.jac(eq.ic, 0) is None:
                continue
            
            jac_analytic = eq.jac(eq.ic, 0)

            d = len(eq.ic)
            jac_fd = np.zeros((d, d))
            for i in range(d):
                ei = np.zeros(d)
                ei[i] = 1
                jac_fd[:, i] = (np.array(eq.rhs(eq.ic + eps*ei, 0)) - np.array(eq.rhs(eq.ic - eps*ei, 0)))/(2 * eps)

            self.assertTrue(np.allclose(jac_analytic, jac_fd, atol=1e-5), f"Jacobian for {name} is incorrect")
            

        
if __name__ == "__main__":
    unittest.main()