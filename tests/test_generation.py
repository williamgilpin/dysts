"""
Test the models and regularizer
> python test_generation.py
"""
#!/usr/bin/env python
import os
import numpy as np
import unittest

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKING_DIR)

import sys

sys.path.insert(1, os.path.join(WORKING_DIR, "dysts"))
from dysts.flows import Lorenz


# class TestUtilities(unittest.TestCase):
#     """
#     Tests helper utilities
#     """
#     def test_fourier(self):
#         """
#         Test discrete fourier analysis
#         """
#         data_test = np.ones((1000, 3))
#         assert True


class TestModels(unittest.TestCase):
    """
    Testsintegration and models
    """
    def test_trajectory(self):
        """
        Test generating a trajectory
        """
        model = Lorenz()
        sol = model.make_trajectory(100)
        assert sol.shape == (1000, 3), "Generated time series has the wrong shape"
        
    def test_trajectory_noise(self):
        """
        Test generating a trajectory with stochasticity
        """
        model = Lorenz()
        sol = model.make_trajectory(100, noise=0.01)
        assert sol.shape == (1000, 3), "Generated time series has the wrong shape"
        
#     def test_trajectory_ensemble(self):
#         """
#         Test generating a trajectory with stochasticity
#         """
#         model = Lorenz()
#         sol = model.make_trajectory(100, noise=0.01)
#         assert sol.shape == (1000, 3), "Generated time series has the wrong shape"

if __name__ == "__main__":
    unittest.main()