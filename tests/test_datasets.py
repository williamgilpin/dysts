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

from dysts.datasets import load_dataset

from dysts.base import get_attractor_list
class TestUtilities(unittest.TestCase):
    """
    Tests helper utilities
    """
#     def test_fourier(self):
#         """
#         Test discrete fourier analysis
#         """
#         data_test = np.ones((1000, 3))
#         assert True
        
    def test_attractor_list(self):
        """
        Test fetching list of attractors
        """
        assert len(get_attractor_list()) > 130
        
class TestDatasets(unittest.TestCase):
    """
    Tests models
    """
    def test_univariate(self):
        """
        Test univariate data loader
        """
        data = load_dataset(data_format="numpy", standardize=True)
        assert data.shape[0] > 131, "Imported time series collection has the wrong shape"
        assert data.shape[-1] == 1000, "Imported time series collection has the wrong shape"
        
    def test_multivariate(self):
        """
        Test multivariate data loader
        """
        data = load_dataset(subsets="train", univariate=False, standardize=False)
        assert data.dataset["Rossler"]["values"].shape == (1000, 3), "Imported time series collection has the wrong shape"

if __name__ == "__main__":
    unittest.main()