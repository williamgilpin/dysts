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

from dysts.metrics import mase, mse, mae, coefficient_of_variation, marre, ope, rmsle, r2_score, mape, smape, spearman, pearson

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.2, 4.1, 5.1])
        self.y_train = np.array([0.9, 1.8, 2.9, 3.8, 4.9])

    def test_mase(self):
        result = mase(self.y_true, self.y_pred, self.y_train)
        self.assertAlmostEqual(result, 0.10434, places=2)

    def test_mse(self):
        result = mse(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.01599, places=2)

    def test_mae(self):
        result = mae(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.1199, places=2)

    def test_coefficient_of_variation(self):
        result = coefficient_of_variation(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 3.26598632, places=2)

    def test_marre(self):
        result = marre(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 2.99999, places=2)

    def test_ope(self):
        result = ope(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.099999, places=2)

    def test_rmsle(self):
        result = rmsle(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.04934, places=2)

    def test_r2_score(self):
        result = r2_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.992, places=2)

    def test_mape(self):
        result = mape(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 5.2333, places=2)

    def test_smape(self):
        result = smape(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 5.110, places=2)
        
    def test_spearman(self):
        result = spearman(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.9999, places=2)
        
    def test_pearson(self):
        result = pearson(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.9999, places=2)
            
    
if __name__ == "__main__":
    unittest.main()