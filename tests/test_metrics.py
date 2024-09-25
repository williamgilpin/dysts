"""
Test the models and regularizer
> python test_generation.py
"""

#!/usr/bin/env python
import os
import sys
import unittest

import numpy as np

from dysts.metrics import (
    coefficient_of_variation,
    mae,
    mape,
    marre,
    mse,
    ope,
    pearson,
    r2_score,
    rmsle,
    smape,
    spearman,
)

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKING_DIR)
sys.path.insert(1, os.path.join(WORKING_DIR, "dysts"))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.2, 4.1, 5.1])
        self.y_train = np.array([0.9, 1.8, 2.9, 3.8, 4.9])

    # # TODO: check the ground truth value for this test
    # def test_mase(self):
    #     result = mase(self.y_true, self.y_pred, self.y_train)
    #     self.assertAlmostEqual(result, 0.10434, places=2)

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
        self.assertAlmostEqual(result, 0.9999, places=2)  # type: ignore

    def test_pearson(self):
        result = pearson(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.9999, places=2)  # type: ignore

    # def test_kl_divergence(self):
    #     y_gaussian = np.random.normal(size=100)

    #     dim = 10  # must be even if we want to compare against the correlated bivariate gaussians
    #     n_samples = 1000
    #     y_multivar_gaussian = np.random.multivariate_normal(
    #         mean=np.zeros(dim), cov=np.eye(dim), size=n_samples
    #     )
    #     yp_multivar_gaussian = np.random.multivariate_normal(
    #         mean=np.zeros(dim), cov=np.eye(dim), size=n_samples
    #     )

    #     # correlated high-dim bivariate gaussians
    #     mu = np.array([0] * dim)

    #     rho1 = 0.6
    #     dim_pair_cov1 = np.array([[1, rho1], [rho1, 1]])
    #     cov1 = np.kron(np.eye(dim // 2), dim_pair_cov1)

    #     rho2 = 0.3
    #     dim_pair_cov2 = np.array([[1, rho2], [rho2, 1]])
    #     cov2 = np.kron(np.eye(dim // 2), dim_pair_cov2)

    #     y_corr_bivar_gaussians = np.random.multivariate_normal(
    #         mean=mu, cov=cov1, size=n_samples
    #     )
    #     yp_corr_bivar_gaussians = np.random.multivariate_normal(
    #         mean=mu, cov=cov2, size=n_samples
    #     )

    #     # check that the KLD estimate returns 0 for identical distributions
    #     result_same = estimate_kl_divergence(y_gaussian, y_gaussian)
    #     self.assertEqual(result_same, 0.0)
    #     result_same = estimate_kl_divergence(y_multivar_gaussian, y_multivar_gaussian)
    #     self.assertEqual(result_same, 0.0)

    #     # self-consistency check
    #     result_rho1_0 = estimate_kl_divergence(
    #         y_corr_bivar_gaussians, yp_multivar_gaussian
    #     )
    #     self.assertGreaterEqual(result_rho1_0, 0.0)
    #     result_rho1_rho2 = estimate_kl_divergence(
    #         y_corr_bivar_gaussians, yp_corr_bivar_gaussians
    #     )
    #     self.assertLessEqual(result_rho1_rho2, result_rho1_0)


if __name__ == "__main__":
    unittest.main()
