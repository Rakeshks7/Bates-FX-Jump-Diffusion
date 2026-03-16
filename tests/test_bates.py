import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bates_simulation import bates_model_mc

class TestBatesModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.S0 = 1.0
        cls.T = 1.0
        cls.r = 0.05
        cls.v0 = 0.04
        cls.kappa = 2.0
        cls.theta = 0.04
        cls.sigma = 0.2
        cls.rho = -0.5
        cls.lambda_j = 3.0
        cls.mu_j = -0.05
        cls.sigma_j = 0.1
        cls.N = 252

        cls.M = 50000 

        cls.S_paths, cls.v_paths = bates_model_mc(
            cls.S0, cls.T, cls.r, cls.v0, cls.kappa, cls.theta, 
            cls.sigma, cls.rho, cls.lambda_j, cls.mu_j, cls.sigma_j, 
            cls.N, cls.M
        )

    def test_martingale_property(self):
        discounted_expectation = np.mean(self.S_paths[-1]) * np.exp(-self.r * self.T)

        self.assertAlmostEqual(
            discounted_expectation, 
            self.S0, 
            places=1,
            msg=f"Martingale property failed. Expected ~{self.S0}, got {discounted_expectation:.4f}"
        )

    def test_variance_non_negativity(self):
        min_variance = np.min(self.v_paths)
        self.assertGreaterEqual(
            min_variance, 
            0.0, 
            msg=f"Feller condition truncation failed. Minimum variance is {min_variance}"
        )

    def test_jump_presence(self):
        max_price = np.max(self.S_paths[-1])
        self.assertGreater(
            max_price, 
            1.2, 
            msg="Max asset price is too low, suggesting jumps are not being generated."
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)