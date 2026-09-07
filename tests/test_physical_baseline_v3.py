import unittest

import numpy as np

from evaluation.evaluate_physical_baseline_v3 import (
    IMMFilter,
    initial_covariance,
    kalman_update,
    measurement_covariance,
    process_covariance,
)


class PhysicalBaselineV3Test(unittest.TestCase):
    def test_covariances_are_symmetric_psd(self):
        r = measurement_covariance([1000.0, 500.0], 5.0, 0.2)
        p = initial_covariance(r, 2.0)
        q = process_covariance(2.0, 3.0)
        for covariance in (r, p, q):
            self.assertTrue(np.allclose(covariance, covariance.T))
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-8)

    def test_zero_innovation_does_not_move_mean(self):
        state = np.array([10.0, 1.0, 20.0, -1.0])
        measurement = state[[0, 2]]
        updated, covariance, _, _ = kalman_update(
            state, np.eye(4), measurement, np.eye(2)
        )
        self.assertTrue(np.allclose(updated, state))
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.0)

    def test_imm_probabilities_normalize_and_covariance_is_psd(self):
        imm = IMMFilter(
            np.zeros(4), np.eye(4), 2.0, (0.5, 5.0),
            [[0.97, 0.03], [0.05, 0.95]], [0.8, 0.2],
        )
        imm.predict()
        _, covariance, _, _ = imm.update(np.array([20.0, -10.0]), np.eye(2))
        self.assertAlmostEqual(float(imm.mode_probability.sum()), 1.0)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.0)

    def test_imm_reports_prior_innovation(self):
        imm = IMMFilter(
            np.zeros(4), np.eye(4), 2.0, (0.5, 5.0),
            [[0.97, 0.03], [0.05, 0.95]], [0.8, 0.2],
        )
        prior, _ = imm.predict()
        measurement = np.array([20.0, -10.0])
        _, _, innovation, _ = imm.update(measurement, np.eye(2))
        self.assertTrue(
            np.allclose(innovation, measurement - prior[[0, 2]])
        )


if __name__ == "__main__":
    unittest.main()
