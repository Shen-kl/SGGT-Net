import unittest

import numpy as np
import torch

from evaluation.evaluate_causal_group_innovation_oracle_v3 import (
    _apply_linear,
    _roll_append,
    _solve_ridge,
    _stratified_derangement,
)


class CausalGroupInnovationOracleTest(unittest.TestCase):
    def test_roll_append_is_right_aligned(self):
        history = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        updated = _roll_append(history, torch.tensor([[4.0]]))
        self.assertTrue(
            torch.equal(updated, torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]))
        )

    def test_ridge_recovers_multilag_mapping(self):
        signal = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 2.0, -1.0],
                [2.0, -1.0, 1.0, 3.0],
                [-1.0, 2.0, 3.0, 1.0],
                [3.0, 1.0, -2.0, 2.0],
            ],
            dtype=np.float64,
        )
        expected = np.array(
            [[0.5, -0.2], [1.0, 0.3], [-0.4, 0.8], [0.2, 0.6]],
            dtype=np.float64,
        )
        target = _apply_linear(signal, expected)
        fitted = _solve_ridge(signal, target)
        self.assertTrue(np.allclose(fitted, expected, atol=2e-5))
        self.assertTrue(
            np.allclose(_apply_linear(signal, fitted), target, atol=2e-5)
        )

    def test_stratified_derangement_preserves_partition_and_event(self):
        data = {
            "event": np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        }
        calibration = np.arange(12) < 6
        evaluation = ~calibration
        permutation = _stratified_derangement(
            data,
            calibration,
            evaluation,
            seed=7,
            extra_stratum=np.array([0, 0, 0, 1, 1, 1] * 2),
        )
        self.assertTrue(np.all(permutation != np.arange(12)))
        self.assertTrue(np.array_equal(calibration[permutation], calibration))
        self.assertTrue(
            np.array_equal(data["event"][permutation], data["event"])
        )


if __name__ == "__main__":
    unittest.main()
