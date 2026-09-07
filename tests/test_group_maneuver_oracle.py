import numpy as np
import torch
import unittest

from evaluation.evaluate_group_maneuver_oracle_v3 import (
    _apply_matrix,
    _fit_matrix,
    _leave_one_out_mean,
    _symmetric_adjacency,
)


class GroupManeuverOracleTest(unittest.TestCase):
    def test_symmetric_adjacency_uses_upper_triangle_order(self):
        # Pairs are (0,1), (0,2), (1,2).
        adjacency = _symmetric_adjacency(torch.tensor([1.0, 0.0, 1.0]), 3)
        expected = torch.tensor(
            [[False, True, False], [True, False, True], [False, True, False]]
        )
        self.assertTrue(torch.equal(adjacency, expected))

    def test_leave_one_out_mean_excludes_self_and_invalid_nodes(self):
        values = torch.tensor(
            [[1.0, 10.0], [3.0, 30.0], [float("nan"), float("nan")]]
        )
        adjacency = torch.ones(3, 3, dtype=torch.bool)
        valid = torch.tensor([True, True, False])

        mean, has_peer, counts = _leave_one_out_mean(values, adjacency, valid)

        self.assertTrue(torch.equal(counts, torch.tensor([1, 1, 0])))
        self.assertTrue(torch.equal(has_peer, torch.tensor([True, True, False])))
        self.assertTrue(torch.allclose(mean[0], values[1]))
        self.assertTrue(torch.allclose(mean[1], values[0]))
        self.assertTrue(torch.equal(mean[2], torch.zeros(2)))

    def test_explicit_matrix_fit_recovers_known_mapping(self):
        signal = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
            dtype=np.float64,
        )
        expected = np.array([[2.0, -0.5], [0.25, 3.0]], dtype=np.float64)
        target = _apply_matrix(signal, expected)

        fitted = _fit_matrix(signal, target)

        self.assertTrue(np.allclose(fitted, expected, atol=1e-7))
        self.assertTrue(np.allclose(_apply_matrix(signal, fitted), target, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
