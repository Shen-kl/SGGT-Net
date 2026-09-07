import unittest

import numpy as np

from evaluation.bootstrap_causal_group_innovation_oracle_v3 import (
    _bootstrap_model_difference,
    _paired_bootstrap,
)


class CausalGroupInnovationBootstrapTest(unittest.TestCase):
    def test_paired_bootstrap_detects_uniform_improvement(self):
        trajectory = np.repeat(np.arange(5), 4)
        mask = np.ones(20, dtype=bool)
        baseline = np.ones((20, 2), dtype=np.float64)
        corrected = {
            "own": baseline * 0.9,
            "geo_joint": baseline * 0.8,
            "group_joint": baseline * 0.5,
            "group_joint_shuffled_control": baseline,
        }
        result = _paired_bootstrap(
            trajectory, mask, baseline, corrected, samples=100, seed=1
        )
        group = result["models"]["group_joint"]
        self.assertGreater(group["improvement_95ci_m"][0], 0.0)
        self.assertEqual(group["trajectory_win_fraction"], 1.0)
        self.assertGreater(
            result["paired_differences"][
                "true_group_increment_over_geometry"
            ]["improvement_95ci_m"][0],
            0.0,
        )

    def test_model_difference_uses_paired_trajectories(self):
        trajectory = np.repeat(np.arange(4), 3)
        mask = np.ones(12, dtype=bool)
        left = np.ones((12, 2), dtype=np.float64)
        right = left * 0.75
        result = _bootstrap_model_difference(
            trajectory, mask, left, right, samples=100, seed=2
        )
        self.assertGreater(result["improvement_95ci_m"][0], 0.0)
        self.assertEqual(result["trajectory_win_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
