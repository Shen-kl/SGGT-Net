import unittest

from evaluation.bootstrap_causal_activation_v3 import _bootstrap_comparison, _index


class CausalActivationBootstrapTests(unittest.TestCase):
    def test_index_prefers_batch_safe_trajectory_id(self):
        report = {"trajectories": [{
            "trajectory_id": 3,
            "trajectory_index": 99,
            "events": {"all": {"count": 1}},
        }]}
        self.assertEqual(set(_index(report)), {3})

    def test_pooled_paired_bootstrap_detects_uniform_improvement(self):
        left, right = {}, {}
        for trajectory in range(5):
            count = trajectory + 2
            left[trajectory] = {"all": {
                "count": count,
                "position_prediction_sse": 4.0 * count,
            }}
            right[trajectory] = {"all": {
                "count": count,
                "position_prediction_sse": 1.0 * count,
            }}
        result = _bootstrap_comparison(
            left, right, "all", "position_prediction", samples=200, seed=7
        )
        self.assertAlmostEqual(result["left_rmse"], 2.0)
        self.assertAlmostEqual(result["right_rmse"], 1.0)
        self.assertGreater(result["improvement_95ci"][0], 0.0)
        self.assertEqual(result["trajectory_win_fraction"], 1.0)

    def test_count_mismatch_is_rejected(self):
        left = {0: {"all": {"count": 2, "position_prediction_sse": 2.0}}}
        right = {0: {"all": {"count": 3, "position_prediction_sse": 1.0}}}
        with self.assertRaises(ValueError):
            _bootstrap_comparison(
                left, right, "all", "position_prediction", 100, 1
            )


if __name__ == "__main__":
    unittest.main()
