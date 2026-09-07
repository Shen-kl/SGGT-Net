import unittest

from evaluation.summarize_bp_tracker_robustness_v3 import mean_std


class BPTrackerSummaryTest(unittest.TestCase):
    def test_mean_std_reports_rollout_count(self):
        result = mean_std([1.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["rollout_count"], 2)
        self.assertGreater(result["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
