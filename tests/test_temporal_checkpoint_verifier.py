import unittest

from evaluation.verify_temporal_checkpoints_v3 import expected_value


class TemporalCheckpointVerifierTest(unittest.TestCase):
    def test_expected_value_accepts_legacy_missing_false_flag(self):
        errors = []
        expected_value({}, "causal_innovation_correction", False, False, errors, 1)
        self.assertEqual(errors, [])

    def test_expected_value_reports_incompatible_flag(self):
        errors = []
        expected_value({"decoder_use_SEAN": True}, "decoder_use_SEAN", True,
                       False, errors, 1)
        self.assertEqual(len(errors), 1)


if __name__ == "__main__":
    unittest.main()
