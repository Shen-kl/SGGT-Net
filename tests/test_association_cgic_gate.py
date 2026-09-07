import unittest

from evaluation.assess_association_cgic_gate import finite_metrics, position_score


class AssociationCGICGateTest(unittest.TestCase):
    def test_finite_metrics_rejects_nonfinite_values(self):
        self.assertTrue(finite_metrics({"a": 1.0, "meta": "x"}))
        self.assertFalse(finite_metrics({"a": float("nan")}))

    def test_position_score_is_prediction_plus_update(self):
        self.assertEqual(
            position_score(
                {"location_rmse_prediction": 7.0, "location_rmse_update": 5.0}
            ),
            12.0,
        )


if __name__ == "__main__":
    unittest.main()

