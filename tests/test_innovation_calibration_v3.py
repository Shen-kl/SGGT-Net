import unittest
from types import SimpleNamespace

import torch

from evaluation.evaluate_innovation_calibration_v3 import (
    InnovationAccumulator,
    node_scalar_or_default,
)


class InnovationCalibrationShapeTest(unittest.TestCase):
    def test_disabled_optional_output_defaults_to_one_false_per_node(self):
        values = torch.empty(0, dtype=torch.bool)

        result = node_scalar_or_default(
            values,
            node_count=4,
            name="causal_innovation_eligible",
            empty_value=False,
        )

        self.assertEqual(tuple(result.shape), (4,))
        self.assertEqual(result.dtype, torch.bool)
        self.assertFalse(result.any())

    def test_enabled_optional_output_is_flattened(self):
        values = torch.tensor([True, False, True]).view(3, 1, 1)

        result = node_scalar_or_default(
            values,
            node_count=3,
            name="causal_innovation_eligible",
        )

        self.assertTrue(torch.equal(result, values.reshape(-1)))

    def test_mismatched_node_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one scalar per node"):
            node_scalar_or_default(
                torch.ones(2, 1, 1),
                node_count=3,
                name="causal_innovation_eligible",
            )

    def test_temporal_model_collects_causal_own_peer_window_statistics(self):
        nodes, history = 3, 8
        model = SimpleNamespace(
            decoder=SimpleNamespace(
                GASF=SimpleNamespace(last_innovation_details={})
            )
        )
        outputs = SimpleNamespace(
            real_mask=torch.ones(nodes, 1),
            measurement_innovation=torch.ones(nodes, 1, 2),
            innovation_covariance=torch.eye(2).repeat(nodes, 1, 1),
            nis=torch.full((nodes, 1, 1), 2.0),
            association_valid=torch.ones(nodes, 1, 1, dtype=torch.bool),
            association_detection_mass=torch.ones(nodes, 1, 1),
            association_entropy=torch.zeros(nodes, 1, 1),
            association_spread=torch.zeros(nodes, 2, 2),
            causal_innovation_eligible=torch.empty(0, dtype=torch.bool),
        )
        raw = torch.ones(nodes, history, 2)
        window = SimpleNamespace(
            innovation_raw=raw,
            innovation_whitened=raw,
            innovation_peer_raw=2.0 * raw,
            innovation_peer_whitened=2.0 * raw,
            nis_history=torch.full((nodes, history, 1), 2.0),
            innovation_valid=torch.ones(
                nodes, history, 1, dtype=torch.bool
            ),
            innovation_peer_valid=torch.ones(
                nodes, history, 1, dtype=torch.bool
            ),
        )
        accumulator = InnovationAccumulator()

        accumulator.observe(
            model,
            outputs,
            event_class=torch.zeros(nodes, dtype=torch.long),
            window=window,
        )
        result = accumulator.result()["all"]

        self.assertEqual(result["causal_history_nis_calibration"]["count"], nodes)
        self.assertEqual(result["statistics"]["history_peer_nis"]["count"], nodes)
        self.assertAlmostEqual(
            result["statistics"]["history_own_peer_cosine"]["mean"],
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
