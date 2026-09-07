import types
import unittest

import numpy as np
import torch

from utils.common.bp_sggt_adapter import (
    association_moments,
    bp_association_marginals,
)
from utils.common.graph_input_builder import GraphInputBuilder


class BpSggtAdapterTests(unittest.TestCase):
    def test_bp_marginals_include_missed_detection_and_normalize(self):
        association = {
            "beta_message": np.array([
                [1.0, 1.0, 1.0],
                [0.2, 3.0, 0.5],
                [0.4, 0.2, 2.0],
            ])
        }
        message = np.array([[1.0, 0.5, 0.25], [1.0, 0.3, 0.7]])
        marginal = bp_association_marginals(association, message)
        np.testing.assert_allclose(marginal.sum(axis=1), 1.0)
        expected = association["beta_message"][1:] * message
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(marginal, expected)

    def test_soft_moments_do_not_hard_select_a_measurement(self):
        moments = association_moments(
            np.array([0.0, 0.0]),
            np.array([[-2.0, 0.0], [2.0, 0.0]]),
            np.array([0.2, 0.4, 0.4]),
        )
        np.testing.assert_allclose(moments["measurement"], [0.0, 0.0])
        self.assertAlmostEqual(moments["detection_mass"], 0.8)
        self.assertGreater(moments["spread"][0, 0], 0.0)
        self.assertGreater(moments["entropy"], 0.0)

    def test_graph_builder_slides_state_history_once_per_frame(self):
        builder = GraphInputBuilder(
            window_size=4,
            max_targets=1,
            group_connect_threshold=10.0,
            residual_window_size=4,
        )

        class Track:
            track_index = 1
            P_scaler = torch.zeros((1, 4, 4), dtype=torch.float32)

            def __init__(self, value):
                self.X = [np.array([[value], [0.0], [0.0], [0.0]])]
                self.associate_plot_track = [np.array([[value], [0.0]])]

        radar = types.SimpleNamespace(legacy_PT_set=[])
        for value in (1.0, 2.0, 3.0, 4.0):
            radar.legacy_PT_set = [Track(value)]
            builder.update(radar)
        np.testing.assert_allclose(builder.node_buffer[0, :, 0], [1, 2, 3, 4])

    def test_graph_builder_capacity_keeps_duration_mask_aligned(self):
        builder = GraphInputBuilder(
            window_size=2,
            max_targets=2,
            group_connect_threshold=10.0,
            residual_window_size=2,
        )

        def track(track_index):
            return types.SimpleNamespace(
                track_index=track_index,
                X=[np.array([[float(track_index)], [0.0], [0.0], [0.0]])],
                associate_plot_track=[np.array([[float(track_index)], [0.0]])],
                P_scaler=torch.zeros((1, 4, 4), dtype=torch.float32),
            )

        radar = types.SimpleNamespace(legacy_PT_set=[track(1), track(2), track(3)])
        data = builder.update(radar, duration_threshold=1)
        self.assertEqual(len(data.active_track_ids), 2)
        self.assertTrue(
            all(edge.max() < 2 for edge in data.edge_indices_list if len(edge))
        )

    def test_graph_builder_uses_strict_prior_innovation_context(self):
        builder = GraphInputBuilder(
            window_size=2,
            max_targets=1,
            group_connect_threshold=10.0,
            residual_window_size=4,
        )
        track = types.SimpleNamespace(
            track_index=1,
            X=[np.array([[10.0], [0.0], [20.0], [0.0]])],
            associate_plot_track=[np.array([[999.0], [999.0]])],
            P_scaler=torch.zeros((1, 4, 4), dtype=torch.float32),
            sggt_association_measurement=np.array([[13.0], [18.0]]),
            sggt_innovation_raw=np.array([3.0, -2.0]),
            sggt_innovation_whitened=np.array([1.5, -1.0]),
            sggt_innovation_covariance=np.eye(2) * 4.0,
            sggt_nis=3.25,
            sggt_association_detection_mass=0.7,
            sggt_association_entropy=0.6,
            sggt_association_spread=np.eye(2) * 2.0,
            sggt_association_valid=True,
        )
        builder.update(types.SimpleNamespace(legacy_PT_set=[track]))
        np.testing.assert_allclose(builder.residual_buffer[0, -1], [3.0, -2.0])
        np.testing.assert_allclose(
            builder.innovation_whitened_buffer[0, -1], [1.5, -1.0]
        )
        self.assertAlmostEqual(builder.association_detection_mass_buffer[0, -1, 0], 0.7)


if __name__ == "__main__":
    unittest.main()
