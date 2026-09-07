import types
import unittest

import torch
from torch_geometric.data import Data

from train.training_utils_v3 import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


class LifecycleMaskingTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input_graph_window_len": 2,
            "output_graph_window_len": 1,
            "z_dimension": 2,
            "MODE": 1,
            "S_POS": 10.0,
            "S_VEL": 2.0,
            "S_RES": 1.0,
            "S_EDGE_P_DIST": 1.0,
            "S_EDGE_V_DIST": 1.0,
        }
        self.args = types.SimpleNamespace(decoder_residual_length=[2])

    def _trajectory(self):
        x = torch.zeros(2, 4, 4)
        x[0, :, 0] = 100.0
        x[1, 2:, 0] = 200.0
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[0] = True
        mask[1, 2:] = True
        return Data(
            x=x,
            y=x.clone(),
            measurement=x[..., [0, 2]].clone(),
            nan_mask=mask.clone(),
            real_mask=mask.clone(),
            edge_index=[torch.empty(2, 0, dtype=torch.long) for _ in range(4)],
            edge_features=[torch.empty(0, 2) for _ in range(4)],
        )

    def test_birth_frame_is_not_scored_as_a_forecast(self):
        trajectory = self._trajectory()
        window = GraphDataProcessor.init_input_window(
            trajectory, self.config, self.args
        )
        self.assertTrue(window.current_real_mask[1, 0, 0])
        self.assertFalse(window.real_mask[1, 0, 0])
        self.assertTrue(window.real_mask[0, 0, 0])

    def test_birth_keeps_packed_initial_state_instead_of_fake_posterior(self):
        trajectory = self._trajectory()
        window = GraphDataProcessor.init_input_window(
            trajectory, self.config, self.args
        )
        fake_posterior = torch.full((2, 1, 4), -999.0)
        fake_prior = torch.zeros(2, 1, 4)
        GraphDataProcessor.update_input_window(
            window, trajectory, self.config, fake_posterior, fake_prior, 1,
            self.args,
        )
        self.assertEqual(float(trajectory.x[1, 2, 0]), 200.0)
        self.assertEqual(float(trajectory.x[0, 2, 0]), -999.0)

    def test_unborn_padding_does_not_shift_group_centroid(self):
        trajectory = self._trajectory()
        window = GraphDataProcessor.init_input_window(
            trajectory, self.config, self.args
        )
        window.batch = torch.zeros(2, dtype=torch.long)
        normalized, centroid = minmax_scaler(
            window, window.clone(), self.config, "sggt_net"
        )
        self.assertAlmostEqual(float(centroid[0, 0]), 100.0)
        self.assertAlmostEqual(float(normalized.x[0, -1, 0]), 0.0)

    def test_maneuver_target_uses_immutable_truth_not_online_posterior(self):
        trajectory = self._trajectory()
        trajectory.y[0, :, 1] = torch.tensor([0.0, 1.0, 4.0, 10.0])
        trajectory.y[0, :, 3] = torch.tensor([0.0, -1.0, -3.0, -6.0])
        window = GraphDataProcessor.init_input_window(
            trajectory, self.config, self.args
        )
        torch.testing.assert_close(
            window.maneuver_delta_v[0, 0], torch.tensor([3.0, -2.0])
        )

        fake_posterior = torch.full((2, 1, 4), -999.0)
        fake_prior = torch.zeros(2, 1, 4)
        GraphDataProcessor.update_input_window(
            window, trajectory, self.config, fake_posterior, fake_prior, 1,
            self.args,
        )
        self.assertEqual(float(trajectory.x[0, 2, 1]), -999.0)
        torch.testing.assert_close(
            window.maneuver_delta_v[0, 0], torch.tensor([6.0, -3.0])
        )


if __name__ == "__main__":
    unittest.main()
