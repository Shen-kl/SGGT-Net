import unittest

import torch

from config import CONFIG
from model.causal_innovation import CausalGroupInnovationCorrector
from model.motion_models_base import GroupTrackFullModel
from train.training_utils_v3 import GraphDataProcessor


class CausalInnovationCorrectorTests(unittest.TestCase):
    def _bundle(self, nodes=3, history=8):
        raw = torch.randn(nodes, history, 2)
        return {
            "raw": raw,
            "peer_raw": torch.randn_like(raw),
            "valid": torch.ones(nodes, history, 1, dtype=torch.bool),
            "peer_valid": torch.ones(
                nodes, history, 1, dtype=torch.bool
            ),
        }

    def _base_acceleration(self, nodes=3, mixtures=1):
        return torch.randn(nodes, mixtures, 2) * 1e-3

    def test_leave_one_out_excludes_self_and_invalid_sources(self):
        feature = torch.tensor([[1.0], [3.0], [9.0]])
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1, 0, 2],
        ])
        output, has_peer = GraphDataProcessor.leave_one_out_peer_mean(
            feature, edge_index, torch.tensor([True, True, False])
        )
        torch.testing.assert_close(output[:2, 0], torch.tensor([3.0, 1.0]))
        self.assertTrue(has_peer[0])
        self.assertTrue(has_peer[1])
        self.assertFalse(has_peer[2])

    def test_peer_aggregation_is_batch_packing_invariant(self):
        feature = torch.tensor([[1.0], [3.0], [5.0], [11.0]])
        combined_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        combined, combined_valid = GraphDataProcessor.leave_one_out_peer_mean(
            feature, combined_edges, torch.ones(4, dtype=torch.bool)
        )
        local_edges = torch.tensor([[0, 1], [1, 0]])
        separated = []
        separated_valid = []
        for start in (0, 2):
            value, valid = GraphDataProcessor.leave_one_out_peer_mean(
                feature[start:start + 2], local_edges,
                torch.ones(2, dtype=torch.bool),
            )
            separated.append(value)
            separated_valid.append(valid)
        torch.testing.assert_close(combined, torch.cat(separated))
        torch.testing.assert_close(
            combined_valid, torch.cat(separated_valid)
        )

    def test_zero_init_exactly_preserves_base_acceleration(self):
        torch.manual_seed(3)
        module = CausalGroupInnovationCorrector().eval()
        correction, eligible = module(
            self._bundle(), self._base_acceleration()
        )
        self.assertTrue(eligible.all())
        self.assertEqual(tuple(correction.shape), (3, 2))
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))

    def test_raw_only_bundle_is_sufficient_and_output_is_bounded(self):
        module = CausalGroupInnovationCorrector(
            max_acceleration_mps2=5.0,
            velocity_scale=40.0,
        ).eval()
        with torch.no_grad():
            module.fusion[-1].weight.fill_(1.0)
            module.fusion[-1].bias.fill_(10.0)
        correction, eligible, activation, demand = module.predict(
            self._bundle(), self._base_acceleration()
        )
        self.assertTrue(eligible.all())
        self.assertTrue(torch.isfinite(correction).all())
        self.assertLessEqual(correction.abs().max().item(), 5.0 / 40.0)
        torch.testing.assert_close(
            activation, torch.ones_like(activation)
        )
        torch.testing.assert_close(demand, torch.zeros_like(demand))

    def test_base_acceleration_conditions_the_increment(self):
        torch.manual_seed(7)
        module = CausalGroupInnovationCorrector().eval()
        with torch.no_grad():
            module.fusion[-1].weight.normal_(0.0, 0.1)
        base_acceleration = self._base_acceleration().requires_grad_(True)
        correction, _ = module(self._bundle(), base_acceleration)
        correction.square().sum().backward()
        self.assertIsNotNone(base_acceleration.grad)
        self.assertGreater(base_acceleration.grad.abs().sum().item(), 0.0)

    def test_incomplete_history_cannot_modify_acceleration(self):
        module = CausalGroupInnovationCorrector().eval()
        bundle = self._bundle()
        bundle["valid"][1, 0] = False
        bundle["peer_valid"][2, 3] = False
        with torch.no_grad():
            module.fusion[-1].weight.fill_(0.1)
            module.fusion[-1].bias.fill_(0.1)
        correction, eligible = module(bundle, self._base_acceleration())
        self.assertTrue(eligible[0])
        self.assertFalse(eligible[1])
        self.assertFalse(eligible[2])
        self.assertGreater(correction[0].abs().sum().item(), 0.0)
        self.assertEqual(correction[1:].abs().max().item(), 0.0)

    def test_partial_history_rule_accepts_four_frames_and_requires_recent_data(self):
        module = CausalGroupInnovationCorrector(
            minimum_valid_history=4, recent_valid_window=3
        ).eval()
        bundle = self._bundle()
        bundle["valid"].zero_()
        bundle["peer_valid"].zero_()
        bundle["valid"][0, [0, 1, 2, 7]] = True
        bundle["peer_valid"][0, [0, 1, 2, 7]] = True
        bundle["valid"][1, [0, 1, 2, 3]] = True
        bundle["peer_valid"][1, [0, 1, 2, 3]] = True
        bundle["valid"][2, [5, 6, 7]] = True
        bundle["peer_valid"][2, [5, 6, 7]] = True
        _, eligible = module(bundle, self._base_acceleration())
        self.assertTrue(eligible[0])
        self.assertFalse(eligible[1])
        self.assertFalse(eligible[2])

    def test_rk4_integrates_acceleration_increment_in_position_and_velocity(self):
        dt = 2.0
        motion = GroupTrackFullModel(
            solver="rk4",
            dt=dt,
            mixtures=1,
            static_f_dim=0,
            n_hidden=8,
            n_layers=2,
            use_SEAN=False,
            use_GASF=False,
        ).eval()
        state = torch.zeros(1, 1, 4)
        physical_acceleration = torch.tensor([[2.0, -1.0]])
        normalized_acceleration = physical_acceleration / float(
            CONFIG["S_VEL"]
        )
        inputs = {
            "gnn_model_output": torch.zeros(1, 8),
            "causal_acceleration_correction": normalized_acceleration,
        }
        predicted, _ = motion(state, inputs, static_f=None)
        expected = torch.tensor([[
            0.5 * 2.0 * dt ** 2 / float(CONFIG["S_POS"]),
            0.5 * -1.0 * dt ** 2 / float(CONFIG["S_POS"]),
            2.0 * dt / float(CONFIG["S_VEL"]),
            -1.0 * dt / float(CONFIG["S_VEL"]),
        ]])
        torch.testing.assert_close(predicted[:, 0], expected, atol=1e-6, rtol=1e-6)

        transition, _ = motion.state_transition_matrix(state, inputs)
        expected_transition = torch.eye(4)
        expected_transition[0, 2] = (
            float(CONFIG["S_VEL"]) / float(CONFIG["S_POS"]) * dt
        )
        expected_transition[1, 3] = expected_transition[0, 2]
        torch.testing.assert_close(
            transition[0, 0], expected_transition, atol=1e-6, rtol=1e-6
        )

    def test_motion_entry_computes_cipic_then_uses_same_rk4_flow(self):
        dt = 2.0
        motion = GroupTrackFullModel(
            solver="rk4",
            dt=dt,
            mixtures=1,
            static_f_dim=0,
            n_hidden=8,
            n_layers=2,
            use_SEAN=False,
            use_GASF=False,
        ).eval()
        corrector = CausalGroupInnovationCorrector(
            hidden_size=8,
            max_acceleration_mps2=5.0,
            velocity_scale=float(CONFIG["S_VEL"]),
        ).eval()
        with torch.no_grad():
            corrector.fusion[-1].bias.copy_(torch.tensor([0.2, -0.1]))
        motion.causal_innovation_corrector = corrector
        motion.causal_counterfactual_enabled = True
        state = torch.zeros(1, 1, 4)
        inputs = {
            "gnn_model_output": torch.zeros(1, 8),
            "causal_innovation_bundle": self._bundle(nodes=1),
        }
        predicted, prepared = motion(state, inputs, static_f=None)
        correction = prepared["causal_acceleration_correction"]
        expected_correction = (
            torch.tanh(torch.tensor([[0.2, -0.1]]))
            * 5.0 / float(CONFIG["S_VEL"])
        )
        torch.testing.assert_close(correction, expected_correction)
        torch.testing.assert_close(
            prepared["causal_base_state"], state, atol=0.0, rtol=0.0
        )
        self.assertGreater(
            (predicted - prepared["causal_base_state"]).abs().sum().item(),
            0.0,
        )
        transition, _ = motion.state_transition_matrix(
            state, prepared
        )
        self.assertTrue(torch.isfinite(transition).all())

    def test_state_offset_diagnostic_changes_only_the_rk4_endpoint(self):
        motion = GroupTrackFullModel(
            solver="rk4",
            dt=2.0,
            mixtures=1,
            static_f_dim=0,
            n_hidden=8,
            n_layers=2,
            use_SEAN=False,
            use_GASF=False,
        ).eval()
        corrector = CausalGroupInnovationCorrector(
            hidden_size=8,
            output_mode="state_offset",
            max_position_m=50.0,
            max_velocity_mps=5.0,
            position_scale=float(CONFIG["S_POS"]),
            velocity_scale=float(CONFIG["S_VEL"]),
        ).eval()
        bias = torch.tensor([0.2, -0.1, 0.3, -0.2])
        with torch.no_grad():
            corrector.fusion[-1].bias.copy_(bias)
        motion.causal_innovation_corrector = corrector
        motion.causal_counterfactual_enabled = True
        state = torch.zeros(1, 1, 4)
        predicted, prepared = motion(
            state,
            {
                "gnn_model_output": torch.zeros(1, 8),
                "causal_innovation_bundle": self._bundle(nodes=1),
            },
            static_f=None,
        )
        expected = torch.tanh(bias).view(1, 1, 4) * torch.tensor([
            50.0 / float(CONFIG["S_POS"]),
            50.0 / float(CONFIG["S_POS"]),
            5.0 / float(CONFIG["S_VEL"]),
            5.0 / float(CONFIG["S_VEL"]),
        ]).view(1, 1, 4)
        torch.testing.assert_close(
            predicted - prepared["causal_base_state"], expected
        )
        torch.testing.assert_close(
            prepared["causal_state_correction"].unsqueeze(1), expected
        )
        self.assertNotIn("causal_acceleration_correction", prepared)
        transition, _ = motion.state_transition_matrix(state, prepared)
        self.assertTrue(torch.isfinite(transition).all())

    def test_state_channel_ablation_keeps_head_size_and_masks_outputs(self):
        modules = {
            channels: CausalGroupInnovationCorrector(
                hidden_size=8,
                output_mode="state_offset",
                state_channels=channels,
                position_scale=float(CONFIG["S_POS"]),
                velocity_scale=float(CONFIG["S_VEL"]),
            ).eval()
            for channels in ("full", "position_only", "velocity_only")
        }
        parameter_counts = {
            channels: sum(parameter.numel() for parameter in module.parameters())
            for channels, module in modules.items()
        }
        self.assertEqual(len(set(parameter_counts.values())), 1)
        bundle = self._bundle()
        base_acceleration = self._base_acceleration()
        for module in modules.values():
            with torch.no_grad():
                module.fusion[-1].bias.fill_(0.2)
        position, _ = modules["position_only"](bundle, base_acceleration)
        velocity, _ = modules["velocity_only"](bundle, base_acceleration)
        self.assertGreater(position[:, :2].abs().sum().item(), 0.0)
        self.assertEqual(position[:, 2:].abs().max().item(), 0.0)
        self.assertEqual(velocity[:, :2].abs().max().item(), 0.0)
        self.assertGreater(velocity[:, 2:].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
