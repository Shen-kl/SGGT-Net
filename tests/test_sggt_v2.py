import os
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from config import CONFIG
from model.gru_gnn_residual import GRUGNNDecoderV2
from model.kalman_utils import ExtendedKalmanFilter
from model.motion_models_base import GroupTrackFullModel, SecondOrderNeuralODE_groupTrack
from model.sggt_net import SGGT_Net
from model.sggt_v2_modules import (
    JointGraphTemporalSpectralFilter,
    SpectralFilterState,
    TemporalProbabilisticSEAN,
)
from train.training_utils import GraphDataProcessor


def complete_edges(node_count):
    return torch.tensor(
        [
            [i for i in range(node_count) for j in range(node_count) if i != j],
            [j for i in range(node_count) for j in range(node_count) if i != j],
        ]
    )


def adjacency_matrix(vector, node_count):
    result = vector.new_zeros(node_count, node_count)
    row, col = torch.triu_indices(node_count, node_count, offset=1)
    result[row, col] = vector
    return result + result.T


class SGGTModulesTest(unittest.TestCase):
    def test_add_fusion_preserves_spectral_increment(self):
        motion = SecondOrderNeuralODE_groupTrack(
            solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
            n_hidden=16, n_layers=1, use_SEAN=True, use_GASF=True,
            gasf_fusion_mode="add",
        )
        motion.eval()
        base = torch.randn(5, 16)
        temporal = torch.randn(5, 16)
        interaction = torch.randn(5, 16) * 0.1
        temporal_inputs = {
            "gnn_model_output": base,
            "GASF_output_fusion": temporal,
            "GASF_confidence": torch.full((5, 1), 0.5),
        }
        expert_inputs = {
            "gnn_model_output": base,
            "GASF_output_fusion": temporal + interaction,
            "GASF_confidence": torch.full((5, 1), 0.5),
        }
        temporal_output = motion._extract_inputs(temporal_inputs)
        expert_output = motion._extract_inputs(expert_inputs)
        self.assertTrue(
            torch.allclose(expert_output - temporal_output, interaction, atol=1e-6)
        )
        self.assertTrue(torch.equal(
            temporal_inputs["gasf_confidence_effective"],
            temporal_inputs["GASF_confidence"],
        ))

    def test_interaction_context_bridge_is_zero_initialized_feature_path(self):
        motion = SecondOrderNeuralODE_groupTrack(
            solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
            n_hidden=16, n_layers=1, use_SEAN=True, use_GASF=True,
            gasf_fusion_mode="concat",
        ).eval()
        base = torch.randn(5, 16)
        gasf = torch.randn(5, 16)
        context = torch.randn(5, 3)

        without_context = motion._extract_inputs({
            "gnn_model_output": base,
            "GASF_output_fusion": gasf,
        })
        zero_initialized = motion._extract_inputs({
            "gnn_model_output": base,
            "GASF_output_fusion": gasf,
            "GASF_interaction_context": context,
        })
        torch.testing.assert_close(without_context, zero_initialized)

        with torch.no_grad():
            motion.gasf_interaction_context_embedding.weight[0, 0] = 1.0
            motion.gasf_interaction_context_embedding.weight[1, 1] = 1.0
        bridged = motion._extract_inputs({
            "gnn_model_output": base,
            "GASF_output_fusion": gasf,
            "GASF_interaction_context": context,
        })
        self.assertGreater(
            float((bridged - without_context).abs().max().detach()), 0.0
        )

    def test_physical_context_gain_is_an_isolated_acceleration_diagnostic(self):
        motion = SecondOrderNeuralODE_groupTrack(
            solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
            n_hidden=16, n_layers=1, use_SEAN=True, use_GASF=True,
            gasf_fusion_mode="concat",
        ).eval()
        nodes = 4
        state = torch.zeros(nodes, 1, 4)
        expert_context = torch.randn(nodes, 3)
        inputs = {
            "gnn_model_output": torch.randn(nodes, 16),
            "GASF_output_fusion": torch.randn(nodes, 16),
            "GASF_interaction_context_expert": expert_context,
        }
        motion.interaction_acceleration_gain_override = 0.5
        motion.interaction_acceleration_use_expert_context = True
        derivative = motion.model_update(state, inputs, static_f=None)
        torch.testing.assert_close(
            derivative[:, 0, 2:], 0.5 * expert_context[:, :2]
        )

    def test_formal_physical_gain_is_bounded_and_correctly_initialized(self):
        motion = SecondOrderNeuralODE_groupTrack(
            solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
            n_hidden=16, n_layers=1, use_SEAN=True, use_GASF=True,
            gasf_fusion_mode="concat",
            interaction_physical_residual=True,
            interaction_physical_max_gain=0.075,
            interaction_physical_init_gain=0.05,
        ).eval()
        gain = motion.interaction_physical_max_gain * torch.sigmoid(
            motion.interaction_physical_gain_raw
        )
        self.assertAlmostEqual(float(gain.detach()), 0.05, places=6)
        self.assertGreater(float(gain.detach()), 0.0)
        self.assertLess(float(gain.detach()), 0.075)

    def test_q_multiplier_does_not_create_cross_node_covariance_axis(self):
        process_noise = torch.randn(7, 1, 2, 2)
        multiplier = torch.rand(7, 1)
        scaled = process_noise * multiplier.unsqueeze(-1).unsqueeze(-1)
        self.assertEqual(scaled.shape, process_noise.shape)
        self.assertTrue(torch.allclose(scaled[3], process_noise[3] * multiplier[3]))

    def test_process_noise_counterfactual_modes(self):
        model = SGGT_Net.__new__(SGGT_Net)
        torch.nn.Module.__init__(model)
        learned = torch.tensor([[[[2.0, 0.5], [0.5, 1.0]]]])
        model.process_noise_mode = "zero"
        self.assertTrue(torch.equal(model._select_process_noise(learned), torch.zeros_like(learned)))
        model.process_noise_mode = "scaled"
        model.process_noise_scale = 1.5
        self.assertTrue(torch.allclose(model._select_process_noise(learned), 1.5 * learned))
        model.process_noise_mode = "fixed"
        model.fixed_process_noise = torch.eye(2)
        self.assertTrue(torch.equal(model._select_process_noise(learned), torch.eye(2).view(1, 1, 2, 2)))

    def test_innovation_adaptive_q_is_causal_psd_and_near_identity(self):
        motion = SecondOrderNeuralODE_groupTrack(
            solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
            n_hidden=16, n_layers=1, use_SEAN=False, use_GASF=True,
            gasf_fusion_mode="concat",
        )
        decoder = GRUGNNDecoderV2(
            motion, 2.0, 9, 16, 1, 1, 0.2, 0.0, (4, 8), 2,
            "graphconv", True, False, adaptive_q_mode="innovation",
        )
        nodes = 5
        hidden = torch.randn(nodes, 16)
        gasf = torch.randn(nodes, 16)
        history = torch.randn(nodes, 8, 2)
        multiplier, position_q, _ = decoder.adaptive_process_noise(
            hidden, gasf, history
        )
        changed_future = history.clone()
        # The adapter has no future argument: identical causal history must
        # produce identical Q regardless of measurements arriving later.
        future_measurement = torch.randn(nodes, 2)
        multiplier_again, position_q_again, _ = decoder.adaptive_process_noise(
            hidden, gasf, changed_future
        )
        self.assertTrue(torch.allclose(multiplier, multiplier_again))
        self.assertTrue(torch.allclose(position_q, position_q_again))
        self.assertTrue(torch.allclose(multiplier, torch.ones_like(multiplier), atol=1e-5))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(position_q[:, 0]).min().detach()), 0.0)
        self.assertTrue(torch.isfinite(future_measurement).all())

    def test_causal_nis_q_has_separate_well_conditioned_initial_heads(self):
        motion = SecondOrderNeuralODE_groupTrack(
            solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
            n_hidden=16, n_layers=1, use_SEAN=False, use_GASF=True,
            gasf_fusion_mode="concat",
        )
        decoder = GRUGNNDecoderV2(
            motion, 2.0, 9, 16, 1, 1, 0.2, 0.0, (4, 8), 2,
            "graphconv", True, False, adaptive_q_mode="causal_nis",
            adaptive_q_max_multiplier=16.0,
            adaptive_q_max_position_variance=16.0,
            adaptive_q_initial_multiplier=8.0,
            adaptive_q_initial_position_variance=1.0,
        )
        nodes, length = 5, 8
        hidden = torch.randn(nodes, 16)
        gasf = torch.randn(nodes, 16)
        residual = torch.randn(nodes, length, 2)
        bundle = {
            "raw": torch.randn(nodes, length, 2),
            "whitened": torch.randn(nodes, length, 2),
            "nis": torch.rand(nodes, length, 1) * 8.0,
            "valid": torch.ones(nodes, length, 1, dtype=torch.bool),
        }
        multiplier, position_q, surprise = decoder.adaptive_process_noise(
            hidden, gasf, residual, bundle
        )
        physical_position = (
            position_q[:, 0, [0, 1], [0, 1]] * float(CONFIG['S_POS']) ** 2
        )
        self.assertTrue(torch.allclose(
            multiplier, torch.full_like(multiplier, 8.0), atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            physical_position, torch.ones_like(physical_position), atol=1e-5
        ))
        self.assertTrue(torch.isfinite(surprise).all())
        (multiplier.mean() + physical_position.mean()).backward()
        acceleration_gradient = (
            decoder.Q_innovation_encoder.acceleration_head.bias.grad.abs().sum()
        )
        position_gradient = (
            decoder.Q_innovation_encoder.position_head.bias.grad.abs().sum()
        )
        self.assertGreater(float(acceleration_gradient), 1e-3)
        self.assertGreater(float(position_gradient), 1e-3)

    def test_fixed_tail_q_is_identity_below_threshold_and_monotone(self):
        motion = SecondOrderNeuralODE_groupTrack(
            solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
            n_hidden=16, n_layers=1, use_SEAN=False, use_GASF=False,
        )
        decoder = GRUGNNDecoderV2(
            motion, 2.0, 9, 16, 1, 1, 0.2, 0.0, (4, 8), 2,
            "graphconv", False, False, adaptive_q_mode="legacy",
            adaptive_q_tail_boost=4.0, adaptive_q_tail_threshold=0.8,
        )
        hidden = torch.zeros(2, 16)
        gasf = torch.zeros_like(hidden)
        residual = torch.zeros(2, 8, 2)
        residual[1].fill_(4.0)
        multiplier, position_q, surprise = decoder.adaptive_process_noise(
            hidden, gasf, residual
        )
        self.assertAlmostEqual(float(multiplier[0]), 1.0, places=6)
        self.assertGreater(float(multiplier[1]), float(multiplier[0]))
        self.assertGreater(float(surprise[1]), 0.8)
        self.assertTrue(torch.equal(position_q, torch.zeros_like(position_q)))

    def test_arbitrary_rollout_window_keeps_history_and_targets_aligned(self):
        nodes, frames = 2, 24
        state = torch.arange(nodes * frames * 4, dtype=torch.float32).view(nodes, frames, 4)
        measurement = state[..., [0, 2]] + 3.0
        edge_index = [torch.empty(2, 0, dtype=torch.long) for _ in range(frames)]
        edge_features = [torch.empty(0, 1) for _ in range(frames)]
        structure = [torch.tensor([[float(index)]]) for index in range(frames)]
        data = Data(
            x=state.clone(), y=state.clone(), measurement=measurement,
            nan_mask=torch.zeros_like(state, dtype=torch.bool),
            real_mask=torch.ones_like(state, dtype=torch.bool),
            edge_index=edge_index, edge_features=edge_features,
            tar_edge_index=edge_index, tar_edge_features=edge_features,
            struct_feat=structure,
        )
        config = {'input_graph_window_len': 8, 'output_graph_window_len': 1, 'z_dimension': 2}
        args = SimpleNamespace(decoder_residual_length=[8, 16], T=2.0)

        window = GraphDataProcessor.init_input_window(data, config, args, start_bias=7)

        self.assertTrue(torch.equal(window.x, state[:, 7:15]))
        self.assertTrue(torch.equal(window.y, state[:, 15:16]))
        self.assertEqual(window.struct_feat[0].item(), 15.0)
        self.assertEqual(window.residual.shape, (nodes, 16, 2))
        self.assertTrue(torch.equal(
            window.residual[:, -15:],
            measurement[:, :15] - state[:, :15, [0, 2]],
        ))

    def setUp(self):
        torch.manual_seed(7)

    def test_sean_is_permutation_equivariant(self):
        node_count = 5
        module = TemporalProbabilisticSEAN(16, 16).eval()
        state = torch.randn(node_count, 1, 4)
        hidden = torch.randn(node_count, 16)
        batch = torch.zeros(node_count, dtype=torch.long)
        mask = torch.ones(node_count, 1, dtype=torch.bool)
        baseline = module(batch, state, hidden, mask)

        permutation = torch.tensor([3, 0, 4, 1, 2])
        permuted = module(batch, state[permutation], hidden[permutation], mask)
        expected = adjacency_matrix(baseline.adjacency_vector, node_count)[permutation][:, permutation]
        actual = adjacency_matrix(permuted.adjacency_vector, node_count)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                permuted.structural_features,
                baseline.structural_features[permutation],
                atol=1e-5,
            )
        )

    def test_sean_selector_observes_causal_maneuver_context(self):
        node_count = 5
        module = TemporalProbabilisticSEAN(16, 16).eval()
        state = torch.randn(node_count, 1, 4)
        hidden = torch.randn(node_count, 16)
        batch = torch.zeros(node_count, dtype=torch.long)
        mask = torch.ones(node_count, 1, dtype=torch.bool)
        quiet = torch.zeros(node_count, 4)
        maneuver = quiet.clone()
        maneuver[0] = torch.tensor([1.0, 1.5, 0.8, 1.2])
        quiet_output = module(
            batch, state, hidden, mask, selector_context=quiet
        )
        maneuver_output = module(
            batch, state, hidden, mask, selector_context=maneuver
        )
        self.assertGreater(
            float((quiet_output.activation - maneuver_output.activation).abs().sum().detach()),
            0.0,
        )

    def test_joint_gasf_is_permutation_equivariant_and_differentiable(self):
        node_count = 5
        module = JointGraphTemporalSpectralFilter(16, windows=(4, 8)).eval()
        residual = torch.randn(node_count, 8, 2, requires_grad=True)
        edge_index = complete_edges(node_count)
        edge_weight = torch.rand(edge_index.size(1))
        entropy = torch.rand(node_count, 1)
        mask = torch.ones(node_count, 8, 1, dtype=torch.bool)
        baseline = module(residual, edge_index, edge_weight, entropy, mask)

        permutation = torch.tensor([3, 0, 4, 1, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(node_count)
        permuted = module(
            residual[permutation],
            inverse[edge_index],
            edge_weight,
            entropy[permutation],
            mask[permutation],
        )
        self.assertTrue(
            torch.allclose(permuted.latent, baseline.latent[permutation], atol=1e-5)
        )
        self.assertTrue(torch.allclose(baseline.scale_weights.sum(-1), torch.ones(node_count)))
        baseline.latent.sum().backward()
        self.assertGreater(residual.grad.abs().sum().item(), 0.0)

    def test_recurrent_abc_gate_interpolates_previous_and_candidate_parameters(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=8,
            windows=(8,),
            graph_mode="temporal_only",
            recurrent_filter=True,
        ).eval()
        with torch.no_grad():
            for parameter in module.parameter_update_gates[0].parameters():
                parameter.zero_()
            # Make the feed-forward candidate observably depend on the current
            # low/high energy ratio while retaining the original architecture.
            for parameter in module.frequency_filters[0].parameters():
                parameter.zero_()
            module.frequency_filters[0][0].weight[0, 0] = 2.0
            module.frequency_filters[0][2].weight[:, 0] = torch.tensor(
                [1.0, -1.0, 0.5]
            )

        nodes = 3
        empty_edges = torch.empty(2, 0, dtype=torch.long)
        empty_weights = torch.empty(0)
        valid = torch.ones(nodes, 8, 1, dtype=torch.bool)
        low_frequency = torch.ones(nodes, 8, 2)
        high_frequency = torch.ones(nodes, 8, 2)
        high_frequency[:, 1::2] = -1.0

        previous = module(
            low_frequency, empty_edges, empty_weights, valid_mask=valid
        )
        candidate = module(
            high_frequency, empty_edges, empty_weights, valid_mask=valid
        )
        recurrent = module(
            high_frequency,
            empty_edges,
            empty_weights,
            valid_mask=valid,
            recurrent_state=previous.recurrent_state,
        )

        expected = 0.5 * (
            previous.recurrent_state.filter_parameters
            + candidate.recurrent_state.filter_parameters
        )
        torch.testing.assert_close(
            recurrent.recurrent_state.filter_parameters, expected
        )
        torch.testing.assert_close(
            recurrent.parameter_update_gates,
            torch.full_like(recurrent.parameter_update_gates, 0.5),
        )
        self.assertLess(
            float(
                recurrent.recurrent_state.filter_parameters[
                    ..., :2
                ].max().detach()
            ),
            0.0,
        )

    def test_previous_abc_recursion_receives_gate_and_candidate_gradients(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=8,
            windows=(8,),
            graph_mode="temporal_only",
            recurrent_filter=True,
        ).train()
        nodes = 3
        previous_state = SpectralFilterState(
            torch.tensor([-0.5, -0.75, 0.25]).view(1, 1, 3).repeat(
                nodes, 1, 1
            ),
            torch.ones(nodes, 1, 1, dtype=torch.bool),
        )
        output = module(
            torch.randn(nodes, 8, 2),
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(0),
            valid_mask=torch.ones(nodes, 8, 1, dtype=torch.bool),
            recurrent_state=previous_state,
        )
        output.recurrent_state.filter_parameters.sum().backward()
        candidate_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.frequency_filters.parameters()
            if parameter.grad is not None
        )
        gate_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameter_update_gates.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(candidate_gradient, 0.0)
        self.assertGreater(gate_gradient, 0.0)
        self.assertEqual(
            output.recurrent_state._fields, ("filter_parameters", "valid")
        )

    def test_disabled_abc_recursion_is_exactly_state_independent(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=8,
            windows=(8,),
            graph_mode="temporal_only",
            recurrent_filter=False,
        ).eval()
        nodes = 3
        residual = torch.randn(nodes, 8, 2)
        valid = torch.ones(nodes, 8, 1, dtype=torch.bool)
        previous_state = SpectralFilterState(
            torch.randn(nodes, 1, 3),
            torch.ones(nodes, 1, 1, dtype=torch.bool),
        )
        empty_edges = torch.empty(2, 0, dtype=torch.long)
        empty_weights = torch.empty(0)
        fresh = module(
            residual, empty_edges, empty_weights, valid_mask=valid
        )
        with_state = module(
            residual,
            empty_edges,
            empty_weights,
            valid_mask=valid,
            recurrent_state=previous_state,
        )
        torch.testing.assert_close(with_state.latent, fresh.latent, rtol=0, atol=0)
        torch.testing.assert_close(
            with_state.recurrent_state.filter_parameters,
            fresh.recurrent_state.filter_parameters,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            with_state.parameter_update_gates,
            torch.ones_like(with_state.parameter_update_gates),
            rtol=0,
            atol=0,
        )

    def test_recurrent_alpha_one_exactly_recovers_feedforward_filter(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=8,
            windows=(8,),
            graph_mode="temporal_only",
            recurrent_filter=True,
        ).eval()
        nodes = 3
        valid = torch.ones(nodes, 8, 1, dtype=torch.bool)
        empty_edges = torch.empty(2, 0, dtype=torch.long)
        empty_weights = torch.empty(0)
        previous = module(
            torch.randn(nodes, 8, 2),
            empty_edges,
            empty_weights,
            valid_mask=valid,
        )
        current_residual = torch.randn(nodes, 8, 2)
        module.recurrent_filter = False
        feedforward = module(
            current_residual, empty_edges, empty_weights, valid_mask=valid
        )
        module.recurrent_filter = True
        module.parameter_update_gate_override = 1.0
        recurrent = module(
            current_residual,
            empty_edges,
            empty_weights,
            valid_mask=valid,
            recurrent_state=previous.recurrent_state,
        )
        torch.testing.assert_close(recurrent.latent, feedforward.latent, rtol=0, atol=0)
        torch.testing.assert_close(
            recurrent.recurrent_state.filter_parameters,
            feedforward.recurrent_state.filter_parameters,
            rtol=0,
            atol=0,
        )

    def test_recurrent_abc_state_resets_for_invalid_and_reborn_nodes(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=8,
            windows=(8,),
            graph_mode="temporal_only",
            recurrent_filter=True,
        ).eval()
        nodes = 2
        residual = torch.randn(nodes, 8, 2)
        empty_edges = torch.empty(2, 0, dtype=torch.long)
        empty_weights = torch.empty(0)
        valid = torch.ones(nodes, 8, 1, dtype=torch.bool)
        first = module(
            residual, empty_edges, empty_weights, valid_mask=valid
        )
        invalid = module(
            residual,
            empty_edges,
            empty_weights,
            valid_mask=valid,
            recurrent_state=first.recurrent_state,
            recurrent_valid_mask=torch.zeros(nodes, 1, dtype=torch.bool),
        )
        self.assertFalse(bool(invalid.recurrent_state.valid.any()))
        self.assertTrue(torch.equal(
            invalid.recurrent_state.filter_parameters,
            torch.zeros_like(invalid.recurrent_state.filter_parameters),
        ))

        reborn = module(
            residual,
            empty_edges,
            empty_weights,
            valid_mask=valid,
            recurrent_state=invalid.recurrent_state,
            recurrent_valid_mask=torch.ones(nodes, 1, dtype=torch.bool),
        )
        fresh = module(
            residual, empty_edges, empty_weights, valid_mask=valid
        )
        torch.testing.assert_close(
            reborn.recurrent_state.filter_parameters,
            fresh.recurrent_state.filter_parameters,
        )
        torch.testing.assert_close(
            reborn.parameter_update_gates,
            torch.ones_like(reborn.parameter_update_gates),
        )

    def test_sggt_forward_and_predict_transport_recurrent_gasf_state(self):
        nodes, history = 3, 4
        motion = GroupTrackFullModel(
            dt=0.1,
            mixtures=1,
            n_hidden=16,
            use_SEAN=False,
            use_GASF=True,
            gasf_fusion_mode="concat",
        ).eval()
        model = SGGT_Net(
            4,
            16,
            1,
            1,
            1,
            0.0,
            "graphconv",
            True,
            motion,
            4,
            16,
            1,
            1,
            0.2,
            0.0,
            (4, 8),
            2,
            "graphconv",
            True,
            False,
            0.1,
            gasf_graph_mode="temporal_only",
            gasf_recurrent_filter=True,
        ).eval()
        edges = complete_edges(nodes)
        edge_history = [edges for _ in range(history)]
        edge_features = [
            torch.rand(edges.size(1), 2) for _ in range(history)
        ]
        x = torch.randn(nodes, history, 4)
        graph = Data(
            x=x,
            y=torch.randn(nodes, 1, 4),
            real_mask=torch.ones(nodes, 1, 1, dtype=torch.bool),
            nan_mask=torch.ones(nodes, history, 1, dtype=torch.bool),
            residual=torch.randn(nodes, 8, 2),
            measurement_future=torch.randn(nodes, 1, 2),
            edge_index=edge_history,
            edge_features=edge_features,
            batch=torch.zeros(nodes, dtype=torch.long),
            P_scaler=torch.eye(4).view(1, 1, 4, 4).repeat(
                nodes, 1, 1, 1
            ),
        )

        first = model(graph)
        self.assertIsNotNone(first.gasf_recurrent_state)
        self.assertEqual(first.gasf_parameter_update_gates.shape, (nodes, 1, 2))
        second = model(
            graph, gasf_recurrent_state=first.gasf_recurrent_state.detach()
        )
        self.assertTrue(torch.all(second.gasf_parameter_update_gates < 1.0))

        predicted = model.predict(
            graph,
            gasf_recurrent_state=second.gasf_recurrent_state.detach(),
            return_recurrent_state=True,
        )
        self.assertEqual(len(predicted), 4)
        self.assertIsNotNone(predicted[-1])

    def test_embedded_sean_activation_receives_trajectory_gradient(self):
        node_count = 5
        module = JointGraphTemporalSpectralFilter(
            16, windows=(4, 8), filter_mode="abc", graph_mode="graph",
            use_sean=True,
        ).train()
        residual = torch.randn(node_count, 8, 2)
        states = torch.randn(node_count, 1, 4)
        edges = complete_edges(node_count)
        edge_weight = torch.rand(edges.size(1))
        batch = torch.zeros(node_count, dtype=torch.long)
        valid = torch.ones(node_count, 8, 1, dtype=torch.bool)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        initial_activation = {
            name: parameter.detach().clone()
            for name, parameter in module.SEAN.activation_head.named_parameters()
        }
        initial_structure = {
            name: parameter.detach().clone()
            for name, parameter in module.SEAN.struct_projection.named_parameters()
        }

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = module(
                residual, edges, edge_weight, valid_mask=valid,
                batch=batch, states=states,
            )
            # A trajectory-like feature objective; no structure labels exist.
            loss = (output.latent - torch.randn_like(output.latent)).square().mean()
            loss.backward()
            activation_grad = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.SEAN.activation_head.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(activation_grad, 0.0)
            optimizer.step()

        activation_changed = any(
            not torch.equal(parameter, initial_activation[name])
            for name, parameter in module.SEAN.activation_head.named_parameters()
        )
        structure_changed = any(
            not torch.equal(parameter, initial_structure[name])
            for name, parameter in module.SEAN.struct_projection.named_parameters()
        )
        self.assertTrue(activation_changed)
        self.assertTrue(structure_changed)

    def test_sean_activation_intervention_is_a_paired_temporal_expert_control(self):
        node_count = 5
        module = JointGraphTemporalSpectralFilter(
            16, windows=(4, 8), filter_mode="abc", graph_mode="graph",
            use_sean=True,
        ).eval()
        residual = torch.randn(node_count, 8, 2)
        states = torch.randn(node_count, 1, 4)
        edges = complete_edges(node_count)
        edge_weight = torch.rand(edges.size(1))
        batch = torch.zeros(node_count, dtype=torch.long)
        valid = torch.ones(node_count, 8, 1, dtype=torch.bool)

        module.interaction_activation_override = 0.0
        temporal = module(
            residual, edges, edge_weight, valid_mask=valid,
            batch=batch, states=states,
        )
        module.interaction_activation_override = 1.0
        expert = module(
            residual, edges, edge_weight, valid_mask=valid,
            batch=batch, states=states,
        )
        module.interaction_activation_override = None
        torch.testing.assert_close(temporal.latent, temporal.temporal_latent)
        torch.testing.assert_close(
            expert.latent, expert.temporal_latent + expert.interaction_delta
        )
        torch.testing.assert_close(
            temporal.temporal_latent, expert.temporal_latent
        )
        self.assertTrue(torch.equal(
            temporal.confidence, torch.zeros_like(temporal.confidence)
        ))
        self.assertTrue(torch.equal(
            expert.confidence, torch.ones_like(expert.confidence)
        ))

    def test_interaction_gain_override_scales_only_graph_increment(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=16,
            windows=(8,),
            signal_dim=2,
            use_sean=True,
        ).eval()
        residual = torch.randn(4, 8, 2)
        states = torch.randn(4, 1, 4)
        batch = torch.zeros(4, dtype=torch.long)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ])
        edge_weight = torch.ones(edge_index.size(1))
        valid = torch.ones(4, 8, 1, dtype=torch.bool)
        module.interaction_activation_override = 1.0
        module.interaction_gain_override = 2.0
        output = module(
            residual,
            edge_index,
            edge_weight,
            valid_mask=valid,
            batch=batch,
            states=states,
        )
        self.assertTrue(torch.allclose(
            output.latent,
            output.temporal_latent + 2.0 * output.interaction_delta,
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            output.confidence, torch.full_like(output.confidence, 2.0)
        ))

    def test_context_activation_head_receives_post_graph_causal_features(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=16,
            windows=(8,),
            signal_dim=2,
            use_sean=True,
            use_context_activation=True,
        )
        residual = torch.randn(4, 8, 2)
        residual[0, -1] *= 4.0
        states = torch.randn(4, 1, 4)
        batch = torch.zeros(4, dtype=torch.long)
        edge_index = complete_edges(4)
        edge_weight = torch.ones(edge_index.size(1))
        valid = torch.ones(4, 8, 1, dtype=torch.bool)
        output = module(
            residual,
            edge_index,
            edge_weight,
            valid_mask=valid,
            batch=batch,
            states=states,
        )
        self.assertEqual(output.confidence.shape, (4, 1))
        self.assertGreater(float(output.confidence.std().detach()), 0.0)
        output.confidence.sum().backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.context_activation_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_activation_floor_calibration_is_bounded(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=16, windows=(8,), signal_dim=2, use_sean=True
        ).eval()
        residual = torch.randn(4, 8, 2)
        states = torch.randn(4, 1, 4)
        batch = torch.zeros(4, dtype=torch.long)
        edge_index = complete_edges(4)
        valid = torch.ones(4, 8, 1, dtype=torch.bool)
        module.interaction_activation_override = 0.0
        module.interaction_activation_floor_override = 0.25
        output = module(
            residual,
            edge_index,
            torch.ones(edge_index.size(1)),
            valid_mask=valid,
            batch=batch,
            states=states,
        )
        torch.testing.assert_close(
            output.confidence, torch.full_like(output.confidence, 0.25)
        )

    def test_formal_activation_floor_preserves_exact_ablation_overrides(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=16,
            windows=(8,),
            signal_dim=2,
            use_sean=True,
            use_context_activation=True,
            interaction_activation_floor=0.75,
        ).eval()
        residual = torch.randn(4, 8, 2)
        states = torch.randn(4, 1, 4)
        batch = torch.zeros(4, dtype=torch.long)
        edge_index = complete_edges(4)
        edge_weight = torch.ones(edge_index.size(1))
        valid = torch.ones(4, 8, 1, dtype=torch.bool)

        learned = module(
            residual, edge_index, edge_weight, valid_mask=valid,
            batch=batch, states=states,
        )
        self.assertTrue(torch.all(learned.confidence >= 0.75))
        self.assertTrue(torch.all(learned.confidence <= 1.0))

        module.interaction_activation_override = 0.0
        temporal = module(
            residual, edge_index, edge_weight, valid_mask=valid,
            batch=batch, states=states,
        )
        module.interaction_activation_override = 1.0
        expert = module(
            residual, edge_index, edge_weight, valid_mask=valid,
            batch=batch, states=states,
        )
        torch.testing.assert_close(
            temporal.confidence, torch.zeros_like(temporal.confidence)
        )
        torch.testing.assert_close(
            expert.confidence, torch.ones_like(expert.confidence)
        )

    def test_sean_candidate_prior_has_bounded_residual_reweighting(self):
        module = TemporalProbabilisticSEAN(hidden_dim=8, struct_dim=8).eval()
        batch = torch.zeros(3, dtype=torch.long)
        states = torch.tensor([
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.0, 1.0, 0.0],
            [0.2, 0.0, 1.0, 0.0],
        ])
        hidden = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        output = module(
            batch,
            states,
            hidden,
            candidate_edge_index=edge_index,
            candidate_edge_weight=torch.ones(edge_index.size(1)),
        )
        self.assertTrue(torch.all(output.edge_weight >= 0.5))
        self.assertTrue(torch.all(output.edge_weight <= 1.5))

    def test_temporal_only_gasf_is_independent_of_graph(self):
        node_count = 5
        module = JointGraphTemporalSpectralFilter(
            16, windows=(4, 8), filter_mode="abc", graph_mode="temporal_only"
        ).eval()
        residual = torch.randn(node_count, 8, 2, requires_grad=True)
        entropy = torch.rand(node_count, 1)
        mask = torch.ones(node_count, 8, 1, dtype=torch.bool)
        edges = complete_edges(node_count)
        baseline = module(
            residual, edges, torch.rand(edges.size(1)), entropy, mask
        )
        no_graph = module(
            residual,
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(0),
            entropy,
            mask,
        )
        self.assertTrue(torch.allclose(baseline.latent, no_graph.latent, atol=1e-6))
        baseline.latent.sum().backward()
        self.assertGreater(residual.grad.abs().sum().item(), 0.0)

    def test_concat_auxiliary_can_supervise_actual_fusion_residual(self):
        model = GroupTrackFullModel(
            dt=0.1,
            mixtures=1,
            n_hidden=16,
            use_SEAN=False,
            use_GASF=True,
            gasf_fusion_mode="concat",
            gasf_aux_source="fusion",
        ).eval()
        model.aux_velocity_head = torch.nn.Identity()
        gnn_feature = torch.randn(4, 16)
        gasf_feature = torch.randn(4, 16)
        inputs = {
            "gnn_model_output": gnn_feature,
            "GASF_output_fusion": gasf_feature,
        }
        base = model.base_fusion(
            model.base_projections["gnn_model_output"](gnn_feature)
        )
        expected_residual = model.gasf_concat_fusion(
            torch.cat((base, gasf_feature), dim=-1)
        )
        fused = model._extract_inputs(inputs)
        self.assertTrue(torch.allclose(fused, base + expected_residual, atol=1e-6))
        self.assertTrue(torch.allclose(
            inputs["auxiliary_delta_v"], expected_residual, atol=1e-6
        ))

    def test_film_starts_as_identity(self):
        model = GroupTrackFullModel(
            dt=0.1, mixtures=1, n_hidden=16, use_SEAN=True, use_GASF=True
        ).eval()
        state = torch.randn(3, 1, 4)
        common = {
            'gnn_model_output': torch.randn(3, 16),
            'struct_feats': torch.randn(3, 16),
        }
        without_gasf, _ = model(state, dict(common), None)
        with_gasf, _ = model(
            state,
            {
                **common,
                'GASF_output_fusion': torch.randn(3, 16),
                'GASF_confidence': torch.ones(3, 1),
            },
            None,
        )
        self.assertTrue(torch.allclose(without_gasf, with_gasf, atol=1e-7))

    def test_decoder_ablation_matrix(self):
        node_count, history = 4, 8
        edges = complete_edges(node_count)
        for use_sean in (False, True):
            for use_gasf in (False, True):
                motion = GroupTrackFullModel(
                    dt=0.1,
                    mixtures=1,
                    n_hidden=16,
                    use_SEAN=use_sean,
                    use_GASF=use_gasf,
                )
                decoder = GRUGNNDecoderV2(
                    motion,
                    0.1,
                    hidden_size=16,
                    residual_length=(4, 8),
                    use_SEAN=use_sean,
                    use_GASF=use_gasf,
                )
                output = decoder((
                    torch.zeros(node_count, 4),
                    torch.randn(node_count, 16),
                    torch.randn(node_count, history, 16),
                    edges,
                    torch.rand(edges.size(1), 2),
                    torch.randn(node_count, 1, 4),
                    torch.zeros(node_count, dtype=torch.long),
                    torch.ones(node_count, history, 1, dtype=torch.bool),
                    torch.randn(node_count, history, 2),
                ))
                self.assertEqual(output.next_state.shape, (node_count, 1, 4))
                self.assertEqual(output.spectral_weights.size(-1), 2 if use_gasf else 0)
                self.assertEqual(
                    output.gasf_recurrent_state is not None, use_gasf
                )
                self.assertEqual(
                    output.gasf_parameter_update_gates.shape,
                    (node_count, 2 if use_gasf else 0),
                )
                self.assertEqual(
                    output.adjacency_vector.numel() > 0,
                    use_sean and use_gasf,
                )

    def test_ode_jacobian_matches_finite_difference_and_preserves_psd(self):
        model = GroupTrackFullModel(
            dt=0.1, mixtures=1, n_hidden=16, use_SEAN=True, use_GASF=False
        ).eval()
        state = torch.randn(2, 1, 4)
        inputs = {
            'gnn_model_output': torch.randn(2, 16),
            'struct_feats': torch.randn(2, 16),
        }
        transition, _ = model.state_transition_matrix(state, inputs)
        epsilon = 1e-4
        positive, negative = state[:1].clone(), state[:1].clone()
        positive[0, 0, 0] += epsilon
        negative[0, 0, 0] -= epsilon
        node_inputs = {key: value[:1] for key, value in inputs.items()}
        finite_difference = (
            model(positive, dict(node_inputs), None)[0]
            - model(negative, dict(node_inputs), None)[0]
        )[0, 0] / (2 * epsilon)
        self.assertTrue(torch.allclose(finite_difference, transition[0, 0, :, 0], atol=2e-3))

        control, _ = model.input_transition_matrix(state, inputs)
        covariance = torch.eye(4).view(1, 1, 4, 4).repeat(2, 1, 1, 1) * 0.1
        process_noise = torch.eye(2).view(1, 1, 2, 2).repeat(2, 1, 1, 1) * 0.01
        propagated, _ = ExtendedKalmanFilter(0.1).time_update(
            covariance, control, process_noise, F_t=transition
        )
        self.assertGreater(torch.linalg.eigvalsh(propagated).min().item(), 0.0)
        self.assertTrue(torch.allclose(propagated, propagated.transpose(-1, -2)))


if __name__ == '__main__':
    unittest.main()
