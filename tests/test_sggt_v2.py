import os
import unittest

import torch

from model.gru_gnn_residual import GRUGNNDecoderV2
from model.kalman_utils import ExtendedKalmanFilter
from model.motion_models_base import GroupTrackFullModel
from model.sggt_v2_modules import (
    JointGraphTemporalSpectralFilter,
    TemporalProbabilisticSEAN,
)


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
                self.assertEqual(output.adjacency_vector.numel() > 0, use_sean)

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
