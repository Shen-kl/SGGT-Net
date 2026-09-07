import unittest

import torch

from model.kalman_utils import CovarianceUtils
from model.sggt_v2_modules import JointGraphTemporalSpectralFilter


def complete_edges(node_count):
    source = []
    target = []
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                source.append(i)
                target.append(j)
    return torch.tensor([source, target], dtype=torch.long)


class WhitenedInnovationTest(unittest.TestCase):
    def test_nis_matches_quadratic_form(self):
        innovation = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
        covariance = torch.tensor(
            [[[4.0, 1.0], [1.0, 2.0]], [[3.0, 0.2], [0.2, 5.0]]]
        )
        _, nis, _ = CovarianceUtils.whiten_innovation(
            innovation, covariance, max_norm=None
        )
        expected = torch.einsum(
            'ni,nij,nj->n', innovation, torch.linalg.inv(covariance), innovation
        )
        self.assertTrue(torch.allclose(nis[:, 0], expected, atol=1e-6))

    def test_whitening_is_rotation_equivariant(self):
        angle = torch.tensor(0.73)
        rotation = torch.stack(
            (
                torch.stack((torch.cos(angle), -torch.sin(angle))),
                torch.stack((torch.sin(angle), torch.cos(angle))),
            )
        )
        innovation = torch.tensor([[3.0, -2.0], [0.5, 1.2]])
        covariance = torch.tensor(
            [[[9.0, 1.5], [1.5, 4.0]], [[2.0, -0.3], [-0.3, 1.0]]]
        )
        whitened, nis, _ = CovarianceUtils.whiten_innovation(
            innovation, covariance, max_norm=None
        )
        rotated_innovation = innovation @ rotation.T
        rotated_covariance = rotation @ covariance @ rotation.T
        rotated_white, rotated_nis, _ = CovarianceUtils.whiten_innovation(
            rotated_innovation, rotated_covariance, max_norm=None
        )
        self.assertTrue(
            torch.allclose(rotated_white, whitened @ rotation.T, atol=2e-5)
        )
        self.assertTrue(torch.allclose(rotated_nis, nis, atol=2e-5))

    def test_near_singular_covariance_and_mask_are_finite(self):
        innovation = torch.tensor([[100.0, -100.0], [1.0, 2.0]])
        covariance = torch.tensor(
            [[[1e-12, 0.0], [0.0, -1e-8]], [[1.0, 0.0], [0.0, 1.0]]]
        )
        valid = torch.tensor([[True], [False]])
        whitened, nis, safe = CovarianceUtils.whiten_innovation(
            innovation, covariance, valid_mask=valid, max_norm=10.0
        )
        self.assertTrue(torch.isfinite(whitened).all())
        self.assertTrue(torch.isfinite(nis).all())
        self.assertGreater(torch.linalg.eigvalsh(safe).min().item(), 0.0)
        self.assertLessEqual(whitened[0].norm().item(), 10.0 + 1e-5)
        self.assertTrue(torch.equal(whitened[1], torch.zeros(2)))
        self.assertEqual(nis[1].item(), 0.0)

    def test_information_decomposition_reconstructs_and_isolates(self):
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=16,
            windows=(4,),
            use_sean=True,
            innovation_decomposition=True,
        )
        raw = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[3.0, 0.0], [4.0, 0.0]],
                [[8.0, 1.0], [9.0, 1.0]],
            ]
        )
        covariance = torch.eye(2).view(1, 1, 2, 2).repeat(3, 2, 1, 1)
        valid = torch.ones(3, 2, 1, dtype=torch.bool)
        # Nodes 0 and 1 communicate; node 2 is isolated.
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_weight = torch.ones(2)
        group, individual, _, _, _, _, has_neighbors = module._decompose_innovation(
            raw, covariance, valid, edge_index, edge_weight
        )
        self.assertTrue(torch.allclose(group + individual, raw, atol=1e-6))
        self.assertTrue(torch.equal(group[2], torch.zeros_like(group[2])))
        self.assertTrue(torch.equal(individual[2], raw[2]))
        self.assertFalse(has_neighbors[2].any())
        expected_common = 0.5 * (raw[0] + raw[1])
        self.assertTrue(torch.allclose(group[0], expected_common, atol=2e-5))
        self.assertTrue(torch.allclose(group[1], expected_common, atol=2e-5))

    def test_zero_initialized_path_preserves_temporal_output(self):
        torch.manual_seed(7)
        node_count = 4
        history = 8
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=16,
            windows=(4, 8),
            use_sean=True,
            innovation_decomposition=True,
        ).eval()
        residual = torch.randn(node_count, history, 2) * 0.1
        edge_index = complete_edges(node_count)
        edge_weight = torch.ones(edge_index.size(1))
        valid = torch.ones(node_count, history, 1, dtype=torch.bool)
        raw = torch.randn(node_count, history, 2)
        covariance = torch.eye(2).view(1, 1, 2, 2).repeat(
            node_count, history, 1, 1
        )
        whitened, nis, _ = CovarianceUtils.whiten_innovation(raw, covariance)
        output = module(
            residual,
            edge_index,
            edge_weight,
            valid_mask=valid,
            batch=torch.zeros(node_count, dtype=torch.long),
            states=torch.randn(node_count, 1, 4),
            innovation_bundle={
                'raw': raw,
                'whitened': whitened,
                'covariance': covariance,
                'nis': nis,
                'valid': valid,
            },
        )
        self.assertTrue(torch.equal(output.latent, output.temporal_latent))
        self.assertEqual(output.interaction_delta.abs().max().item(), 0.0)
        self.assertLessEqual(
            module.last_innovation_diagnostics['reconstruction_max_error'].item(),
            1e-6,
        )

    def test_decomposition_is_independent_of_graph_batch_packing(self):
        torch.manual_seed(11)
        history = 8
        module = JointGraphTemporalSpectralFilter(
            hidden_dim=16,
            windows=(4, 8),
            use_sean=True,
            innovation_decomposition=True,
        ).eval()
        # Make the zero-initialized projection observable for this invariance
        # test without introducing a different architecture.
        torch.nn.init.xavier_uniform_(
            module.innovation_decomposition_projection[-1].weight
        )
        node_count = 4
        residual = torch.randn(node_count, history, 2)
        raw = torch.randn(node_count, history, 2)
        covariance = torch.eye(2).view(1, 1, 2, 2).repeat(
            node_count, history, 1, 1
        )
        whitened, nis, _ = CovarianceUtils.whiten_innovation(raw, covariance)
        valid = torch.ones(node_count, history, 1, dtype=torch.bool)
        states = torch.randn(node_count, 1, 4)
        combined_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        combined = module(
            residual,
            combined_edges,
            torch.ones(4),
            valid_mask=valid,
            batch=torch.tensor([0, 0, 1, 1]),
            states=states,
            innovation_bundle={
                'raw': raw,
                'whitened': whitened,
                'covariance': covariance,
                'nis': nis,
                'valid': valid,
            },
        )
        separated = []
        local_edges = torch.tensor([[0, 1], [1, 0]])
        for start in (0, 2):
            node_slice = slice(start, start + 2)
            separated.append(module(
                residual[node_slice],
                local_edges,
                torch.ones(2),
                valid_mask=valid[node_slice],
                batch=torch.zeros(2, dtype=torch.long),
                states=states[node_slice],
                innovation_bundle={
                    'raw': raw[node_slice],
                    'whitened': whitened[node_slice],
                    'covariance': covariance[node_slice],
                    'nis': nis[node_slice],
                    'valid': valid[node_slice],
                },
            ))
        separated_latent = torch.cat([output.latent for output in separated])
        separated_delta = torch.cat(
            [output.interaction_delta for output in separated]
        )
        self.assertTrue(
            torch.allclose(combined.latent, separated_latent, atol=2e-5)
        )
        self.assertTrue(
            torch.allclose(
                combined.interaction_delta, separated_delta, atol=2e-5
            )
        )


if __name__ == '__main__':
    unittest.main()
