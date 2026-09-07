import unittest

import torch

from model.causal_innovation import CausalGroupInnovationCorrector
from train.training_utils_v3 import GraphDataProcessor


class CausalInnovationCorrectorTests(unittest.TestCase):
    def _bundle(self, nodes=3, history=8):
        raw = torch.randn(nodes, history, 2)
        covariance = torch.eye(2).view(1, 1, 2, 2).repeat(
            nodes, history, 1, 1
        )
        return {
            "raw": raw,
            "peer_raw": torch.randn_like(raw),
            "whitened": raw.clone(),
            "peer_whitened": torch.randn_like(raw),
            "covariance": covariance,
            "nis": raw.square().sum(-1, keepdim=True),
            "valid": torch.ones(nodes, history, 1, dtype=torch.bool),
            "peer_valid": torch.ones(nodes, history, 1, dtype=torch.bool),
        }

    def test_leave_one_out_excludes_self_and_invalid_sources(self):
        feature = torch.tensor([[1.0], [3.0], [9.0]])
        # Self-loops plus directed cross edges. Node 2 is invalid and must not
        # influence either valid target.
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
        torch.testing.assert_close(combined_valid, torch.cat(separated_valid))

    def test_zero_init_and_alpha_zero_exactly_preserve_temporal_prior(self):
        torch.manual_seed(3)
        module = CausalGroupInnovationCorrector().eval()
        correction, eligible = module(self._bundle())
        self.assertTrue(eligible.all())
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))
        base = torch.randn(3, 1, 4)
        synthetic_nonzero = torch.randn(3, 4)
        alpha_zero = module.apply_correction(base, synthetic_nonzero, alpha=0)
        self.assertTrue(torch.equal(alpha_zero, base))

    def test_incomplete_history_cannot_modify_prior(self):
        module = CausalGroupInnovationCorrector().eval()
        bundle = self._bundle()
        bundle["valid"][1, 0] = False
        bundle["peer_valid"][2, 3] = False
        with torch.no_grad():
            module.fusion[-1].weight.fill_(0.1)
            module.fusion[-1].bias.fill_(0.1)
        correction, eligible = module(bundle)
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
        _, eligible = module(bundle)
        self.assertTrue(eligible[0])
        self.assertFalse(eligible[1])
        self.assertFalse(eligible[2])

    def test_association_context_is_finite_with_missing_frames(self):
        module = CausalGroupInnovationCorrector(
            minimum_valid_history=4,
            association_context=True,
        ).eval()
        bundle = self._bundle()
        bundle["valid"][:, :2] = False
        bundle["peer_valid"][:, :2] = False
        bundle["association_detection_mass"] = torch.rand(3, 8, 1)
        bundle["association_entropy"] = torch.rand(3, 8, 1)
        spread = torch.rand(3, 8, 2, 2)
        bundle["association_spread"] = spread @ spread.transpose(-1, -2)
        correction, eligible, activation, demand = module.predict(bundle)
        self.assertTrue(eligible.all())
        for value in (correction, activation, demand):
            self.assertTrue(torch.isfinite(value).all())

    def test_selector_is_causal_bounded_and_masks_ineligible_nodes(self):
        torch.manual_seed(7)
        module = CausalGroupInnovationCorrector().eval()
        bundle = self._bundle()
        bundle["peer_valid"][1, 0] = False
        correction, eligible, activation, demand = module.predict(bundle)
        self.assertEqual(tuple(activation.shape), (3, 1))
        self.assertTrue(((activation >= 0.0) & (activation <= 1.0)).all())
        self.assertTrue(eligible[0])
        self.assertFalse(eligible[1])
        self.assertEqual(activation[1].item(), 0.0)
        self.assertEqual(correction[1].abs().max().item(), 0.0)
        self.assertTrue(((demand >= 0.0) & (demand <= 1.0)).all())
        self.assertEqual(demand[1].item(), 0.0)

    def test_tensor_activation_mixes_each_node_independently(self):
        module = CausalGroupInnovationCorrector().eval()
        base = torch.zeros(3, 1, 4)
        correction = torch.ones(3, 4)
        alpha = torch.tensor([[0.0], [0.25], [1.0]])
        mixed = module.apply_correction(base, correction, alpha)
        torch.testing.assert_close(
            mixed[:, 0, 0], torch.tensor([0.0, 0.25, 1.0])
        )


if __name__ == "__main__":
    unittest.main()
