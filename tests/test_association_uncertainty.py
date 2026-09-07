import unittest

import torch

from model.association_uncertainty import (
    CausalAssociationAugmentor,
    probabilistic_measurement_update,
)
from model.kalman_utils import ExtendedKalmanFilter


class AssociationUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.state = torch.tensor([
            [[0.10, -0.20, 0.02, -0.01]],
            [[-0.15, 0.05, -0.01, 0.03]],
        ])
        self.covariance = torch.eye(4).unsqueeze(0).repeat(2, 1, 1) * 1e-3
        self.measurement = self.state[..., :2] + torch.tensor(
            [[[0.01, -0.005]], [[-0.004, 0.006]]]
        )
        self.r = torch.eye(2).unsqueeze(0).repeat(2, 1, 1) * 1e-4

    def test_single_certain_hypothesis_matches_ekf(self):
        candidates = self.measurement.clone()
        result = probabilistic_measurement_update(
            self.state,
            self.covariance,
            candidates,
            torch.ones(2, 1),
            torch.zeros(2),
            self.r,
        )
        expected_state, expected_covariance, innovation, covariance = (
            ExtendedKalmanFilter(2.0).measurement_update(
                self.state, self.covariance, self.measurement, self.r
            )
        )
        torch.testing.assert_close(result.state, expected_state)
        torch.testing.assert_close(
            result.covariance, expected_covariance, atol=2e-7, rtol=2e-5
        )
        torch.testing.assert_close(result.innovation, innovation)
        torch.testing.assert_close(result.innovation_covariance, covariance)
        self.assertTrue(result.valid.all())

    def test_certain_missed_detection_preserves_prior(self):
        result = probabilistic_measurement_update(
            self.state,
            self.covariance,
            self.measurement,
            torch.zeros(2, 1),
            torch.ones(2),
            self.r,
        )
        torch.testing.assert_close(result.state, self.state)
        torch.testing.assert_close(result.covariance, self.covariance)
        self.assertFalse(result.valid.any())
        self.assertEqual(float(result.innovation.abs().max()), 0.0)

    def test_ambiguous_candidates_increase_posterior_uncertainty(self):
        candidates = torch.cat(
            (self.measurement - 0.02, self.measurement + 0.02), dim=1
        )
        ambiguous = probabilistic_measurement_update(
            self.state,
            self.covariance,
            candidates,
            torch.full((2, 2), 0.5),
            torch.zeros(2),
            self.r,
        )
        certain = probabilistic_measurement_update(
            self.state,
            self.covariance,
            candidates[:, :1],
            torch.ones(2, 1),
            torch.zeros(2),
            self.r,
        )
        self.assertTrue(
            (ambiguous.covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
             > certain.covariance.diagonal(dim1=-2, dim2=-1).sum(-1)).all()
        )
        self.assertTrue((ambiguous.entropy > certain.entropy).all())
        self.assertTrue((ambiguous.spread.diagonal(dim1=-2, dim2=-1).sum(-1) > 0).all())

    def test_augmentation_is_finite_and_respects_packed_scenes(self):
        torch.manual_seed(9)
        augmentor = CausalAssociationAugmentor(
            pd_min=0.8,
            pd_max=0.8,
            clutter_mu_min=2.0,
            clutter_mu_max=2.0,
            max_clutter=4,
            max_local_clutter=1,
        )
        result = augmentor.update(
            self.state,
            self.covariance,
            self.measurement,
            self.r,
            torch.tensor([0, 1]),
            torch.ones(2, dtype=torch.bool),
        )
        for value in result:
            if torch.is_tensor(value):
                self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(((result.detection_mass >= 0) & (result.detection_mass <= 1)).all())

    def test_no_perturbation_augmentation_matches_ekf_even_outside_gate(self):
        augmentor = CausalAssociationAugmentor(
            pd_min=1.0,
            pd_max=1.0,
            clutter_mu_min=0.0,
            clutter_mu_max=0.0,
            max_clutter=0,
            max_local_clutter=0,
            gate_threshold=1e-9,
        )
        result = augmentor.update(
            self.state,
            self.covariance,
            self.measurement,
            self.r,
            torch.tensor([0, 1]),
            torch.ones(2, dtype=torch.bool),
        )
        expected_state, expected_covariance, _, _ = ExtendedKalmanFilter(
            2.0
        ).measurement_update(
            self.state, self.covariance, self.measurement, self.r
        )
        torch.testing.assert_close(result.state, expected_state)
        torch.testing.assert_close(
            result.covariance, expected_covariance, atol=2e-7, rtol=2e-5
        )
        self.assertTrue(result.valid.all())

    def test_missed_detection_cannot_update_from_clutter_in_training_mode(self):
        torch.manual_seed(13)
        augmentor = CausalAssociationAugmentor(
            pd_min=1e-6,
            pd_max=1e-6,
            clutter_mu_min=20.0,
            clutter_mu_max=20.0,
            max_clutter=20,
            local_clutter_rate=2.0,
            max_local_clutter=2,
            clutter_requires_detection=True,
        )
        result = augmentor.update(
            self.state,
            self.covariance,
            self.measurement,
            self.r,
            torch.tensor([0, 1]),
            torch.ones(2, dtype=torch.bool),
        )
        torch.testing.assert_close(result.state, self.state)
        torch.testing.assert_close(result.covariance, self.covariance)
        self.assertFalse(result.valid.any())

    def test_exclusive_bp_prevents_double_assignment(self):
        log_weight = torch.tensor([[4.0], [4.0]])
        log_miss = torch.zeros(2)
        probability, miss = CausalAssociationAugmentor.exclusive_association_probabilities(
            log_weight,
            log_miss,
            torch.ones(2, 1, dtype=torch.bool),
            torch.zeros(2, dtype=torch.long),
            iterations=50,
        )
        self.assertLessEqual(float(probability.sum()), 1.01)
        torch.testing.assert_close(
            probability.sum(dim=-1) + miss,
            torch.ones(2),
        )

    def test_exclusive_bp_does_not_couple_packed_scenes(self):
        log_weight = torch.tensor([[4.0], [4.0]])
        probability, _ = CausalAssociationAugmentor.exclusive_association_probabilities(
            log_weight,
            torch.zeros(2),
            torch.ones(2, 1, dtype=torch.bool),
            torch.tensor([0, 1]),
            iterations=20,
        )
        self.assertTrue((probability[:, 0] > 0.9).all())


if __name__ == "__main__":
    unittest.main()
