import unittest
from types import SimpleNamespace

import torch

from train.trainer import Trainer
from train.trainer_v3 import Trainer as TrainerV3
from utils.losses import EWTALoss


class TrainingPolicyTests(unittest.TestCase):
    def test_v3_wta_applies_position_weight(self):
        prediction = torch.zeros(1, 1, 1, 4)
        target = torch.ones(1, 1, 4)
        mask = torch.ones(1, 1)
        node_weight = torch.ones(1)
        coordinate_weight = torch.tensor([16.0, 16.0, 1.0, 1.0])
        weighted = TrainerV3._weighted_wta(
            prediction, target, mask, node_weight, 1, coordinate_weight
        )
        unweighted = TrainerV3._weighted_wta(
            prediction, target, mask, node_weight, 1
        )
        self.assertAlmostEqual(float(weighted), 17.0, places=6)
        self.assertAlmostEqual(float(unweighted), 2.0, places=6)

    def test_ewta_applies_coordinate_weights(self):
        loss = EWTALoss()
        prediction = torch.zeros(1, 1, 1, 4)
        target = torch.ones(1, 1, 4)
        mask = torch.ones(1, 1)
        weights = torch.tensor([1.0, 1.0, 0.04, 0.04])
        weighted = loss(prediction, target, mask, 1, weights)
        unweighted = loss(prediction, target, mask, 1)
        self.assertAlmostEqual(float(weighted), 1.04, places=6)
        self.assertAlmostEqual(float(unweighted), 2.0, places=6)

    def test_expert_without_counterfactual_uses_physical_composite(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            training_stage='expert',
            counterfactual_training=False,
        )
        metrics = {
            'location_rmse_prediction': 1.0,
            'velocity_rmse_prediction': 2.0,
            'location_rmse_update': 3.0,
            'velocity_rmse_update': 4.0,
            'counterfactual_expert_mse': 0.0,
        }
        self.assertEqual(trainer._validation_score(metrics), 10.0)

    def test_counterfactual_expert_uses_expert_mse(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            training_stage='expert',
            counterfactual_training=True,
        )
        metrics = {
            'location_rmse_prediction': 1.0,
            'velocity_rmse_prediction': 2.0,
            'location_rmse_update': 3.0,
            'velocity_rmse_update': 4.0,
            'counterfactual_expert_mse': 0.25,
        }
        self.assertEqual(trainer._validation_score(metrics), 0.25)

    def test_v3_position_selection_ignores_velocity(self):
        trainer = TrainerV3.__new__(TrainerV3)
        trainer.args = SimpleNamespace(checkpoint_metric="position")
        metrics = {
            "location_rmse_prediction": 3.0,
            "location_rmse_update": 2.0,
            "velocity_rmse_prediction": 100.0,
            "velocity_rmse_update": 100.0,
        }
        self.assertEqual(trainer._position_score(metrics), 5.0)
        self.assertEqual(trainer._selection_score(metrics), 5.0)

    def test_v3_association_position_selection_uses_fixed_perturbation(self):
        trainer = TrainerV3.__new__(TrainerV3)
        trainer.args = SimpleNamespace(checkpoint_metric="association_position")
        trainer.last_association_validation_metrics = {
            "location_rmse_prediction": 7.0,
            "location_rmse_update": 5.0,
        }
        clean_metrics = {
            "location_rmse_prediction": 3.0,
            "location_rmse_update": 2.0,
        }
        self.assertEqual(trainer._selection_score(clean_metrics), 12.0)

    def test_v3_graph_expert_selection_uses_counterfactual_expert_loss(self):
        trainer = TrainerV3.__new__(TrainerV3)
        trainer.args = SimpleNamespace(checkpoint_metric="expert")
        metrics = {
            "loss_interaction_expert": 0.125,
            "location_rmse_prediction": 100.0,
            "location_rmse_update": 100.0,
        }
        self.assertEqual(trainer._selection_score(metrics), 0.125)

    def test_v3_graph_expert_strong_selection_uses_strong_loss(self):
        trainer = TrainerV3.__new__(TrainerV3)
        trainer.args = SimpleNamespace(checkpoint_metric="expert_strong")
        metrics = {
            "loss_interaction_expert": 0.125,
            "loss_interaction_strong_expert": 0.25,
        }
        self.assertEqual(trainer._selection_score(metrics), 0.25)

    def test_v3_gate_selection_includes_regret_protection(self):
        trainer = TrainerV3.__new__(TrainerV3)
        trainer.args = SimpleNamespace(
            checkpoint_metric="gate", lambda_interaction_regret=0.25
        )
        metrics = {
            "loss_interaction_gate": 0.6,
            "loss_interaction_regret": 0.08,
        }
        self.assertAlmostEqual(trainer._selection_score(metrics), 0.62)


if __name__ == '__main__':
    unittest.main()
