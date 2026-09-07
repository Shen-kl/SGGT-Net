import unittest
from types import SimpleNamespace

import numpy as np

from evaluation.run_bp_tracker_robustness_v3 import (
    cutoff_assignment,
    matched_state_statistics,
    ospa_components,
)
from utils.common.radar import Radar
from utils.common.kalman_filter import KalmanFilter


class BPTrackerMetricTest(unittest.TestCase):
    def test_coincident_tracks_have_finite_zero_repulsion(self):
        identity = np.eye(4)
        filter_ = KalmanFilter(
            identity, identity, identity, identity, "KF", {"R1": 1.0, "R2": 1.0}
        )
        repulsion = filter_.compute_repulsion(np.zeros((4, 2), dtype=float))
        self.assertTrue(np.isfinite(repulsion).all())
        np.testing.assert_allclose(repulsion, 0.0)

    def test_identical_sets_have_zero_ospa(self):
        states = np.array([[0.0, 1.0, 0.0, 1.0]])
        self.assertEqual(ospa_components(states, states), (0.0, 0.0, 0.0))

    def test_missing_target_is_cardinality_only(self):
        estimates = np.array([[0.0, 0.0, 0.0, 0.0]])
        truth = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ])
        total, localization, cardinality = ospa_components(estimates, truth)
        self.assertEqual(localization, 0.0)
        self.assertEqual(cardinality, 50.0)
        self.assertEqual(total, 50.0)

    def test_neural_prior_preserves_group_hypothesis_axis(self):
        radar = Radar.__new__(Radar)
        radar.group_manager = {
            "IDX": np.array([1, 2]),
            "group_partition": [["a", "b", "c"], ["d", "e"]],
        }
        self.assertEqual(radar.group_partition_hypothesis_count(0), 3)
        self.assertEqual(radar.group_partition_hypothesis_count(1), 2)

    def test_neural_prior_refreshes_every_gm_component_and_partition(self):
        radar = Radar.__new__(Radar)
        radar.group_manager = {
            "IDX": np.array([1]),
            "group_partition": [["a", "b", "c"]],
        }
        radar.tracker = SimpleNamespace(
            KalmanPredict=lambda state, covariance, q_coe: (
                state,
                covariance + q_coe * np.eye(4),
            )
        )
        components = [
            SimpleNamespace(P=np.eye(4)),
            SimpleNamespace(P=2.0 * np.eye(4)),
        ]
        track = SimpleNamespace(
            track_index=7,
            numberOfGmComponents=2,
            gmComponents=components,
            gmWeight=np.array([0.6, 0.4]),
            Q_coe=0.5,
        )
        neural_prior = np.arange(4.0).reshape(4, 1)

        representative_covariance = (
            radar.apply_neural_prior_to_all_gm_components(
                0, track, neural_prior
            )
        )

        np.testing.assert_allclose(
            representative_covariance, 1.5 * np.eye(4)
        )
        for component, expected_scale in zip(components, (1.5, 2.5)):
            self.assertEqual(component.X_group_partition.shape, (4, 3))
            self.assertEqual(component.P_group_partition.shape, (4, 4, 3))
            np.testing.assert_allclose(component.X, neural_prior)
            np.testing.assert_allclose(
                component.P, expected_scale * np.eye(4)
            )
            for partition_index in range(3):
                np.testing.assert_allclose(
                    component.X_group_partition[:, partition_index],
                    neural_prior[:, 0],
                )
                np.testing.assert_allclose(
                    component.P_group_partition[:, :, partition_index],
                    expected_scale * np.eye(4),
                )

    def test_neural_prior_rejects_inconsistent_gm_bookkeeping(self):
        radar = Radar.__new__(Radar)
        radar.group_manager = {
            "IDX": np.array([1]),
            "group_partition": [["a"]],
        }
        track = SimpleNamespace(
            track_index=9,
            numberOfGmComponents=2,
            gmComponents=[SimpleNamespace(P=np.eye(4))],
            gmWeight=np.array([1.0]),
            Q_coe=1.0,
        )

        with self.assertRaisesRegex(
                ValueError, "Gaussian-mixture bookkeeping mismatch"):
            radar.apply_neural_prior_to_all_gm_components(
                0, track, np.zeros((4, 1))
            )

    def test_cutoff_assignment_rejects_distant_tracks(self):
        estimates = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [1000.0, 0.0, 1000.0, 0.0],
        ])
        truth = np.array([[0.0, 0.0, 0.0, 0.0]])
        assignments = cutoff_assignment(estimates, truth, cutoff=100.0)
        self.assertEqual(assignments, [(0, 0, np.sqrt(2.0))])

    def test_matched_state_statistics_use_xy_and_velocity_axes(self):
        estimates = np.array([[3.0, 4.0, 0.0, 0.0]])
        truth = np.zeros((1, 4))
        covariance = np.eye(4)[None]
        metrics = matched_state_statistics(
            estimates, covariance, truth, [(0, 0, 5.0)]
        )
        self.assertEqual(metrics["position_sse"], 9.0)
        self.assertEqual(metrics["velocity_sse"], 16.0)
        self.assertEqual(metrics["nis_sum"], 25.0)
        self.assertEqual(metrics["matched_state_count"], 1)


if __name__ == "__main__":
    unittest.main()
