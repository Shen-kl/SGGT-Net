import unittest
from types import SimpleNamespace

import numpy as np

from utils.common.simulate_measurements import simulate_measurements


class SimulateMeasurementsTest(unittest.TestCase):
    def test_detection_and_clutter_do_not_modify_nominal_tracker_parameters(self):
        radar = SimpleNamespace(
            Pd=0.85,
            lambda_FA=20.0,
            detection_distance_range=[0.0, 100.0],
        )
        measurements = np.array([[10.0, 20.0], [5.0, 15.0]])
        simulate_measurements(
            radar,
            measurements,
            prob_detection=0.0,
            clutter_mean=4.0,
            surveillance_bounds=np.array([[-50.0, 50.0], [-25.0, 25.0]]),
            rng=np.random.default_rng(7),
        )
        self.assertEqual(radar.Pd, 0.85)
        self.assertEqual(radar.lambda_FA, 20.0)
        self.assertEqual(radar.plot_track_num, len(radar.plot_track_set))
        self.assertGreater(radar.plot_track_num, 0)
        for plot in radar.plot_track_set:
            self.assertTrue(-50.0 <= plot.X[0, 0] <= 50.0)
            self.assertTrue(-25.0 <= plot.X[1, 0] <= 25.0)

    def test_one_detection_no_clutter_preserves_measurement(self):
        radar = SimpleNamespace(
            Pd=0.85,
            lambda_FA=20.0,
            detection_distance_range=[0.0, 100.0],
        )
        measurement = np.array([[10.0], [5.0]])
        simulate_measurements(
            radar,
            measurement,
            prob_detection=1.0,
            clutter_mean=0.0,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(radar.plot_track_num, 1)
        np.testing.assert_allclose(radar.plot_track_set[0].X[:, 0], [10.0, 5.0])


if __name__ == "__main__":
    unittest.main()
