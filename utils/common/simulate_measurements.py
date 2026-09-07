import numpy as np
from utils.common.plot_track import *
'''
功能：产生当前时刻的量测
'''
def simulate_measurements(
    radar,
    measurement_plots_each_frame,
    prob_detection=1.0,
    clutter_mean=0.0,
    surveillance_bounds=None,
    rng=None,
):
    """Generate detections plus Poisson clutter without changing the tracker.

    ``prob_detection`` and ``clutter_mean`` describe the data-generating
    process.  In robustness tests, ``radar.Pd`` and ``radar.lambda_FA`` remain
    fixed nominal parameters.
    """
    rng = np.random if rng is None else rng
    plot_track_set = []
    for measurement_index in range(measurement_plots_each_frame.shape[1]):
        if prob_detection >= float(rng.random()):
            plot_track_set.append(
                Plot_Track(
                    measurement_plots_each_frame[:, measurement_index],
                    measurement_index,
                )
            )

    clutter_count = int(rng.poisson(max(float(clutter_mean), 0.0)))
    if clutter_count:
        if surveillance_bounds is None:
            extent = float(max(radar.detection_distance_range[1], 1.0))
            x_bounds = y_bounds = (-extent, extent)
        else:
            bounds = np.asarray(surveillance_bounds, dtype=float)
            if bounds.shape == (2,):
                x_bounds = y_bounds = tuple(bounds)
            elif bounds.shape == (2, 2):
                x_bounds = tuple(bounds[0])
                y_bounds = tuple(bounds[1])
            else:
                raise ValueError(
                    "surveillance_bounds must have shape (2,) or (2, 2)"
                )
        clutter = np.vstack((
            rng.uniform(x_bounds[0], x_bounds[1], clutter_count),
            rng.uniform(y_bounds[0], y_bounds[1], clutter_count),
        ))
        offset = measurement_plots_each_frame.shape[1]
        for clutter_index in range(clutter_count):
            plot_track_set.append(
                Plot_Track(clutter[:, clutter_index], offset + clutter_index)
            )

    radar.plot_track_set = plot_track_set
    radar.plot_track_num = len(plot_track_set)

    return radar
