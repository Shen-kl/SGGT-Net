"""Run one reproducible BP-tracker robustness cell with an SGGT checkpoint."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import CONFIG
from utils.common.bp_sggt_adapter import (
    attach_bp_association_context,
    attach_missed_detection_context,
)
from utils.common.compute_posterior_spatial_distributions import (
    compute_posterior_spatial_distributions,
)
from utils.common.generateMeasurements import generateMeasurements
from utils.common.loopy_belief_propagation import loopy_belief_propagation
from utils.common.mcst_bp_adapter import MCSTBPAdapter
from utils.common.measurement_evaluation import measurement_evaluation
from utils.common.param_config import param_config
from utils.common.prediction_step import prediction_step
from utils.common.pure_PT import pure_PT
from utils.common.simulate_measurements import simulate_measurements


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth_at_frame(trajectory_info, frame_index, return_ids=False):
    truth = np.asarray(trajectory_info["labels_set"][frame_index])
    if truth.ndim != 2:
        raise ValueError("labels_set entries must be two-dimensional")
    # Legacy generated scenes store a leading label/ID row.
    truth_ids = np.arange(truth.shape[1], dtype=int)
    if truth.shape[0] == 5:
        truth_ids = truth[0].astype(int)
        truth = truth[1:]
    if truth.shape[0] != 4:
        raise ValueError(f"unexpected truth shape: {truth.shape}")
    truth = truth.T
    return (truth, truth_ids) if return_ids else truth


def confirmed_estimates(radar):
    estimates = [
        np.asarray(track.X[-1], dtype=np.float64).reshape(-1)
        for track in radar.legacy_PT_set
        if len(track.X) >= radar.confirm_trajectory_len
    ]
    return np.stack(estimates) if estimates else np.zeros((0, 4))


def confirmed_tracks(radar):
    tracks = [
        track for track in radar.legacy_PT_set
        if len(track.X) >= radar.confirm_trajectory_len
    ]
    if not tracks:
        return (
            np.zeros((0, 4)),
            np.zeros((0,), dtype=int),
            np.zeros((0, 4, 4)),
        )
    covariances = []
    for track in tracks:
        if hasattr(track, "P_scaler"):
            normalized = np.asarray(
                track.P_scaler.detach().cpu()
                if torch.is_tensor(track.P_scaler) else track.P_scaler,
                dtype=np.float64,
            ).reshape(-1, 4, 4)[0]
            scale = np.diag([
                float(CONFIG["S_POS"]), float(CONFIG["S_POS"]),
                float(CONFIG["S_VEL"]), float(CONFIG["S_VEL"]),
            ])
            model_order = scale @ normalized @ scale
            raw_order = (0, 2, 1, 3)
            covariance = model_order[np.ix_(raw_order, raw_order)]
        else:
            covariance = np.asarray(
                track.P, dtype=np.float64
            ).reshape(4, 4)
        covariances.append(covariance)
    return (
        np.stack([
            np.asarray(track.X[-1], dtype=np.float64).reshape(-1)
            for track in tracks
        ]),
        np.asarray([track.track_index for track in tracks], dtype=int),
        np.stack(covariances),
    )


def cutoff_assignment(estimates, truth, cutoff=100.0):
    """Hungarian assignment followed by an explicit localization cutoff."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    if not len(estimates) or not len(truth):
        return []
    distance = cdist(estimates[:, [0, 2]], truth[:, [0, 2]])
    row, column = linear_sum_assignment(distance)
    return [
        (int(i), int(j), float(distance[i, j]))
        for i, j in zip(row, column)
        if distance[i, j] <= cutoff
    ]


def matched_state_statistics(estimates, covariances, truth, assignments):
    """Accumulate state error and calibrated Gaussian statistics."""
    position_sse = 0.0
    velocity_sse = 0.0
    nis_sum = 0.0
    nll_sum = 0.0
    nis_95_count = 0
    for estimate_index, truth_index, _ in assignments:
        error = estimates[estimate_index] - truth[truth_index]
        position_sse += float(error[[0, 2]] @ error[[0, 2]])
        velocity_sse += float(error[[1, 3]] @ error[[1, 3]])
        covariance = 0.5 * (
            covariances[estimate_index] + covariances[estimate_index].T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
        nis = float(error @ inverse @ error)
        log_determinant = float(np.log(eigenvalues).sum())
        nll = 0.5 * (
            4.0 * np.log(2.0 * np.pi) + log_determinant + nis
        )
        nis_sum += nis
        nll_sum += nll
        # Chi-square(4) 95th percentile.
        nis_95_count += int(nis <= 9.487729036781154)
    return {
        "position_sse": position_sse,
        "velocity_sse": velocity_sse,
        "nis_sum": nis_sum,
        "nll_sum": nll_sum,
        "nis_95_count": nis_95_count,
        "matched_state_count": len(assignments),
    }


def ospa_components(
    estimates, truth, cutoff=100.0, order=1.0, state_columns=(0, 2)
):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    n, m = len(estimates), len(truth)
    if max(n, m) == 0:
        return 0.0, 0.0, 0.0
    if min(n, m):
        if state_columns is None:
            estimate_values = estimates
            truth_values = truth
        else:
            estimate_values = estimates[:, state_columns]
            truth_values = truth[:, state_columns]
        distance = np.minimum(
            cutoff, cdist(estimate_values, truth_values)
        ) ** order
        row, column = linear_sum_assignment(distance)
        localization_cost = float(distance[row, column].sum())
    else:
        localization_cost = 0.0
    denominator = max(n, m)
    localization = (localization_cost / denominator) ** (1.0 / order)
    cardinality = (
        cutoff ** order * abs(n - m) / denominator
    ) ** (1.0 / order)
    total = (
        (localization_cost + cutoff ** order * abs(n - m)) / denominator
    ) ** (1.0 / order)
    return total, localization, cardinality


def plot_tracking_run(plot_data, run, args):
    """Render a headless scene/tracking diagnostic without changing results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth_tracks = {}
    estimate_tracks = {}
    measurements = []
    for frame in plot_data:
        for target_id, state in zip(frame["truth_ids"], frame["truth_states"]):
            truth_tracks.setdefault(int(target_id), []).append((state[0], state[2]))
        for track_id, state in zip(frame["estimate_ids"], frame["estimate_states"]):
            estimate_tracks.setdefault(int(track_id), []).append((state[0], state[2]))
        measurements.extend(frame["measurements"])

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    trajectory_axis, ospa_axis, count_axis, rmse_axis = axes.flat
    if measurements:
        measurement_array = np.asarray(measurements, dtype=float)
        trajectory_axis.scatter(
            measurement_array[:, 0], measurement_array[:, 1], s=4,
            color="0.75", alpha=0.25, label="Measurements", rasterized=True,
        )
    for index, (_, points) in enumerate(sorted(truth_tracks.items())):
        points = np.asarray(points, dtype=float)
        trajectory_axis.plot(
            points[:, 0], points[:, 1], color=plt.cm.tab20(index % 20),
            linewidth=1.8, label="Truth" if index == 0 else None,
        )
        trajectory_axis.plot(points[0, 0], points[0, 1], "o", color=plt.cm.tab20(index % 20), markersize=3)
    for index, (_, points) in enumerate(sorted(estimate_tracks.items())):
        if len(points) < 2:
            continue
        points = np.asarray(points, dtype=float)
        trajectory_axis.plot(
            points[:, 0], points[:, 1], color="tab:red", alpha=0.55,
            linewidth=1.0, linestyle="--", label="P4 tracks" if index == 0 else None,
        )
    trajectory_axis.set_title("scene02 truth, measurements, and P4 tracks")
    trajectory_axis.set_xlabel("East / m")
    trajectory_axis.set_ylabel("North / m")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.grid(alpha=0.2)
    trajectory_axis.legend(loc="best")

    frames = np.arange(len(run["per_frame"]))
    ospa_axis.plot(frames, [row["ospa"] for row in run["per_frame"]], label="OSPA")
    ospa_axis.plot(
        frames, [row["ospa_localization"] for row in run["per_frame"]],
        label="Localization", alpha=0.85,
    )
    ospa_axis.plot(
        frames, [row["ospa_cardinality"] for row in run["per_frame"]],
        label="Cardinality", alpha=0.85,
    )
    ospa_axis.set_title("Framewise OSPA")
    ospa_axis.set_xlabel("Frame")
    ospa_axis.set_ylabel("Error / m")
    ospa_axis.grid(alpha=0.2)
    ospa_axis.legend()

    count_axis.plot(frames, [row["truth_count"] for row in run["per_frame"]], label="Truth")
    count_axis.plot(frames, [row["estimated_count"] for row in run["per_frame"]], label="Estimated")
    count_axis.plot(frames, [row["measurement_count"] for row in run["per_frame"]], label="Measurements", alpha=0.75)
    count_axis.set_title("Cardinality and detections")
    count_axis.set_xlabel("Frame")
    count_axis.set_ylabel("Count")
    count_axis.grid(alpha=0.2)
    count_axis.legend()

    position_rmse = []
    velocity_rmse = []
    for row in run["per_frame"]:
        count = row["matched_state_count"]
        position_rmse.append(np.sqrt(row["position_sse"] / count) if count else np.nan)
        velocity_rmse.append(np.sqrt(row["velocity_sse"] / count) if count else np.nan)
    rmse_axis.plot(frames, position_rmse, label="Position RMSE / m")
    rmse_axis.plot(frames, velocity_rmse, label="Velocity RMSE / m/s")
    rmse_axis.set_title("Matched-state framewise RMSE")
    rmse_axis.set_xlabel("Frame")
    rmse_axis.grid(alpha=0.2)
    rmse_axis.legend()

    figure.suptitle(
        f"P4 scene02 | actual Pd={args.actual_pd:g}, clutter={args.actual_clutter_mean:g} | "
        f"nominal Pd={args.nominal_pd:g}, clutter={args.nominal_clutter_mean:g} | seed={run['seed']}"
    )
    output = Path(args.plot_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    if args.truth_plot_output:
        truth_figure, truth_axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        for index, (target_id, points) in enumerate(sorted(truth_tracks.items())):
            points = np.asarray(points, dtype=float)
            color = plt.cm.tab20(index % 20)
            truth_axis.plot(points[:, 0], points[:, 1], color=color, linewidth=2, label=f"T{target_id}")
            truth_axis.plot(points[0, 0], points[0, 1], "o", color=color, markersize=4)
            truth_axis.plot(points[-1, 0], points[-1, 1], ">", color=color, markersize=5)
        truth_axis.set_title("scene02 ground-truth trajectories")
        truth_axis.set_xlabel("East / m")
        truth_axis.set_ylabel("North / m")
        truth_axis.set_aspect("equal", adjustable="datalim")
        truth_axis.grid(alpha=0.2)
        truth_axis.legend(ncol=2, fontsize=8, loc="best")
        truth_output = Path(args.truth_plot_output)
        truth_output.parent.mkdir(parents=True, exist_ok=True)
        truth_figure.savefig(truth_output, dpi=220, bbox_inches="tight")
        plt.close(truth_figure)


def run_once(args, monte_carlo_index):
    trajectory_info = torch.load(
        args.trajectory, map_location="cpu", weights_only=False
    )
    seed = int(args.seed + monte_carlo_index)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    measurement_noise_std = np.array(
        [args.range_noise_std_m, args.azimuth_noise_std_deg], dtype=float
    )
    measurements, _, zone_x, zone_y = generateMeasurements(
        trajectory_info["trajectory"],
        trajectory_info["born_dead_set"],
        trajectory_info["target_num"],
        trajectory_info["frame_max"],
        trajectory_info["delta_T"],
        trajectory_info["radar_azi_resolution"],
        trajectory_info["radar_range_resolution"],
        measurement_noise_std,
    )
    bounds = np.array([zone_x, zone_y], dtype=float)
    padding = float(args.surveillance_padding_m)
    bounds[:, 0] -= padding
    bounds[:, 1] += padding
    mcst_adapter = None
    neural_override = None
    if args.model_backend == "mcst":
        mcst_adapter = MCSTBPAdapter(
            args.checkpoint,
            repository=args.mcst_repository,
            device=args.device,
        )
        neural_override = {
            "model_name": "MCST",
            "model": mcst_adapter.model,
            "adapter": mcst_adapter,
            "config": mcst_adapter.config,
            "warmup_frames": mcst_adapter.warmup_frames,
        }
    radar = param_config(
        trajectory_info,
        measurement_noise_std,
        nominal_pd=args.nominal_pd,
        nominal_clutter_mean=args.nominal_clutter_mean,
        checkpoint=args.checkpoint if args.model_backend == "sggt" else None,
        device=args.device,
        adaptive_detection_probability=False,
        neural_measurement_covariance_scale=(
            args.neural_measurement_covariance_scale
        ),
        verbose=False,
        neural_network_override=neural_override,
    )
    neural_model = radar.neural_network["model"]
    if args.model_backend == "mcst":
        radar.tracker.Q = (
            radar.tracker.Q * float(args.neural_process_noise_scale)
        )
        radar.tracker.Q[0, 0] += float(args.neural_position_q_variance_m2)
        radar.tracker.Q[2, 2] += float(args.neural_position_q_variance_m2)
    else:
        if args.neural_process_noise_scale == 0.0:
            neural_model.process_noise_mode = "zero"
        else:
            neural_model.process_noise_mode = "scaled"
            neural_model.process_noise_scale = float(
                args.neural_process_noise_scale
            )
        if not hasattr(neural_model.decoder, "q_position_variance_override_physical"):
            raise RuntimeError(
                "loaded neural decoder does not support a fixed direct position-Q override"
            )
        neural_model.decoder.q_position_variance_override_physical = float(
            args.neural_position_q_variance_m2
        )
    finished_count = 0
    finished_history = []
    per_frame = []
    collect_plot_data = bool(args.plot_output) and monte_carlo_index == args.plot_mc_index
    plot_data = []
    truth_match_history = {}
    fragmentation_count = 0
    identity_switch_count = 0
    start = time.perf_counter()
    frame_count = int(trajectory_info["frame_max"])
    if args.max_frames is not None:
        frame_count = min(frame_count, int(args.max_frames))
    for frame_index in range(frame_count):
        radar = simulate_measurements(
            radar,
            measurements[frame_index],
            prob_detection=args.actual_pd,
            clutter_mean=args.actual_clutter_mean,
            surveillance_bounds=bounds,
            rng=rng,
        )
        detection_count = radar.plot_track_num
        radar = prediction_step(radar)
        if radar.plot_track_num > 0:
            radar, association_info, posterior = measurement_evaluation(radar)
            message_aq, message_bv = loopy_belief_propagation(
                radar, association_info
            )
            attach_bp_association_context(radar, association_info, message_aq)
            radar = compute_posterior_spatial_distributions(
                radar, association_info, message_aq, message_bv, posterior
            )
        else:
            attach_missed_detection_context(radar)
            for track in radar.legacy_PT_set:
                track.r = track.r * (1 - radar.Pd) / (1 - radar.Pd * track.r)
        radar, finished_count, finished_history = pure_PT(
            radar, finished_count, finished_history
        )
        estimates, estimate_ids, estimate_covariances = confirmed_tracks(radar)
        truth, truth_ids = truth_at_frame(
            trajectory_info, frame_index, return_ids=True
        )
        if collect_plot_data:
            plot_data.append({
                "truth_ids": truth_ids.tolist(),
                "truth_states": truth.tolist(),
                "estimate_ids": estimate_ids.tolist(),
                "estimate_states": estimates.tolist(),
                "measurements": [
                    np.asarray(measurement.X, dtype=float).reshape(-1).tolist()
                    for measurement in radar.plot_track_set
                ],
            })
        assignments = cutoff_assignment(
            estimates, truth, cutoff=args.ospa_cutoff_m
        )
        truth_to_estimate = {
            truth_index: estimate_index
            for estimate_index, truth_index, _ in assignments
        }
        for truth_index, truth_id in enumerate(truth_ids):
            history = truth_match_history.setdefault(
                int(truth_id),
                {"ever_matched": False, "matched_last": False,
                 "last_track_id": None},
            )
            estimate_index = truth_to_estimate.get(truth_index)
            if estimate_index is None:
                if history["ever_matched"]:
                    history["matched_last"] = False
                continue
            current_track_id = int(estimate_ids[estimate_index])
            if history["ever_matched"] and not history["matched_last"]:
                fragmentation_count += 1
            if (
                history["matched_last"]
                and history["last_track_id"] != current_track_id
            ):
                identity_switch_count += 1
            history.update({
                "ever_matched": True,
                "matched_last": True,
                "last_track_id": current_track_id,
            })
        matched_count = len(assignments)
        state_statistics = matched_state_statistics(
            estimates, estimate_covariances, truth, assignments
        )
        false_track_count = len(estimates) - matched_count
        missed_target_count = len(truth) - matched_count
        ospa, localization, cardinality = ospa_components(
            estimates, truth, cutoff=args.ospa_cutoff_m, order=args.ospa_order
        )
        ospa_full_state, localization_full_state, _ = ospa_components(
            estimates, truth, cutoff=args.ospa_cutoff_m,
            order=args.ospa_order, state_columns=None,
        )
        per_frame.append({
            "frame": frame_index,
            "truth_count": len(truth),
            "estimated_count": len(estimates),
            "measurement_count": detection_count,
            "matched_count": matched_count,
            "false_track_count": false_track_count,
            "missed_target_count": missed_target_count,
            **state_statistics,
            "ospa": ospa,
            "ospa_localization": localization,
            "ospa_cardinality": cardinality,
            "ospa_full_state": ospa_full_state,
            "ospa_full_state_localization": localization_full_state,
        })
    elapsed = time.perf_counter() - start
    matched_state_count = sum(
        row["matched_state_count"] for row in per_frame
    )
    position_sse = sum(row["position_sse"] for row in per_frame)
    velocity_sse = sum(row["velocity_sse"] for row in per_frame)
    nis_sum = sum(row["nis_sum"] for row in per_frame)
    nll_sum = sum(row["nll_sum"] for row in per_frame)
    nis_95_count = sum(row["nis_95_count"] for row in per_frame)
    result = {
        "monte_carlo_index": monte_carlo_index,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "milliseconds_per_frame": 1000.0 * elapsed / len(per_frame),
        "mean_ospa": float(np.mean([row["ospa"] for row in per_frame])),
        "mean_ospa_localization": float(np.mean([
            row["ospa_localization"] for row in per_frame
        ])),
        "mean_ospa_cardinality": float(np.mean([
            row["ospa_cardinality"] for row in per_frame
        ])),
        "mean_ospa_full_state": float(np.mean([
            row["ospa_full_state"] for row in per_frame
        ])),
        "rms_ospa_position": float(np.sqrt(np.mean([
            row["ospa"] ** 2 for row in per_frame
        ]))),
        "rms_ospa_full_state": float(np.sqrt(np.mean([
            row["ospa_full_state"] ** 2 for row in per_frame
        ]))),
        "estimated_count_mae": float(np.mean([
            abs(row["estimated_count"] - row["truth_count"])
            for row in per_frame
        ])),
        "mean_false_track_count": float(np.mean([
            row["false_track_count"] for row in per_frame
        ])),
        "mean_missed_target_count": float(np.mean([
            row["missed_target_count"] for row in per_frame
        ])),
        "fragmentation_count": int(fragmentation_count),
        "identity_switch_count": int(identity_switch_count),
        "matched_position_rmse": (
            float(np.sqrt(position_sse / matched_state_count))
            if matched_state_count else None
        ),
        "matched_velocity_rmse": (
            float(np.sqrt(velocity_sse / matched_state_count))
            if matched_state_count else None
        ),
        "matched_state_nis_mean": (
            float(nis_sum / matched_state_count)
            if matched_state_count else None
        ),
        "matched_state_nll_mean": (
            float(nll_sum / matched_state_count)
            if matched_state_count else None
        ),
        "matched_state_nis_95_coverage": (
            float(nis_95_count / matched_state_count)
            if matched_state_count else None
        ),
        "matched_state_count": int(matched_state_count),
        "matched_position_sse": float(position_sse),
        "matched_velocity_sse": float(velocity_sse),
        "matched_state_nis_sum": float(nis_sum),
        "matched_state_nll_sum": float(nll_sum),
        "matched_state_nis_95_count": int(nis_95_count),
        "frame_count": len(per_frame),
        "per_frame": per_frame,
    }
    if collect_plot_data:
        result["_plot_data"] = plot_data
    return result


def summarize_runs(runs, key):
    values = [run[key] for run in runs if run[key] is not None]
    if not values:
        return {"mean": None, "std": None, "valid_runs": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "valid_runs": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model_backend", choices=("sggt", "mcst"), default="sggt",
        help="Neural state predictor/update implementation used inside BP.",
    )
    parser.add_argument(
        "--mcst_repository",
        default=r"E:\PycharmProjects\Neural Network\MCST_2D_update",
        help="MCST source repository; used only with --model_backend mcst.",
    )
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--actual_pd", type=float, required=True)
    parser.add_argument("--actual_clutter_mean", type=float, required=True)
    parser.add_argument("--nominal_pd", type=float, default=0.85)
    parser.add_argument("--nominal_clutter_mean", type=float, default=20.0)
    parser.add_argument("--monte_carlo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--range_noise_std_m", type=float, default=3.0)
    parser.add_argument("--azimuth_noise_std_deg", type=float, default=0.01)
    parser.add_argument("--surveillance_padding_m", type=float, default=500.0)
    parser.add_argument("--ospa_cutoff_m", type=float, default=100.0)
    parser.add_argument("--ospa_order", type=float, default=1.0)
    parser.add_argument(
        "--neural_measurement_covariance_scale", type=float, default=1.0,
        help="Multiplier applied to Cartesian R only at the neural update.",
    )
    parser.add_argument(
        "--neural_process_noise_scale", type=float, default=1.0,
        help="Evaluation-only multiplier applied to learned process noise Q.",
    )
    parser.add_argument(
        "--neural_position_q_variance_m2", type=float, default=0.0,
        help=(
            "Fixed direct position process-noise variance per x/y axis in "
            "physical m^2."
        ),
    )
    parser.add_argument(
        "--plot_output", default=None,
        help="Optional PNG path for one run's scene and tracking diagnostics.",
    )
    parser.add_argument(
        "--truth_plot_output", default=None,
        help="Optional PNG path for a scene02 ground-truth-only plot.",
    )
    parser.add_argument(
        "--plot_mc_index", type=int, default=0,
        help="Zero-based Monte Carlo run rendered by --plot_output.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Development-only frame cap; omit for a formal full rollout.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.actual_pd <= 1.0:
        raise ValueError("actual_pd must lie in [0, 1]")
    if args.monte_carlo < 1:
        raise ValueError("monte_carlo must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive when provided")
    if args.neural_measurement_covariance_scale <= 0:
        raise ValueError("neural_measurement_covariance_scale must be positive")
    if args.neural_process_noise_scale < 0:
        raise ValueError("neural_process_noise_scale must be non-negative")
    if args.neural_position_q_variance_m2 < 0:
        raise ValueError("neural_position_q_variance_m2 must be non-negative")
    if args.plot_output and not 0 <= args.plot_mc_index < args.monte_carlo:
        raise ValueError("plot_mc_index must select an executed Monte Carlo run")
    runs = [run_once(args, index) for index in range(args.monte_carlo)]
    if args.plot_output:
        plot_data = runs[args.plot_mc_index].pop("_plot_data")
        plot_tracking_run(plot_data, runs[args.plot_mc_index], args)
    checkpoint = Path(args.checkpoint).resolve()
    trajectory = Path(args.trajectory).resolve()
    report = {
        "protocol": {
            "version": 1,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_backend": args.model_backend,
            "trajectory": str(trajectory),
            "trajectory_sha256": sha256_file(trajectory),
            "actual_pd": args.actual_pd,
            "actual_clutter_mean": args.actual_clutter_mean,
            "nominal_pd": args.nominal_pd,
            "nominal_clutter_mean": args.nominal_clutter_mean,
            "range_noise_std_m": args.range_noise_std_m,
            "azimuth_noise_std_deg": args.azimuth_noise_std_deg,
            "surveillance_padding_m": args.surveillance_padding_m,
            "adaptive_detection_probability": False,
            "matched_parameters": (
                args.actual_pd == args.nominal_pd
                and args.actual_clutter_mean == args.nominal_clutter_mean
            ),
            "monte_carlo_runs": args.monte_carlo,
            "seed_start": args.seed,
            "ospa_cutoff_m": args.ospa_cutoff_m,
            "ospa_order": args.ospa_order,
            "neural_measurement_covariance_scale": (
                args.neural_measurement_covariance_scale
            ),
            "neural_process_noise_scale": args.neural_process_noise_scale,
            "neural_position_q_variance_per_axis_m2": (
                args.neural_position_q_variance_m2
            ),
            "position_ospa_state_columns": [0, 2],
            "full_state_ospa_state_columns": [0, 1, 2, 3],
            "matched_state_covariance_source": (
                "BP/Kalman hypothesis covariance with MCST state means"
                if args.model_backend == "mcst"
                else (
                    "neural P_scaler, physically denormalized and reordered to "
                    "[x,vx,y,vy]; track.P fallback only when P_scaler is absent"
                )
            ),
            "evaluated_frames": args.max_frames or "full_trajectory",
        },
        "summary": {
            key: summarize_runs(runs, key)
            for key in (
                "mean_ospa", "mean_ospa_localization",
                "mean_ospa_cardinality", "estimated_count_mae",
                "mean_ospa_full_state", "rms_ospa_position",
                "rms_ospa_full_state",
                "mean_false_track_count", "mean_missed_target_count",
                "fragmentation_count", "identity_switch_count",
                "matched_position_rmse", "matched_velocity_rmse",
                "matched_state_nis_mean", "matched_state_nll_mean",
                "matched_state_nis_95_coverage", "matched_state_count",
                "frame_count",
                "milliseconds_per_frame",
            )
        },
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
