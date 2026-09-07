"""Evaluate causal CV or two-mode IMM priors on the packed V9 protocol."""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data_v3 import GroupTargetTrackingDatasetV3


EVENT_NAMES = {-1: "unclassified", 0: "steady", 1: "switch", 2: "strong"}
STATE_METRICS = (
    "position_prediction", "position_update",
    "velocity_prediction", "velocity_update",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transition_matrix(delta_t):
    return np.array([
        [1.0, delta_t, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, delta_t],
        [0.0, 0.0, 0.0, 1.0],
    ])


def process_covariance(delta_t, acceleration_std):
    block = np.array([
        [delta_t ** 4 / 4.0, delta_t ** 3 / 2.0],
        [delta_t ** 3 / 2.0, delta_t ** 2],
    ]) * float(acceleration_std) ** 2
    result = np.zeros((4, 4))
    result[:2, :2] = block
    result[2:, 2:] = block
    return result


def measurement_covariance(measurement, range_std, azimuth_std_deg):
    x, y = np.asarray(measurement, dtype=np.float64)
    radius = max(float(np.hypot(x, y)), 1.0)
    azimuth = math.atan2(x, y)
    sigma_a = math.radians(float(azimuth_std_deg))
    sin_a, cos_a = math.sin(azimuth), math.cos(azimuth)
    jacobian = np.array([
        [sin_a, radius * cos_a],
        [cos_a, -radius * sin_a],
    ])
    polar = np.diag([float(range_std) ** 2, sigma_a ** 2])
    covariance = jacobian @ polar @ jacobian.T
    return 0.5 * (covariance + covariance.T)


def initial_covariance(measurement_r, delta_t):
    r00, r01, r11 = (
        measurement_r[0, 0], measurement_r[0, 1], measurement_r[1, 1]
    )
    dt, dt2 = float(delta_t), float(delta_t) ** 2
    return np.array([
        [r00, r00 / dt, r01, r01 / dt],
        [r00 / dt, 2.0 * r00 / dt2, r01 / dt, 2.0 * r01 / dt2],
        [r01, r01 / dt, r11, r11 / dt],
        [r01 / dt, 2.0 * r01 / dt2, r11 / dt, 2.0 * r11 / dt2],
    ])


def ensure_psd(covariance, minimum=1e-6):
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, minimum)) @ vectors.T


def kalman_update(state, covariance, measurement, measurement_r):
    observation = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
    innovation = measurement - observation @ state
    innovation_covariance = ensure_psd(
        observation @ covariance @ observation.T + measurement_r
    )
    gain = np.linalg.solve(
        innovation_covariance, observation @ covariance
    ).T
    updated_state = state + gain @ innovation
    identity = np.eye(4)
    residual = identity - gain @ observation
    updated_covariance = (
        residual @ covariance @ residual.T + gain @ measurement_r @ gain.T
    )
    return (
        updated_state,
        ensure_psd(updated_covariance),
        innovation,
        innovation_covariance,
    )


def gaussian_log_likelihood(error, covariance):
    covariance = ensure_psd(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -float("inf")
    quadratic = float(error @ np.linalg.solve(covariance, error))
    return -0.5 * (
        len(error) * math.log(2.0 * math.pi) + logdet + quadratic
    )


def mix_gaussians(states, covariances, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / max(probabilities.sum(), 1e-300)
    mean = np.sum(probabilities[:, None] * states, axis=0)
    covariance = np.zeros_like(covariances[0])
    for probability, state, component_covariance in zip(
        probabilities, states, covariances
    ):
        difference = state - mean
        covariance += probability * (
            component_covariance + np.outer(difference, difference)
        )
    return mean, ensure_psd(covariance)


class CVFilter:
    def __init__(self, state, covariance, delta_t, acceleration_std):
        self.state = state.copy()
        self.covariance = covariance.copy()
        self.transition = transition_matrix(delta_t)
        self.process = process_covariance(delta_t, acceleration_std)

    def predict(self):
        self.state = self.transition @ self.state
        self.covariance = ensure_psd(
            self.transition @ self.covariance @ self.transition.T
            + self.process
        )
        return self.state.copy(), self.covariance.copy()

    def update(self, measurement, measurement_r):
        result = kalman_update(
            self.state, self.covariance, measurement, measurement_r
        )
        self.state, self.covariance = result[:2]
        return result


class IMMFilter:
    """Standard two-mode IMM with low/high white-acceleration CV modes."""

    def __init__(self, state, covariance, delta_t, acceleration_stds,
                 mode_transition, initial_mode_probability):
        self.transition = transition_matrix(delta_t)
        self.processes = [
            process_covariance(delta_t, value)
            for value in acceleration_stds
        ]
        self.states = np.repeat(state[None], 2, axis=0)
        self.covariances = np.repeat(covariance[None], 2, axis=0)
        self.mode_transition = np.asarray(mode_transition, dtype=np.float64)
        self.mode_probability = np.asarray(
            initial_mode_probability, dtype=np.float64
        )
        self.predicted_mode_probability = self.mode_probability.copy()
        self.prior_state = state.copy()
        self.prior_covariance = covariance.copy()

    def predict(self):
        c = self.mode_probability @ self.mode_transition
        mixing = (
            self.mode_probability[:, None] * self.mode_transition
        ) / np.maximum(c[None], 1e-300)
        mixed_states, mixed_covariances = [], []
        for destination in range(2):
            state, covariance = mix_gaussians(
                self.states,
                self.covariances,
                mixing[:, destination],
            )
            mixed_states.append(state)
            mixed_covariances.append(covariance)
        for mode in range(2):
            self.states[mode] = self.transition @ mixed_states[mode]
            self.covariances[mode] = ensure_psd(
                self.transition @ mixed_covariances[mode] @ self.transition.T
                + self.processes[mode]
            )
        self.predicted_mode_probability = c / c.sum()
        self.prior_state, self.prior_covariance = mix_gaussians(
            self.states, self.covariances, self.predicted_mode_probability
        )
        return self.prior_state.copy(), self.prior_covariance.copy()

    def update(self, measurement, measurement_r):
        likelihood_logs = []
        innovations = []
        innovation_covariances = []
        observation = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0]])
        for mode in range(2):
            result = kalman_update(
                self.states[mode], self.covariances[mode],
                measurement, measurement_r,
            )
            self.states[mode], self.covariances[mode] = result[:2]
            innovations.append(result[2])
            innovation_covariances.append(result[3])
            likelihood_logs.append(
                gaussian_log_likelihood(result[2], result[3])
            )
        log_weight = np.log(
            np.maximum(self.predicted_mode_probability, 1e-300)
        ) + np.asarray(likelihood_logs)
        log_weight -= np.max(log_weight)
        self.mode_probability = np.exp(log_weight)
        self.mode_probability /= self.mode_probability.sum()
        state, covariance = mix_gaussians(
            self.states, self.covariances, self.mode_probability
        )
        # Calibration must use the pre-measurement fused prior, not the
        # posterior after the same measurement has already been assimilated.
        innovation = measurement - observation @ self.prior_state
        innovation_covariance = ensure_psd(
            observation @ self.prior_covariance @ observation.T + measurement_r
        )
        return state, covariance, innovation, innovation_covariance


def event_at_step(data, time_step, input_length):
    start = time_step - input_length
    result = -1
    for value, name in ((0, "steady_window_mask"),
                        (1, "switch_window_mask"),
                        (2, "strong_window_mask")):
        mask = np.asarray(getattr(data, name)).reshape(-1)
        if 0 <= start < len(mask) and bool(mask[start]):
            result = value
    return result


def add_state_errors(bucket, prediction, update, truth):
    prediction_error = prediction - truth
    update_error = update - truth
    bucket["count"] += 1
    bucket["position_prediction_sse"] += float(
        prediction_error[[0, 2]] @ prediction_error[[0, 2]]
    )
    bucket["velocity_prediction_sse"] += float(
        prediction_error[[1, 3]] @ prediction_error[[1, 3]]
    )
    bucket["position_update_sse"] += float(
        update_error[[0, 2]] @ update_error[[0, 2]]
    )
    bucket["velocity_update_sse"] += float(
        update_error[[1, 3]] @ update_error[[1, 3]]
    )


def run_trajectory(data, args):
    states = np.nan_to_num(np.asarray(data.x, dtype=np.float64))
    truth = np.nan_to_num(np.asarray(data.y, dtype=np.float64))
    measurements = np.nan_to_num(np.asarray(data.measurement, dtype=np.float64))
    real_mask = np.asarray(data.real_mask[..., 0], dtype=bool)
    target_count, frame_count = real_mask.shape
    input_length = int(args.input_length)
    range_std = float(np.asarray(data.range_noise_std).reshape(-1)[0])
    azimuth_std = float(
        np.asarray(data.azimuth_noise_std_deg).reshape(-1)[0]
    )
    filters = [None] * target_count
    initial_time = input_length - 1
    for node in range(target_count):
        if real_mask[node, initial_time]:
            r = measurement_covariance(
                measurements[node, initial_time], range_std, azimuth_std
            )
            covariance = initial_covariance(r, args.delta_t)
            filters[node] = make_filter(states[node, initial_time], covariance, args)

    buckets = defaultdict(lambda: defaultdict(float))
    for time_step in range(input_length, frame_count):
        event_name = EVENT_NAMES[event_at_step(data, time_step, input_length)]
        for node in range(target_count):
            current_valid = real_mask[node, time_step]
            previous_valid = real_mask[node, time_step - 1]
            if not current_valid:
                filters[node] = None
                continue
            r = measurement_covariance(
                measurements[node, time_step], range_std, azimuth_std
            )
            if not previous_valid or filters[node] is None:
                covariance = initial_covariance(r, args.delta_t)
                filters[node] = make_filter(states[node, time_step], covariance, args)
                continue
            prediction, prediction_covariance = filters[node].predict()
            update, _, innovation, innovation_covariance = filters[node].update(
                measurements[node, time_step], r
            )
            for name in ("all", event_name):
                add_state_errors(
                    buckets[name], prediction, update, truth[node, time_step]
                )
            nis = float(
                innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
            nll = -gaussian_log_likelihood(innovation, innovation_covariance)
            for name in ("all", event_name):
                buckets[name]["measurement_nis_sum"] += nis
                buckets[name]["measurement_nll_sum"] += nll
                buckets[name]["measurement_nis_95_count"] += int(
                    nis <= 5.991464547107979
                )
                buckets[name]["prediction_covariance_trace_sum"] += float(
                    np.trace(prediction_covariance)
                )
    return buckets


def make_filter(state, covariance, args):
    if args.model == "cv":
        return CVFilter(state, covariance, args.delta_t, args.cv_acceleration_std)
    return IMMFilter(
        state, covariance, args.delta_t,
        (args.imm_low_acceleration_std, args.imm_high_acceleration_std),
        np.asarray(args.imm_transition, dtype=float).reshape(2, 2),
        args.imm_initial_probability,
    )


def merge_bucket(destination, source):
    for key, value in source.items():
        destination[key] += value


def bucket_report(bucket):
    count = int(bucket.get("count", 0))
    if not count:
        return {"count": 0}
    result = {"count": count}
    for metric in STATE_METRICS:
        sse = float(bucket[f"{metric}_sse"])
        result[f"{metric}_sse"] = sse
        result[f"{metric}_rmse"] = math.sqrt(sse / count)
    result.update({
        "location_rmse_prediction": result["position_prediction_rmse"],
        "velocity_rmse_prediction": result["velocity_prediction_rmse"],
        "location_rmse_update": result["position_update_rmse"],
        "velocity_rmse_update": result["velocity_update_rmse"],
        "measurement_nis_mean": bucket["measurement_nis_sum"] / count,
        "measurement_nll_mean": bucket["measurement_nll_sum"] / count,
        "measurement_nis_95_coverage": (
            bucket["measurement_nis_95_count"] / count
        ),
        "prediction_covariance_trace_mean": (
            bucket["prediction_covariance_trace_sum"] / count
        ),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", default="val", choices=["val", "iid_test", "ood_test"]
    )
    parser.add_argument("--model", required=True, choices=["cv", "imm"])
    parser.add_argument("--start_trajectory", type=int, default=0)
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--input_length", type=int, default=8)
    parser.add_argument("--delta_t", type=float, default=2.0)
    parser.add_argument("--cv_acceleration_std", type=float, default=2.0)
    parser.add_argument("--imm_low_acceleration_std", type=float, default=0.5)
    parser.add_argument("--imm_high_acceleration_std", type=float, default=5.0)
    parser.add_argument(
        "--imm_transition", type=float, nargs=4,
        default=[0.97, 0.03, 0.05, 0.95],
    )
    parser.add_argument(
        "--imm_initial_probability", type=float, nargs=2,
        default=[0.8, 0.2],
    )
    args = parser.parse_args()
    if args.input_length < 2 or args.delta_t <= 0:
        raise ValueError("input_length must be >=2 and delta_t must be positive")
    transition = np.asarray(args.imm_transition).reshape(2, 2)
    if not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("each IMM transition row must sum to one")
    initial_probability = np.asarray(args.imm_initial_probability)
    if not np.isclose(initial_probability.sum(), 1.0):
        raise ValueError("IMM initial probabilities must sum to one")

    dataset = GroupTargetTrackingDatasetV3(
        args.data_dir, args.split, load_structure=False,
        load_analysis_metadata=False,
    )
    start = max(0, args.start_trajectory)
    stop = len(dataset)
    if args.max_trajectories > 0:
        stop = min(stop, start + args.max_trajectories)
    if start >= stop:
        raise ValueError(f"empty trajectory range [{start}, {stop})")

    global_buckets = defaultdict(lambda: defaultdict(float))
    trajectories = []
    begin = time.perf_counter()
    for trajectory_id in range(start, stop):
        data = dataset[trajectory_id]
        buckets = run_trajectory(data, args)
        for event_name, bucket in buckets.items():
            merge_bucket(global_buckets[event_name], bucket)
        trajectories.append({
            "trajectory_id": trajectory_id,
            "trajectory_index": trajectory_id,
            "events": {
                event_name: bucket_report(bucket)
                for event_name, bucket in buckets.items()
            },
        })
    elapsed = time.perf_counter() - begin
    all_metrics = bucket_report(global_buckets["all"])
    report = {
        "protocol": {
            "version": 1,
            "dataset_root": str(Path(args.data_dir).resolve()),
            "split": args.split,
            "trajectory_start": start,
            "trajectory_stop_exclusive": stop,
            "trajectory_count": stop - start,
            "online_rollout": True,
            "known_identity_measurements": True,
            "birth_frame_scored": False,
            "real_mask_semantics": "zero means unborn or permanently dead",
            "input_length": args.input_length,
            "delta_t": args.delta_t,
        },
        "model": {
            "name": args.model,
            "cv_acceleration_std": args.cv_acceleration_std,
            "imm_acceleration_stds": [
                args.imm_low_acceleration_std,
                args.imm_high_acceleration_std,
            ],
            "imm_transition": transition.tolist(),
            "imm_initial_probability": initial_probability.tolist(),
        },
        "timing": {
            "total_seconds": elapsed,
            "milliseconds_per_valid_node_step": (
                1000.0 * elapsed / all_metrics["count"]
                if all_metrics.get("count") else None
            ),
        },
        "global": all_metrics,
        "events": {
            name: bucket_report(global_buckets[name])
            for name in EVENT_NAMES.values()
        },
        "trajectories": trajectories,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "model": report["model"],
        "global": report["global"],
        "timing": report["timing"],
    }, indent=2))


if __name__ == "__main__":
    main()
