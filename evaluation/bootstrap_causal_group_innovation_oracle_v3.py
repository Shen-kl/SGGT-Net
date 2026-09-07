"""Trajectory-level paired bootstrap for the causal innovation oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_causal_group_innovation_oracle_v3 import (
    _apply_linear,
    _stratified_derangement,
)


EVENT_VALUES = {"all": None, "strong": 2}
MODEL_NAMES = ("own", "geo_joint", "group_joint", "group_joint_shuffled_control")


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _paired_bootstrap(
    trajectory: np.ndarray,
    event_mask: np.ndarray,
    baseline_error: np.ndarray,
    corrected_errors: dict[str, np.ndarray],
    samples: int,
    seed: int,
) -> dict:
    selected_trajectories = np.unique(trajectory[event_mask])
    baseline_sse = []
    counts = []
    corrected_sse = {name: [] for name in corrected_errors}
    per_trajectory_gain = {name: [] for name in corrected_errors}
    for trajectory_id in selected_trajectories:
        mask = event_mask & (trajectory == trajectory_id)
        count = int(mask.sum())
        base = float(np.square(baseline_error[mask]).sum())
        baseline_sse.append(base)
        counts.append(count)
        base_rmse = np.sqrt(base / count)
        for name, error in corrected_errors.items():
            value = float(np.square(error[mask]).sum())
            corrected_sse[name].append(value)
            per_trajectory_gain[name].append(base_rmse - np.sqrt(value / count))

    baseline_sse = np.asarray(baseline_sse, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    corrected_sse = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in corrected_sse.items()
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, selected_trajectories.size,
        size=(int(samples), selected_trajectories.size),
    )
    bootstrap_counts = counts[indices].sum(axis=1)
    baseline_bootstrap = np.sqrt(
        baseline_sse[indices].sum(axis=1) / bootstrap_counts
    )
    total_count = float(counts.sum())
    baseline_point = float(np.sqrt(baseline_sse.sum() / total_count))
    result = {
        "trajectory_count": int(selected_trajectories.size),
        "point_count": int(total_count),
        "baseline_position_rmse_m": baseline_point,
        "models": {},
        "paired_differences": {},
    }
    bootstrap_rmse = {}
    for name, sse in corrected_sse.items():
        point = float(np.sqrt(sse.sum() / total_count))
        values = np.sqrt(sse[indices].sum(axis=1) / bootstrap_counts)
        bootstrap_rmse[name] = values
        improvement = baseline_bootstrap - values
        trajectory_gain = np.asarray(per_trajectory_gain[name])
        result["models"][name] = {
            "corrected_position_rmse_m": point,
            "improvement_m": baseline_point - point,
            "improvement_95ci_m": _percentile_interval(improvement),
            "probability_improvement_positive": float((improvement > 0).mean()),
            "trajectory_win_fraction": float((trajectory_gain > 0).mean()),
            "trajectory_gain_median_m": float(np.median(trajectory_gain)),
        }
    for name, left, right in (
        ("group_increment_over_own", "own", "group_joint"),
        ("true_group_increment_over_geometry", "geo_joint", "group_joint"),
        (
            "aligned_increment_over_shuffled",
            "group_joint_shuffled_control",
            "group_joint",
        ),
    ):
        point = (
            result["models"][left]["corrected_position_rmse_m"]
            - result["models"][right]["corrected_position_rmse_m"]
        )
        bootstrap_difference = bootstrap_rmse[left] - bootstrap_rmse[right]
        result["paired_differences"][name] = {
            "improvement_m": point,
            "improvement_95ci_m": _percentile_interval(bootstrap_difference),
            "probability_positive": float((bootstrap_difference > 0).mean()),
        }
    return result


def _bootstrap_model_difference(
    trajectory: np.ndarray,
    event_mask: np.ndarray,
    left_error: np.ndarray,
    right_error: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    """Estimate positive RMSE improvement of right model over left model."""

    trajectories = np.unique(trajectory[event_mask])
    counts, left_sse, right_sse = [], [], []
    trajectory_differences = []
    for trajectory_id in trajectories:
        mask = event_mask & (trajectory == trajectory_id)
        count = int(mask.sum())
        left = float(np.square(left_error[mask]).sum())
        right = float(np.square(right_error[mask]).sum())
        counts.append(count)
        left_sse.append(left)
        right_sse.append(right)
        trajectory_differences.append(
            np.sqrt(left / count) - np.sqrt(right / count)
        )
    counts = np.asarray(counts, dtype=np.float64)
    left_sse = np.asarray(left_sse, dtype=np.float64)
    right_sse = np.asarray(right_sse, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, trajectories.size, size=(int(samples), trajectories.size)
    )
    bootstrap_counts = counts[indices].sum(axis=1)
    differences = (
        np.sqrt(left_sse[indices].sum(axis=1) / bootstrap_counts)
        - np.sqrt(right_sse[indices].sum(axis=1) / bootstrap_counts)
    )
    point = (
        np.sqrt(left_sse.sum() / counts.sum())
        - np.sqrt(right_sse.sum() / counts.sum())
    )
    trajectory_differences = np.asarray(trajectory_differences)
    return {
        "improvement_m": float(point),
        "improvement_95ci_m": _percentile_interval(differences),
        "probability_positive": float((differences > 0).mean()),
        "trajectory_win_fraction": float((trajectory_differences > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle_report", required=True)
    parser.add_argument("--tensor_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    report_path = Path(args.oracle_report)
    oracle = json.loads(report_path.read_text(encoding="utf-8"))
    cache = torch.load(args.tensor_cache, map_location="cpu", weights_only=False)
    if cache["checkpoint_sha256"] != oracle["protocol"]["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs between report and tensor cache")
    tensors = cache["tensors"]
    data = {name: value.numpy() for name, value in tensors.items()}
    boundary = int(
        oracle["analysis"]["split"]["calibration_trajectory_stop_exclusive"]
    )
    calibration_partition = data["trajectory"] < boundary
    evaluation_partition = data["trajectory"] >= boundary
    baseline_error = data["prediction_error"][:, [0, 2]].astype(np.float64)

    output = {
        "protocol": {
            "version": 1,
            "oracle_report": str(report_path.resolve()),
            "tensor_cache": str(Path(args.tensor_cache).resolve()),
            "checkpoint_sha256": cache["checkpoint_sha256"],
            "bootstrap_unit": "trajectory",
            "paired_resampling": True,
            "bootstrap_samples": int(args.samples),
            "seed": int(args.seed),
            "confidence_interval": "percentile 95%",
        },
        "results": {},
        "cross_history_comparisons": {},
    }
    for representation, own_name in (
        ("raw", "own_raw_history"),
        ("whitened", "own_white_history"),
    ):
        representation_result = {}
        group_errors_by_history = {}
        for history_text, length_report in oracle["analysis"][
            "history_lengths"
        ].items():
            history_length = int(history_text)
            history_valid = data["history_valid"][
                :, -history_length:, 0
            ].all(axis=1)
            permutation = _stratified_derangement(
                data,
                calibration_partition,
                evaluation_partition,
                seed=20260812 + history_length,
                extra_stratum=history_valid.astype(np.int64),
            )
            own = data[own_name][:, -history_length:].reshape(
                data[own_name].shape[0], -1
            ).astype(np.float64)
            peer_name = (
                "group_raw_history"
                if representation == "raw"
                else "group_white_history"
            )
            group_peer = data[peer_name][:, -history_length:].reshape(
                data[peer_name].shape[0], -1
            ).astype(np.float64)
            geo_name = (
                "geo_raw_history"
                if representation == "raw"
                else "geo_white_history"
            )
            geo_peer = data[geo_name][:, -history_length:].reshape(
                data[geo_name].shape[0], -1
            ).astype(np.float64)
            features = {
                "own": own,
                "geo_joint": np.concatenate((own, geo_peer), axis=1),
                "group_joint": np.concatenate((own, group_peer), axis=1),
                "group_joint_shuffled_control": np.concatenate(
                    (own, group_peer[permutation]), axis=1
                ),
            }
            model_report = length_report["representations"][representation]
            corrected_errors = {}
            for name in MODEL_NAMES:
                matrix = np.asarray(
                    model_report[name]["position_matrix"], dtype=np.float64
                )
                correction = np.zeros_like(baseline_error)
                correction[history_valid] = _apply_linear(
                    features[name][history_valid], matrix
                )
                corrected_errors[name] = baseline_error - correction
            group_errors_by_history[history_length] = corrected_errors[
                "group_joint"
            ]
            history_result = {}
            for event_index, (event_name, event_value) in enumerate(
                EVENT_VALUES.items()
            ):
                event_mask = evaluation_partition.copy()
                if event_value is not None:
                    event_mask &= data["event"] == event_value
                history_result[event_name] = _paired_bootstrap(
                    data["trajectory"],
                    event_mask,
                    baseline_error,
                    corrected_errors,
                    args.samples,
                    seed=args.seed + 100 * history_length + event_index,
                )
            representation_result[history_text] = history_result
        output["results"][representation] = representation_result
        comparisons = {}
        for left, right in ((1, 4), (4, 8), (8, 16)):
            comparison = {}
            for event_index, (event_name, event_value) in enumerate(
                EVENT_VALUES.items()
            ):
                event_mask = evaluation_partition.copy()
                if event_value is not None:
                    event_mask &= data["event"] == event_value
                comparison[event_name] = _bootstrap_model_difference(
                    data["trajectory"],
                    event_mask,
                    group_errors_by_history[left],
                    group_errors_by_history[right],
                    args.samples,
                    seed=(
                        args.seed + 10000 * left + 100 * right + event_index
                    ),
                )
            comparisons[f"history_{right}_over_{left}"] = comparison
        output["cross_history_comparisons"][representation] = comparisons

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "raw": output["results"]["raw"],
    }, indent=2))


if __name__ == "__main__":
    main()
