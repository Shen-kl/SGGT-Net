"""Held-out diagnostic for strictly causal group-innovation predictability.

For the prediction made at frame ``t``, every feature in this evaluator ends
at ``t-1``.  The latest innovation is the physical EKF innovation written only
after the preceding measurement update.  The target's current measurement or
truth is never used as an input feature.

The evaluator compares the target's own innovation history with leave-one-out
geometric-neighbour and synthetic true-group histories.  The latter is an
oracle-topology ceiling only.  Small ridge-linear correction maps are fitted
on one trajectory partition and reported on a disjoint held-out partition.
They diagnose usable information; they are not tracking-model results.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from evaluation.compare_sean_temporal_v3 import build_model, load_compatible_state_dict
from evaluation.evaluate_group_maneuver_oracle_v3 import (
    EVENT_NAMES,
    _checkpoint_args,
    _edge_adjacency,
    _leave_one_out_mean,
    _rmse,
    _symmetric_adjacency,
)
from train.trainer_v3 import Trainer


DEFAULT_HISTORY_LENGTHS = (1, 4, 8, 16)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _roll_append(buffer: torch.Tensor, latest: torch.Tensor) -> torch.Tensor:
    """Return a right-aligned causal history after appending one frame."""

    if latest.ndim == buffer.ndim - 1:
        latest = latest.unsqueeze(1)
    if latest.shape[0] != buffer.shape[0] or latest.shape[2:] != buffer.shape[2:]:
        raise ValueError(
            f"cannot append {tuple(latest.shape)} to {tuple(buffer.shape)}"
        )
    return torch.cat((buffer[:, 1:], latest), dim=1)


class CausalInnovationCollector:
    """Collect aligned own and peer histories during batch-one online rollout."""

    def __init__(self, trainer: Trainer, history_length: int):
        self.trainer = trainer
        self.history_length = int(history_length)
        self.parts: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.active_trajectory: int | None = None
        self.peer_buffers: dict[str, torch.Tensor] = {}

    def _append(self, name: str, value: torch.Tensor, mask: torch.Tensor):
        self.parts[name].append(value[mask].detach().cpu())

    def _reset_peer_buffers(self, trajectory: int, reference: torch.Tensor):
        self.active_trajectory = trajectory
        shape = (reference.size(0), self.history_length, reference.size(-1))
        self.peer_buffers = {
            f"{graph}_{representation}": reference.new_zeros(shape)
            for graph in ("geo", "group")
            for representation in ("raw", "white")
        }
        bool_shape = (reference.size(0), self.history_length, 1)
        self.peer_buffers.update({
            f"{graph}_has": torch.zeros(
                bool_shape, dtype=torch.bool, device=reference.device
            )
            for graph in ("geo", "group")
        })

    @staticmethod
    def _right_align(history: torch.Tensor, length: int) -> torch.Tensor:
        history = history[:, -length:]
        if history.size(1) == length:
            return history
        padding = history.new_zeros(
            history.size(0), length - history.size(1), *history.shape[2:]
        )
        return torch.cat((padding, history), dim=1)

    def __call__(self, **values):
        graph_data = values["graph_data"]
        window = values["window"]
        time_step = int(values["time_step"])
        outputs = values["outputs"]
        centroids = values["centroids"]
        event_class = values["event_class"]

        if int(graph_data.batch.max()) != 0:
            raise ValueError("causal oracle requires evaluation batch_size=1")
        trajectory = int(graph_data.trajectory_index.reshape(-1)[0])
        node_count = graph_data.x.size(0)

        own_raw = self._right_align(window.innovation_raw, self.history_length)
        own_white = self._right_align(
            window.innovation_whitened, self.history_length
        )
        history_valid = self._right_align(
            window.innovation_valid.bool(), self.history_length
        )
        if self.active_trajectory != trajectory:
            self._reset_peer_buffers(trajectory, own_raw)

        # The latest buffer entry is innovation at t-1.  Its topology is also
        # taken at t-1 and was fully observed before predicting frame t.
        latest_valid = history_valid[:, -1, 0]
        geometric = _edge_adjacency(
            graph_data.edge_index[time_step - 1], node_count, own_raw.device
        )
        group = _symmetric_adjacency(
            graph_data.struct_feat[time_step - 1].to(own_raw.device), node_count
        )
        latest_features = torch.cat(
            (own_raw[:, -1], own_white[:, -1]), dim=-1
        )
        for graph_name, adjacency in (("geo", geometric), ("group", group)):
            peer, has_peer, _ = _leave_one_out_mean(
                latest_features, adjacency, latest_valid
            )
            self.peer_buffers[f"{graph_name}_raw"] = _roll_append(
                self.peer_buffers[f"{graph_name}_raw"], peer[:, :2]
            )
            self.peer_buffers[f"{graph_name}_white"] = _roll_append(
                self.peer_buffers[f"{graph_name}_white"], peer[:, 2:]
            )
            self.peer_buffers[f"{graph_name}_has"] = _roll_append(
                self.peer_buffers[f"{graph_name}_has"], has_peer.unsqueeze(-1)
            )

        current = graph_data.real_mask[:, time_step, 0].bool()
        previous = graph_data.real_mask[:, time_step - 1, 0].bool()
        prediction_valid = current & previous
        if not prediction_valid.any():
            return

        prediction = self.trainer._denormalize_prior(
            outputs, centroids, graph_data.batch
        )[:, 0]
        update = self.trainer._denormalize_update(
            outputs, centroids, graph_data.batch
        )[:, 0]
        truth = graph_data.y[:, time_step]
        prediction_error = truth - prediction
        update_error = truth - update
        trajectory_tensor = torch.full(
            (node_count,), trajectory, dtype=torch.long, device=truth.device
        )
        frame = torch.full_like(trajectory_tensor, time_step)

        self._append("trajectory", trajectory_tensor, prediction_valid)
        self._append("frame", frame, prediction_valid)
        self._append("event", event_class, prediction_valid)
        self._append("prediction_error", prediction_error, prediction_valid)
        self._append("update_error", update_error, prediction_valid)
        self._append("own_raw_history", own_raw, prediction_valid)
        self._append("own_white_history", own_white, prediction_valid)
        self._append("history_valid", history_valid, prediction_valid)
        for name, value in self.peer_buffers.items():
            self._append(f"{name}_history", value, prediction_valid)

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self.parts:
            raise RuntimeError("causal innovation collector observed no valid steps")
        return {name: torch.cat(parts, dim=0) for name, parts in self.parts.items()}


def _solve_ridge(signal: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit a small no-intercept ridge map without invoking NumPy BLAS."""

    rows, dimension = signal.shape
    output_dimension = target.shape[1]
    if rows == 0:
        return np.zeros((dimension, output_dimension), dtype=np.float64)
    scale = np.sqrt(np.mean(np.square(signal), axis=0))
    scale = np.maximum(scale, 1e-8)
    standardized = signal / scale
    gram = np.empty((dimension, dimension), dtype=np.float64)
    rhs = np.empty((dimension, output_dimension), dtype=np.float64)
    for row in range(dimension):
        for column in range(dimension):
            gram[row, column] = float(
                np.sum(standardized[:, row] * standardized[:, column])
            )
        for column in range(output_dimension):
            rhs[row, column] = float(
                np.sum(standardized[:, row] * target[:, column])
            )
    ridge = max(float(np.trace(gram)) / max(dimension, 1) * 1e-6, 1e-8)
    gram[np.diag_indices(dimension)] += ridge
    augmented = np.concatenate((gram, rhs), axis=1)

    # Gauss-Jordan elimination with partial pivoting is sufficient here:
    # dimension is at most 64 and ridge makes the system nonsingular.
    for pivot in range(dimension):
        best = pivot + int(np.argmax(np.abs(augmented[pivot:, pivot])))
        if best != pivot:
            augmented[[pivot, best]] = augmented[[best, pivot]]
        pivot_value = float(augmented[pivot, pivot])
        if abs(pivot_value) < 1e-14:
            continue
        augmented[pivot] /= pivot_value
        for row in range(dimension):
            if row == pivot:
                continue
            factor = float(augmented[row, pivot])
            if factor != 0.0:
                augmented[row] -= factor * augmented[pivot]
    standardized_matrix = augmented[:, dimension:]
    return standardized_matrix / scale[:, None]


def _apply_linear(signal: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.sum(signal[:, :, None] * matrix[None, :, :], axis=1)


def _event_masks(data: dict[str, np.ndarray], base: np.ndarray):
    yield "all", base
    for value, name in EVENT_NAMES.items():
        yield name, base & (data["event"] == value)


def _stratified_derangement(
    data: dict[str, np.ndarray],
    calibration_partition: np.ndarray,
    evaluation_partition: np.ndarray,
    seed: int,
    extra_stratum: np.ndarray | None = None,
) -> np.ndarray:
    """Shuffle samples within partition/event strata without fixed points."""

    permutation = np.arange(data["event"].shape[0])
    rng = np.random.default_rng(seed)
    event_values = tuple(EVENT_NAMES) + (999,)
    classified = np.isin(data["event"], tuple(EVENT_NAMES))
    if extra_stratum is None:
        extra_stratum = np.zeros_like(data["event"], dtype=np.int64)
    for partition in (calibration_partition, evaluation_partition):
        for event_value in event_values:
            event_mask = (
                ~classified if event_value == 999
                else data["event"] == event_value
            )
            for stratum_value in np.unique(extra_stratum):
                indices = np.flatnonzero(
                    partition & event_mask & (extra_stratum == stratum_value)
                )
                if indices.size <= 1:
                    continue
                ordered = indices.copy()
                rng.shuffle(ordered)
                # Mapping each shuffled position to the preceding position is
                # a derangement while retaining partition/event/validity.
                permutation[ordered] = np.roll(ordered, 1)
    return permutation


def _fit_correction(
    data: dict[str, np.ndarray],
    feature: np.ndarray,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    population_evaluation: np.ndarray | None = None,
) -> dict:
    error = data["prediction_error"]
    finite = np.isfinite(feature).all(axis=1) & np.isfinite(error).all(axis=1)
    train = calibration & finite
    position_matrix = _solve_ridge(feature[train], error[train][:, [0, 2]])
    velocity_matrix = _solve_ridge(feature[train], error[train][:, [1, 3]])
    result = {
        "position_matrix": position_matrix.tolist(),
        "velocity_matrix": velocity_matrix.tolist(),
    }
    for event, mask in _event_masks(data, evaluation & finite):
        if not mask.any():
            continue
        current_error = error[mask]
        current_feature = feature[mask]
        corrected_position = (
            current_error[:, [0, 2]]
            - _apply_linear(current_feature, position_matrix)
        )
        corrected_velocity = (
            current_error[:, [1, 3]]
            - _apply_linear(current_feature, velocity_matrix)
        )
        baseline_position = _rmse(current_error[:, [0, 2]])
        baseline_velocity = _rmse(current_error[:, [1, 3]])
        position = _rmse(corrected_position)
        velocity = _rmse(corrected_velocity)
        result[event] = {
            "count": int(mask.sum()),
            "baseline_position_rmse_m": baseline_position,
            "baseline_velocity_rmse_mps": baseline_velocity,
            "corrected_position_rmse_m": position,
            "corrected_velocity_rmse_mps": velocity,
            "position_improvement_m": baseline_position - position,
            "position_improvement_percent": 100.0 * (
                (baseline_position - position) / max(baseline_position, 1e-12)
            ),
            "velocity_improvement_mps": baseline_velocity - velocity,
            "velocity_improvement_percent": 100.0 * (
                (baseline_velocity - velocity) / max(baseline_velocity, 1e-12)
            ),
        }
    if population_evaluation is not None:
        hybrid = {}
        for event, mask in _event_masks(data, population_evaluation & finite):
            if not mask.any():
                continue
            current_error = error[mask]
            current_feature = feature[mask]
            eligible = evaluation[mask]
            position_correction = np.zeros(
                (current_error.shape[0], 2), dtype=np.float64
            )
            velocity_correction = np.zeros_like(position_correction)
            if eligible.any():
                position_correction[eligible] = _apply_linear(
                    current_feature[eligible], position_matrix
                )
                velocity_correction[eligible] = _apply_linear(
                    current_feature[eligible], velocity_matrix
                )
            baseline_position = _rmse(current_error[:, [0, 2]])
            baseline_velocity = _rmse(current_error[:, [1, 3]])
            position = _rmse(
                current_error[:, [0, 2]] - position_correction
            )
            velocity = _rmse(
                current_error[:, [1, 3]] - velocity_correction
            )
            hybrid[event] = {
                "count": int(mask.sum()),
                "eligible_count": int(eligible.sum()),
                "eligible_fraction": float(eligible.mean()),
                "baseline_position_rmse_m": baseline_position,
                "baseline_velocity_rmse_mps": baseline_velocity,
                "corrected_position_rmse_m": position,
                "corrected_velocity_rmse_mps": velocity,
                "position_improvement_m": baseline_position - position,
                "position_improvement_percent": 100.0 * (
                    (baseline_position - position)
                    / max(baseline_position, 1e-12)
                ),
                "velocity_improvement_mps": baseline_velocity - velocity,
                "velocity_improvement_percent": 100.0 * (
                    (baseline_velocity - velocity)
                    / max(baseline_velocity, 1e-12)
                ),
            }
        result["population_hybrid"] = hybrid
    return result


def analyse(
    tensors: dict[str, torch.Tensor],
    calibration_trajectories: int,
    history_lengths: tuple[int, ...],
) -> dict:
    data = {name: value.numpy() for name, value in tensors.items()}
    boundary = int(calibration_trajectories)
    calibration_partition = data["trajectory"] < boundary
    evaluation_partition = data["trajectory"] >= boundary
    if not calibration_partition.any() or not evaluation_partition.any():
        calibration_partition = np.ones_like(data["trajectory"], dtype=bool)
        evaluation_partition = calibration_partition.copy()
        split_kind = "in_sample_smoke"
    else:
        split_kind = "held_out_by_trajectory"

    report = {
        "split": {
            "kind": split_kind,
            "calibration_trajectory_stop_exclusive": boundary,
            "calibration_prediction_points": int(calibration_partition.sum()),
            "evaluation_prediction_points": int(evaluation_partition.sum()),
        },
        "history_lengths": {},
    }
    for history_length in history_lengths:
        history_valid = data["history_valid"][:, -history_length:, 0].all(axis=1)
        shuffled_indices = _stratified_derangement(
            data,
            calibration_partition,
            evaluation_partition,
            seed=20260812 + history_length,
            extra_stratum=history_valid.astype(np.int64),
        )
        calibration = calibration_partition & history_valid
        evaluation = evaluation_partition & history_valid
        length_report = {
            "calibration_points": int(calibration.sum()),
            "evaluation_points": int(evaluation.sum()),
            "evaluation_history_coverage": float(
                evaluation.sum() / max(evaluation_partition.sum(), 1)
            ),
            "peer_coverage": {},
            "representations": {},
        }
        for graph_name in ("geo", "group"):
            has = data[f"{graph_name}_has_history"][
                :, -history_length:, 0
            ].all(axis=1)
            length_report["peer_coverage"][graph_name] = {
                "all_lags_evaluation_fraction": float(
                    (evaluation & has).sum() / max(evaluation.sum(), 1)
                )
            }

        for representation, own_name in (
            ("raw", "own_raw_history"),
            ("whitened", "own_white_history"),
        ):
            own = data[own_name][:, -history_length:].reshape(
                data[own_name].shape[0], -1
            )
            representation_report = {}
            representation_report["own"] = _fit_correction(
                data, own, calibration, evaluation, evaluation_partition
            )
            for graph_name in ("geo", "group"):
                peer_name = (
                    f"{graph_name}_raw_history"
                    if representation == "raw"
                    else f"{graph_name}_white_history"
                )
                peer = data[peer_name][:, -history_length:].reshape(
                    data[peer_name].shape[0], -1
                )
                representation_report[f"{graph_name}_peer"] = _fit_correction(
                    data, peer, calibration, evaluation, evaluation_partition
                )
                joint = np.concatenate((own, peer), axis=1)
                representation_report[f"{graph_name}_joint"] = _fit_correction(
                    data, joint, calibration, evaluation, evaluation_partition
                )

                if graph_name == "group":
                    shuffled_peer = peer[shuffled_indices]
                    shuffled_joint = np.concatenate((own, shuffled_peer), axis=1)
                    representation_report["group_joint_shuffled_control"] = (
                        _fit_correction(
                            data,
                            shuffled_joint,
                            calibration,
                            evaluation,
                            evaluation_partition,
                        )
                    )

            increments = {}
            for event, _ in _event_masks(data, evaluation):
                if event not in representation_report["own"]:
                    continue
                own_metric = representation_report["own"][event]
                geo_metric = representation_report["geo_joint"][event]
                group_metric = representation_report["group_joint"][event]
                increments[event] = {
                    "group_joint_gain_over_temporal_m": group_metric[
                        "position_improvement_m"
                    ],
                    "group_increment_over_own_history_m": (
                        own_metric["corrected_position_rmse_m"]
                        - group_metric["corrected_position_rmse_m"]
                    ),
                    "true_group_topology_increment_over_geometry_m": (
                        geo_metric["corrected_position_rmse_m"]
                        - group_metric["corrected_position_rmse_m"]
                    ),
                    "group_peer_only_gain_over_temporal_m": (
                        representation_report["group_peer"][event][
                            "position_improvement_m"
                        ]
                    ),
                }
            representation_report["increments"] = increments
            hybrid_increments = {}
            for event, _ in _event_masks(data, evaluation_partition):
                own_metric = representation_report["own"][
                    "population_hybrid"
                ].get(event)
                geo_metric = representation_report["geo_joint"][
                    "population_hybrid"
                ].get(event)
                group_metric = representation_report["group_joint"][
                    "population_hybrid"
                ].get(event)
                shuffled_metric = representation_report[
                    "group_joint_shuffled_control"
                ]["population_hybrid"].get(event)
                if not all(
                    metric is not None for metric in
                    (own_metric, geo_metric, group_metric, shuffled_metric)
                ):
                    continue
                hybrid_increments[event] = {
                    "group_joint_gain_over_temporal_m": group_metric[
                        "position_improvement_m"
                    ],
                    "group_increment_over_own_history_m": (
                        own_metric["corrected_position_rmse_m"]
                        - group_metric["corrected_position_rmse_m"]
                    ),
                    "true_group_topology_increment_over_geometry_m": (
                        geo_metric["corrected_position_rmse_m"]
                        - group_metric["corrected_position_rmse_m"]
                    ),
                    "shuffled_group_joint_gain_m": shuffled_metric[
                        "position_improvement_m"
                    ],
                    "aligned_increment_over_shuffled_control_m": (
                        shuffled_metric["corrected_position_rmse_m"]
                        - group_metric["corrected_position_rmse_m"]
                    ),
                    "eligible_fraction": group_metric["eligible_fraction"],
                }
            representation_report["population_hybrid_increments"] = (
                hybrid_increments
            )
            length_report["representations"][representation] = representation_report
        report["history_lengths"][str(history_length)] = length_report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_trajectories", type=int, default=200)
    parser.add_argument("--calibration_trajectories", type=int, default=100)
    parser.add_argument(
        "--history_lengths", type=int, nargs="+", default=DEFAULT_HISTORY_LENGTHS
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--tensor_cache", default="",
        help="optional torch file for collected rollout tensors",
    )
    cli = parser.parse_args()

    history_lengths = tuple(sorted(set(int(value) for value in cli.history_lengths)))
    if not history_lengths or history_lengths[0] <= 0:
        raise ValueError("history lengths must be positive")
    device = torch.device(cli.device)
    checkpoint = Path(cli.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    args = _checkpoint_args(payload, device)
    if max(history_lengths) > int(max(args.decoder_residual_length)):
        raise ValueError(
            f"requested history {max(history_lengths)} exceeds model buffer "
            f"{max(args.decoder_residual_length)}"
        )
    config = dict(payload.get("config", CONFIG))
    config.update({
        "S_VEL": 40.0,
        "S_EDGE_V_DIST": 40.0,
        "output_graph_window_len": 1,
    })
    CONFIG.update(config)
    model = build_model(args).to(device)
    load_compatible_state_dict(model, payload["state_dict"])
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    trainer = Trainer(model, optimizer, scheduler, None, config, args)

    dataset = GroupTargetTrackingDatasetV3(
        cli.data_dir, cli.split, load_structure=True, load_analysis_metadata=False
    )
    stop = len(dataset)
    if cli.max_trajectories > 0:
        stop = min(stop, int(cli.max_trajectories))
    loader = DataLoader(
        Subset(dataset, range(stop)), batch_size=1, shuffle=False, num_workers=0
    )
    collector = CausalInnovationCollector(trainer, max(history_lengths))
    trainer.step_observer = collector
    start = time.perf_counter()
    try:
        validation_metrics = trainer.validate(loader, "Causal innovation oracle")
    finally:
        trainer.step_observer = None
    elapsed = time.perf_counter() - start
    tensors = collector.tensors()
    if cli.tensor_cache:
        cache = Path(cli.tensor_cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_sha256": _file_sha256(checkpoint),
            "trajectory_count": stop,
            "history_length": max(history_lengths),
            "tensors": tensors,
        }, cache)
    analysis = analyse(tensors, cli.calibration_trajectories, history_lengths)
    report = {
        "protocol": {
            "version": 1,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "dataset_root": str(Path(cli.data_dir).resolve()),
            "split": cli.split,
            "trajectory_count": stop,
            "online_temporal_rollout": True,
            "feature_information_cutoff": "t-1 after measurement update",
            "prediction_target_time": "t before measurement update",
            "current_measurement_or_truth_used_as_feature": False,
            "topology_time_for_each_innovation": "same historical frame",
            "target_excluded_from_peer_signal": True,
            "synthetic_group_labels_for_oracle_topology_only": True,
            "history_lengths": list(history_lengths),
            "tensor_cache": str(Path(cli.tensor_cache).resolve()) if cli.tensor_cache else None,
            "elapsed_seconds": elapsed,
        },
        "validation_metrics": {
            key: float(value) for key, value in validation_metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        "analysis": analysis,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    concise = {}
    for history_length in history_lengths:
        length_report = analysis["history_lengths"][str(history_length)]
        concise[str(history_length)] = {
            "evaluation_points": length_report["evaluation_points"],
            "raw": length_report["representations"]["raw"][
                "population_hybrid_increments"
            ],
            "whitened": length_report["representations"]["whitened"][
                "population_hybrid_increments"
            ],
        }
    print(json.dumps({
        "output": str(output),
        "protocol": report["protocol"],
        "split": analysis["split"],
        "summary": concise,
    }, indent=2))


if __name__ == "__main__":
    main()
