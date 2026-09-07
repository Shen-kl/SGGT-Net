"""Measure the held-out upper bound of group-common maneuver information.

The Temporal checkpoint is rolled out online without altering its state.  At
each target step, oracle signals use the truth of *other* targets only:

* leave-one-out neighbour truth velocity increments; and
* leave-one-out neighbour Temporal prediction errors.

The candidate graph is taken at t-1, so topology and lifecycle are causal.
Future neighbour truth at t is deliberately allowed because this is an oracle
upper-bound diagnostic, never a deployable feature.  The target's own future
truth is excluded from group signals.  Synthetic group labels are evaluated
only as an additional ceiling and are not used to train the tracking model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from evaluation.compare_sean_temporal_v3 import build_model, load_compatible_state_dict
from train.trainer_v3 import Trainer


EVENT_NAMES = {-1: "unclassified", 0: "steady", 1: "switch", 2: "strong"}
STATE_KEYS = ("x", "vx", "y", "vy")


def _symmetric_adjacency(vector: torch.Tensor, node_count: int) -> torch.Tensor:
    """Rebuild a symmetric adjacency from an upper-triangle label vector."""

    adjacency = torch.zeros(
        node_count, node_count, dtype=torch.bool, device=vector.device
    )
    expected = node_count * (node_count - 1) // 2
    vector = vector.reshape(-1)
    if vector.numel() != expected:
        raise ValueError(
            f"structure vector has {vector.numel()} entries; expected {expected} "
            f"for {node_count} nodes"
        )
    row, column = torch.triu_indices(
        node_count, node_count, offset=1, device=vector.device
    )
    active = vector > 0.5
    adjacency[row[active], column[active]] = True
    adjacency[column[active], row[active]] = True
    return adjacency


def _edge_adjacency(
    edge_index: torch.Tensor, node_count: int, device: torch.device
) -> torch.Tensor:
    adjacency = torch.zeros(node_count, node_count, dtype=torch.bool, device=device)
    if edge_index.numel():
        source, target = edge_index
        usable = (source != target) & (source >= 0) & (target >= 0)
        adjacency[target[usable], source[usable]] = True
    return adjacency


def _leave_one_out_mean(
    values: torch.Tensor, adjacency: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate source values for each target, excluding target itself."""

    node_count = values.size(0)
    identity = torch.eye(node_count, dtype=torch.bool, device=values.device)
    usable = adjacency & ~identity
    usable &= valid.view(1, -1)
    usable &= valid.view(-1, 1)
    counts = usable.sum(dim=1)
    # Lifecycle-padded truth entries can be NaN. A zero adjacency coefficient
    # does not suppress them in IEEE arithmetic (0 * NaN is NaN), so sanitize
    # invalid sources before aggregation rather than relying on the mask alone.
    safe_values = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
    aggregate = usable.to(values.dtype) @ safe_values
    mean = aggregate / counts.clamp_min(1).to(values.dtype).unsqueeze(-1)
    mean = torch.where(
        counts.unsqueeze(-1) > 0, mean, torch.zeros_like(mean)
    )
    return mean, counts > 0, counts


class OracleCollector:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.parts: dict[str, list[torch.Tensor]] = defaultdict(list)

    def _append(self, name: str, value: torch.Tensor, mask: torch.Tensor):
        self.parts[name].append(value[mask].detach().cpu())

    def __call__(self, **values):
        graph_data = values["graph_data"]
        time_step = int(values["time_step"])
        outputs = values["outputs"]
        centroids = values["centroids"]
        event_class = values["event_class"]
        node_count = graph_data.x.size(0)

        current = graph_data.real_mask[:, time_step, 0].bool()
        previous = graph_data.real_mask[:, time_step - 1, 0].bool()
        valid = current & previous
        if not valid.any():
            return

        prediction = self.trainer._denormalize_prior(
            outputs, centroids, graph_data.batch
        )[:, 0]
        update = self.trainer._denormalize_update(
            outputs, centroids, graph_data.batch
        )[:, 0]
        truth = graph_data.y[:, time_step]
        previous_truth = graph_data.y[:, time_step - 1]
        truth_delta_v = truth[:, [1, 3]] - previous_truth[:, [1, 3]]
        prediction_error = truth - prediction
        update_error = truth - update

        geometric = _edge_adjacency(
            graph_data.edge_index[time_step - 1], node_count, truth.device
        )
        structure_vector = graph_data.struct_feat[time_step - 1].to(truth.device)
        true_group = _symmetric_adjacency(structure_vector, node_count)

        geo_dv, geo_has, geo_count = _leave_one_out_mean(
            truth_delta_v, geometric, valid
        )
        group_dv, group_has, group_count = _leave_one_out_mean(
            truth_delta_v, true_group, valid
        )
        geo_error, _, _ = _leave_one_out_mean(
            prediction_error, geometric, valid
        )
        group_error, _, _ = _leave_one_out_mean(
            prediction_error, true_group, valid
        )

        trajectory = graph_data.trajectory_index.reshape(-1)[graph_data.batch]
        frame = torch.full_like(trajectory, time_step)
        self._append("trajectory", trajectory, valid)
        self._append("frame", frame, valid)
        self._append("event", event_class, valid)
        self._append("prediction_error", prediction_error, valid)
        self._append("update_error", update_error, valid)
        self._append("own_dv", truth_delta_v, valid)
        self._append("geo_dv", geo_dv, valid)
        self._append("group_dv", group_dv, valid)
        self._append("geo_peer_error", geo_error, valid)
        self._append("group_peer_error", group_error, valid)
        self._append("geo_has", geo_has, valid)
        self._append("group_has", group_has, valid)
        self._append("geo_count", geo_count, valid)
        self._append("group_count", group_count, valid)

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self.parts:
            raise RuntimeError("oracle collector observed no valid node steps")
        return {name: torch.cat(parts, dim=0) for name, parts in self.parts.items()}


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def _fit_matrix(signal: np.ndarray, target: np.ndarray) -> np.ndarray:
    if signal.shape[0] == 0:
        return np.zeros((signal.shape[1], target.shape[1]), dtype=np.float64)
    # All oracle signals are two-dimensional.  Use an explicit ridge-stabilized
    # 2x2 normal-equation solution instead of np.linalg.lstsq: importing a
    # second BLAS/OpenMP runtime after CUDA PyTorch is unsafe on Windows.
    gram = np.array([
        [float(np.sum(signal[:, 0] * signal[:, 0])),
         float(np.sum(signal[:, 0] * signal[:, 1]))],
        [float(np.sum(signal[:, 1] * signal[:, 0])),
         float(np.sum(signal[:, 1] * signal[:, 1]))],
    ], dtype=np.float64)
    rhs = np.array([
        [float(np.sum(signal[:, 0] * target[:, 0])),
         float(np.sum(signal[:, 0] * target[:, 1]))],
        [float(np.sum(signal[:, 1] * target[:, 0])),
         float(np.sum(signal[:, 1] * target[:, 1]))],
    ], dtype=np.float64)
    ridge = max(float(np.trace(gram)) * 1e-9, 1e-9)
    a = float(gram[0, 0]) + ridge
    b = float(gram[0, 1])
    c = float(gram[1, 0])
    d = float(gram[1, 1]) + ridge
    determinant = a * d - b * c
    if abs(determinant) < 1e-18:
        return np.zeros((signal.shape[1], target.shape[1]), dtype=np.float64)
    inverse = np.array([[d, -b], [-c, a]], dtype=np.float64) / determinant
    return np.array([
        [inverse[0, 0] * rhs[0, 0] + inverse[0, 1] * rhs[1, 0],
         inverse[0, 0] * rhs[0, 1] + inverse[0, 1] * rhs[1, 1]],
        [inverse[1, 0] * rhs[0, 0] + inverse[1, 1] * rhs[1, 0],
         inverse[1, 0] * rhs[0, 1] + inverse[1, 1] * rhs[1, 1]],
    ], dtype=np.float64)


def _apply_matrix(signal: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x2 matrix without entering NumPy's BLAS backend."""

    return np.stack((
        signal[:, 0] * matrix[0, 0] + signal[:, 1] * matrix[1, 0],
        signal[:, 0] * matrix[0, 1] + signal[:, 1] * matrix[1, 1],
    ), axis=1)


def _fit_scalar(signal: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.square(signal).sum())
    return float((signal * target).sum() / max(denominator, 1e-12))


def _metric_masks(data: dict[str, np.ndarray], base: np.ndarray):
    yield "all", base
    for value, name in EVENT_NAMES.items():
        yield name, base & (data["event"] == value)


def _coherence(
    data: dict[str, np.ndarray], signal_name: str,
    calibration: np.ndarray, evaluation: np.ndarray,
) -> dict:
    has = data[signal_name.replace("_dv", "_has")].astype(bool)
    train = calibration & has
    signal = data[signal_name]
    target = data["own_dv"]
    matrix = _fit_matrix(signal[train], target[train])
    result = {"linear_matrix": matrix.tolist()}
    for name, mask in _metric_masks(data, evaluation & has):
        if not mask.any():
            continue
        x, y = signal[mask], target[mask]
        fitted = _apply_matrix(x, matrix)
        x_norm = np.sqrt(np.sum(np.square(x), axis=1))
        y_norm = np.sqrt(np.sum(np.square(y), axis=1))
        usable = (x_norm > 1e-8) & (y_norm > 1e-8)
        cosine = (
            np.sum(x[usable] * y[usable], axis=1)
            / np.maximum(x_norm[usable] * y_norm[usable], 1e-12)
            if usable.any() else np.empty(0)
        )
        zero_sse = float(np.square(y).sum())
        residual_sse = float(np.square(y - fitted).sum())
        common_fraction = float(
            np.square(x).sum() / max(np.square(y).sum(), 1e-12)
        )
        result[name] = {
            "count": int(mask.sum()),
            "coverage": float(mask.sum() / max((evaluation & (data["event"] == (
                next((key for key, value in EVENT_NAMES.items() if value == name), -999)
            ))).sum(), 1)) if name != "all" else float(mask.sum() / max(evaluation.sum(), 1)),
            "peer_count_mean": float(data[signal_name.replace("_dv", "_count")][mask].mean()),
            "direct_rmse_mps": _rmse(y - x),
            "linear_rmse_mps": _rmse(y - fitted),
            "linear_explained_fraction": 1.0 - residual_sse / max(zero_sse, 1e-12),
            "peer_to_own_energy_ratio": common_fraction,
            "cosine_mean": float(cosine.mean()) if cosine.size else None,
            "positive_cosine_fraction": float((cosine > 0).mean()) if cosine.size else None,
        }
    return result


def _physical_oracle(
    data: dict[str, np.ndarray], signal_name: str,
    calibration: np.ndarray, evaluation: np.ndarray, dt: float,
) -> dict:
    signal = data[signal_name]
    error = data["prediction_error"]
    train = calibration
    position_signal = dt * signal
    velocity_signal = signal
    beta_position = _fit_scalar(position_signal[train], error[train][:, [0, 2]])
    beta_velocity = _fit_scalar(velocity_signal[train], error[train][:, [1, 3]])
    matrix_position = _fit_matrix(signal[train], error[train][:, [0, 2]])
    matrix_velocity = _fit_matrix(signal[train], error[train][:, [1, 3]])
    result = {
        "beta_position": beta_position,
        "beta_velocity": beta_velocity,
        "position_matrix": matrix_position.tolist(),
        "velocity_matrix": matrix_velocity.tolist(),
    }
    for event, mask in _metric_masks(data, evaluation):
        if not mask.any():
            continue
        e = error[mask]
        x = signal[mask]
        direct_pos = e[:, [0, 2]] - dt * x
        direct_vel = e[:, [1, 3]] - x
        scalar_pos = e[:, [0, 2]] - beta_position * dt * x
        scalar_vel = e[:, [1, 3]] - beta_velocity * x
        matrix_pos = e[:, [0, 2]] - _apply_matrix(x, matrix_position)
        matrix_vel = e[:, [1, 3]] - _apply_matrix(x, matrix_velocity)
        result[event] = {
            "count": int(mask.sum()),
            "baseline_position_rmse_m": _rmse(e[:, [0, 2]]),
            "baseline_velocity_rmse_mps": _rmse(e[:, [1, 3]]),
            "direct_position_rmse_m": _rmse(direct_pos),
            "direct_velocity_rmse_mps": _rmse(direct_vel),
            "scalar_position_rmse_m": _rmse(scalar_pos),
            "scalar_velocity_rmse_mps": _rmse(scalar_vel),
            "linear_position_rmse_m": _rmse(matrix_pos),
            "linear_velocity_rmse_mps": _rmse(matrix_vel),
        }
        for prefix in ("scalar", "linear"):
            result[event][f"{prefix}_position_improvement_m"] = (
                result[event]["baseline_position_rmse_m"]
                - result[event][f"{prefix}_position_rmse_m"]
            )
            result[event][f"{prefix}_position_improvement_percent"] = 100.0 * (
                result[event][f"{prefix}_position_improvement_m"]
                / max(result[event]["baseline_position_rmse_m"], 1e-12)
            )
    return result


def _peer_error_oracle(
    data: dict[str, np.ndarray], signal_name: str,
    calibration: np.ndarray, evaluation: np.ndarray,
) -> dict:
    signal = data[signal_name]
    error = data["prediction_error"]
    beta_position = _fit_scalar(
        signal[calibration][:, [0, 2]], error[calibration][:, [0, 2]]
    )
    beta_velocity = _fit_scalar(
        signal[calibration][:, [1, 3]], error[calibration][:, [1, 3]]
    )
    result = {"beta_position": beta_position, "beta_velocity": beta_velocity}
    for event, mask in _metric_masks(data, evaluation):
        if not mask.any():
            continue
        e, x = error[mask], signal[mask]
        corrected_pos = e[:, [0, 2]] - beta_position * x[:, [0, 2]]
        corrected_vel = e[:, [1, 3]] - beta_velocity * x[:, [1, 3]]
        base_pos = _rmse(e[:, [0, 2]])
        result[event] = {
            "count": int(mask.sum()),
            "baseline_position_rmse_m": base_pos,
            "baseline_velocity_rmse_mps": _rmse(e[:, [1, 3]]),
            "oracle_position_rmse_m": _rmse(corrected_pos),
            "oracle_velocity_rmse_mps": _rmse(corrected_vel),
        }
        result[event]["position_improvement_m"] = (
            base_pos - result[event]["oracle_position_rmse_m"]
        )
        result[event]["position_improvement_percent"] = 100.0 * (
            result[event]["position_improvement_m"] / max(base_pos, 1e-12)
        )
    return result


def analyse(
    tensors: dict[str, torch.Tensor], calibration_trajectories: int, dt: float
) -> dict:
    data = {name: value.numpy() for name, value in tensors.items()}
    boundary = int(calibration_trajectories)
    calibration = data["trajectory"] < boundary
    evaluation = data["trajectory"] >= boundary
    if not calibration.any() or not evaluation.any():
        # Small smoke tests cannot form a held-out split. Keep this explicit.
        calibration = np.ones_like(data["trajectory"], dtype=bool)
        evaluation = calibration.copy()
        split_kind = "in_sample_smoke"
    else:
        split_kind = "held_out_by_trajectory"

    report = {
        "split": {
            "kind": split_kind,
            "calibration_trajectory_stop_exclusive": boundary,
            "calibration_points": int(calibration.sum()),
            "evaluation_points": int(evaluation.sum()),
        },
        "baseline": {},
        "maneuver_coherence": {},
        "maneuver_oracles": {},
        "shared_error_oracles": {},
    }
    for event, mask in _metric_masks(data, evaluation):
        if not mask.any():
            continue
        error = data["prediction_error"][mask]
        update_error = data["update_error"][mask]
        report["baseline"][event] = {
            "count": int(mask.sum()),
            "prediction_position_rmse_m": _rmse(error[:, [0, 2]]),
            "prediction_velocity_rmse_mps": _rmse(error[:, [1, 3]]),
            "update_position_rmse_m": _rmse(update_error[:, [0, 2]]),
            "update_velocity_rmse_mps": _rmse(update_error[:, [1, 3]]),
        }
    for graph_name in ("geo", "group"):
        report["maneuver_coherence"][graph_name] = _coherence(
            data, f"{graph_name}_dv", calibration, evaluation
        )
        report["maneuver_oracles"][graph_name] = _physical_oracle(
            data, f"{graph_name}_dv", calibration, evaluation, dt
        )
        report["shared_error_oracles"][graph_name] = _peer_error_oracle(
            data, f"{graph_name}_peer_error", calibration, evaluation
        )
    report["maneuver_oracles"]["individual_future_dv_ceiling"] = _physical_oracle(
        data, "own_dv", calibration, evaluation, dt
    )
    return report


def _checkpoint_args(payload: dict, device: torch.device):
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = device
    for name, default in (
        ("decoder_use_GASF", True), ("decoder_use_SEAN", False),
        ("model_variant", "sggt_v2"), ("covariance_transition", "cv"),
        ("gasf_filter_mode", "abc"), ("gasf_graph_mode", "temporal_only"),
        ("gasf_fusion_mode", "concat"), ("gasf_aux_source", "fusion"),
        ("innovation_decomposition", False), ("innovation_whiten_clip", 20.0),
        ("adaptive_q_mode", "legacy"),
        ("interaction_physical_residual", False),
        ("interaction_physical_max_gain", 0.075),
        ("interaction_physical_init_gain", 0.05),
        ("interaction_activation_floor", 0.0),
    ):
        setattr(args, name, payload.get(name, default))
    args.counterfactual_training = False
    args.encoder_execution_mode = "stream"
    args.full_time_supervision = True
    args.eval_batch_size = 1
    args.report_trajectory_quantiles = False
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_trajectories", type=int, default=200)
    parser.add_argument("--calibration_trajectories", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()

    device = torch.device(cli.device)
    checkpoint = Path(cli.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    args = _checkpoint_args(payload, device)
    config = dict(payload.get("config", CONFIG))
    config.update({"S_VEL": 40.0, "S_EDGE_V_DIST": 40.0, "output_graph_window_len": 1})
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
    collector = OracleCollector(trainer)
    trainer.step_observer = collector
    start = time.perf_counter()
    try:
        validation_metrics = trainer.validate(loader, "Group maneuver oracle rollout")
    finally:
        trainer.step_observer = None
    elapsed = time.perf_counter() - start
    tensors = collector.tensors()
    analysis = analyse(tensors, cli.calibration_trajectories, float(args.T))
    report = {
        "protocol": {
            "version": 1,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "dataset_root": str(Path(cli.data_dir).resolve()),
            "split": cli.split,
            "trajectory_count": stop,
            "dt_seconds": float(args.T),
            "online_temporal_rollout": True,
            "topology_time": "t-1",
            "oracle_information_time": "neighbour truth at t",
            "target_future_truth_in_group_signal": False,
            "synthetic_group_labels_for_upper_bound_only": True,
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
    print(json.dumps({
        "output": str(output),
        "protocol": report["protocol"],
        "split": analysis["split"],
        "baseline": analysis["baseline"].get("all"),
        "geo_maneuver": analysis["maneuver_oracles"]["geo"].get("all"),
        "group_maneuver": analysis["maneuver_oracles"]["group"].get("all"),
        "individual": analysis["maneuver_oracles"]["individual_future_dv_ceiling"].get("all"),
    }, indent=2))


if __name__ == "__main__":
    main()
