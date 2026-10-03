"""Evaluate an SGGT checkpoint on scene02 with known target identities.

This diagnostic preserves the model's causal online rollout, EKF measurement
update, covariance propagation, and innovation history.  It removes detection,
clutter, BP association, and track-management decisions by assigning one noisy
measurement stream directly to each ground-truth target identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from evaluation.compare_sean_temporal_v3 import (
    build_model,
    load_compatible_state_dict,
)
from train.trainer_v3 import Trainer


CHECKPOINT_ARGUMENTS = (
    "decoder_use_GASF",
    "decoder_use_SEAN",
    "model_variant",
    "covariance_transition",
    "gasf_filter_mode",
    "gasf_graph_mode",
    "gasf_fusion_mode",
    "gasf_aux_source",
    "innovation_decomposition",
    "innovation_whiten_clip",
    "adaptive_q_mode",
    "adaptive_q_max_multiplier",
    "adaptive_q_max_position_variance",
    "adaptive_q_initial_multiplier",
    "adaptive_q_initial_position_variance",
    "adaptive_q_acceleration",
    "interaction_physical_residual",
    "interaction_physical_max_gain",
    "interaction_physical_init_gain",
    "interaction_activation_floor",
    "causal_innovation_correction",
    "causal_innovation_history",
    "causal_innovation_hidden_size",
    "causal_innovation_output_mode",
    "causal_innovation_max_acceleration_mps2",
    "causal_innovation_max_position_m",
    "causal_innovation_max_velocity_mps",
    "causal_innovation_dynamic_activation",
    "causal_innovation_activation_floor",
    "causal_innovation_activation_nis_low",
    "causal_innovation_activation_nis_high",
    "causal_innovation_minimum_valid_history",
    "causal_innovation_recent_valid_window",
    "causal_innovation_association_context",
    "causal_innovation_use_peer",
    "association_augmentation",
    "association_pd_min",
    "association_pd_max",
    "association_clutter_mu_min",
    "association_clutter_mu_max",
    "association_surveillance_side_m",
    "association_max_clutter",
    "association_local_clutter_rate",
    "association_max_local_clutter",
    "association_local_clutter_scale",
    "association_gate_threshold",
    "association_minimum_detection_mass",
    "association_include_target_competitors",
    "association_clutter_requires_detection",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def noisy_identity_measurements(
    trajectory_info: dict,
    seed: int,
    range_noise_std_m: float,
    azimuth_noise_std_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return truth, measurements, and lifecycle mask with fixed identities."""

    target_count = int(trajectory_info["target_num"])
    frame_count = int(trajectory_info["frame_max"])
    truth = np.zeros((target_count, frame_count, 4), dtype=np.float32)
    measurement = np.zeros((target_count, frame_count, 2), dtype=np.float32)
    valid = np.zeros((target_count, frame_count), dtype=bool)
    # Match the legacy generator's RandomState and target-by-target draw order.
    rng = np.random.RandomState(int(seed))
    for target_id, (track, lifecycle) in enumerate(zip(
        trajectory_info["trajectory"], trajectory_info["born_dead_set"]
    )):
        track = np.asarray(track, dtype=np.float64)
        born, dead = (int(lifecycle[0]) - 1, int(lifecycle[1]))
        length = dead - born
        if track.ndim != 2 or track.shape[0] != 4 or track.shape[1] < length:
            raise ValueError(
                f"target {target_id}: track shape {track.shape} does not match "
                f"or cover lifecycle [{born + 1}, {dead}]"
            )
        # The legacy scene stores one redundant terminal state for targets
        # born after frame 1.  generateMeasurements indexes only 0:length, and
        # labels_set confirms that the extra column is outside the lifecycle.
        track = track[:, :length]
        truth[target_id, born:dead] = track.T.astype(np.float32)
        valid[target_id, born:dead] = True
        x, y = track[0], track[2]
        radius = np.hypot(x, y)
        azimuth_deg = np.degrees(np.arctan2(x, y))
        noisy_radius = radius + float(range_noise_std_m) * rng.randn(length)
        noisy_azimuth = np.radians(
            azimuth_deg + float(azimuth_noise_std_deg) * rng.randn(length)
        )
        measurement[target_id, born:dead, 0] = (
            noisy_radius * np.sin(noisy_azimuth)
        ).astype(np.float32)
        measurement[target_id, born:dead, 1] = (
            noisy_radius * np.cos(noisy_azimuth)
        ).astype(np.float32)
    return (
        torch.from_numpy(truth),
        torch.from_numpy(measurement),
        torch.from_numpy(valid),
    )


def initial_state_from_measurements(
    measurement: torch.Tensor, valid: torch.Tensor, delta_t: float
) -> torch.Tensor:
    """Build the non-oracle finite-difference state used by Dataset2D v3."""

    state = measurement.new_zeros((*measurement.shape[:-1], 4))
    state[..., 0] = measurement[..., 0]
    state[..., 2] = measurement[..., 1]
    continuing = valid[:, 1:] & valid[:, :-1]
    velocity = (measurement[:, 1:] - measurement[:, :-1]) / float(delta_t)
    state[:, 1:, 1] = torch.where(continuing, velocity[..., 0], 0.0)
    state[:, 1:, 3] = torch.where(continuing, velocity[..., 1], 0.0)
    return state


def input_graph(
    state: torch.Tensor,
    valid: torch.Tensor,
    distance_threshold_m: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    edge_indices, edge_features = [], []
    for frame in range(state.size(1)):
        active = torch.nonzero(valid[:, frame], as_tuple=False).flatten()
        if active.numel() == 0:
            edge_indices.append(torch.empty(2, 0, dtype=torch.long))
            edge_features.append(torch.empty(0, 2, dtype=state.dtype))
            continue
        positions = state[active, frame][:, [0, 2]]
        velocities = state[active, frame][:, [1, 3]]
        position_distance = torch.cdist(positions, positions)
        velocity_distance = torch.cdist(velocities, velocities)
        source_relative, target_relative = torch.where(
            position_distance <= float(distance_threshold_m)
        )
        source = active[source_relative]
        target = active[target_relative]
        edge_indices.append(torch.stack((source, target)))
        edge_features.append(torch.stack((
            position_distance[source_relative, target_relative],
            velocity_distance[source_relative, target_relative],
        ), dim=-1))
    return edge_indices, edge_features


def build_scene_data(
    trajectory_info: dict,
    seed: int,
    range_noise_std_m: float,
    azimuth_noise_std_deg: float,
    distance_threshold_m: float,
) -> Data:
    truth, measurement, valid = noisy_identity_measurements(
        trajectory_info, seed, range_noise_std_m, azimuth_noise_std_deg
    )
    delta_t = float(trajectory_info["delta_T"])
    state = initial_state_from_measurements(measurement, valid, delta_t)
    edge_index, edge_features = input_graph(
        state, valid, distance_threshold_m
    )
    frame_count = state.size(1)
    real_mask = valid.unsqueeze(-1).expand(-1, -1, 4).clone()
    legal_starts = torch.arange(
        0, max(frame_count - int(CONFIG["input_graph_window_len"]), 0),
        dtype=torch.long,
    )
    return Data(
        x=state,
        y=truth,
        measurement=measurement,
        nan_mask=real_mask.clone(),
        real_mask=real_mask,
        edge_index=edge_index,
        edge_features=edge_features,
        steady_window_mask=torch.ones(1, frame_count, dtype=torch.bool),
        switch_window_mask=torch.zeros(1, frame_count, dtype=torch.bool),
        strong_window_mask=torch.zeros(1, frame_count, dtype=torch.bool),
        range_noise_std=torch.tensor([range_noise_std_m], dtype=torch.float32),
        azimuth_noise_std_deg=torch.tensor(
            [azimuth_noise_std_deg], dtype=torch.float32
        ),
        is_v3=torch.tensor([True], dtype=torch.bool),
        trajectory_id=torch.tensor([seed], dtype=torch.long),
        trajectory_index=torch.tensor([seed], dtype=torch.long),
        # Retained for an auditable equivalence to a packed v3 scene.
        legal_window_starts=legal_starts,
    )


def build_runtime(checkpoint: Path, device: torch.device, delta_t: float):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = device
    args.T = float(delta_t)
    for name in CHECKPOINT_ARGUMENTS:
        if name in payload:
            setattr(args, name, payload[name])
    args.counterfactual_training = False
    args.encoder_execution_mode = "stream"
    args.full_time_supervision = True
    args.eval_batch_size = 1
    args.report_trajectory_quantiles = False
    args.finite_check_interval = 0

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
    return payload, args, config, model, trainer


def numeric_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    }


def summarize(runs: list[dict]) -> dict:
    keys = sorted(set.intersection(*(
        set(run["metrics"]) for run in runs
    )))
    result = {}
    for key in keys:
        values = np.asarray([run["metrics"][key] for run in runs], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "valid_runs": len(values),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--monte_carlo_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--range_noise_std_m", type=float, default=3.0)
    parser.add_argument("--azimuth_noise_std_deg", type=float, default=0.1)
    parser.add_argument("--edge_distance_threshold_m", type=float, default=150.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    cli = parser.parse_args()
    if cli.monte_carlo_runs <= 0:
        raise ValueError("monte_carlo_runs must be positive")
    if cli.range_noise_std_m <= 0 or cli.azimuth_noise_std_deg <= 0:
        raise ValueError("measurement standard deviations must be positive")

    trajectory_path = Path(cli.trajectory).resolve()
    checkpoint = Path(cli.checkpoint).resolve()
    trajectory_info = torch.load(
        trajectory_path, map_location="cpu", weights_only=False
    )
    device = torch.device(cli.device)
    payload, args, config, model, trainer = build_runtime(
        checkpoint, device, float(trajectory_info["delta_T"])
    )

    runs = []
    start = time.perf_counter()
    for run_index in range(cli.monte_carlo_runs):
        seed = int(cli.seed + run_index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        scene = build_scene_data(
            trajectory_info,
            seed,
            cli.range_noise_std_m,
            cli.azimuth_noise_std_deg,
            cli.edge_distance_threshold_m,
        )
        run_start = time.perf_counter()
        metrics = trainer.validate(
            DataLoader([scene], batch_size=1, shuffle=False, num_workers=0),
            f"scene02 known-identity run {run_index + 1}/{cli.monte_carlo_runs}",
        )
        runs.append({
            "monte_carlo_index": run_index,
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - run_start,
            "metrics": numeric_metrics(metrics),
        })
    report = {
        "protocol": {
            "version": 1,
            "diagnostic": "known-identity causal online neural/EKF rollout",
            "trajectory": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "frame_count": int(trajectory_info["frame_max"]),
            "target_count": int(trajectory_info["target_num"]),
            "delta_t_seconds": float(trajectory_info["delta_T"]),
            "range_noise_std_m": cli.range_noise_std_m,
            "azimuth_noise_std_deg": cli.azimuth_noise_std_deg,
            "edge_distance_threshold_m": cli.edge_distance_threshold_m,
            "monte_carlo_runs": cli.monte_carlo_runs,
            "seed_start": cli.seed,
            "causal_window_length": int(config["input_graph_window_len"]),
            "measurement_update_each_step": True,
            "covariance_and_innovation_history_rolled_forward": True,
            "known_identity_measurements": True,
            "detection_probability": 1.0,
            "clutter_mean": 0.0,
            "bp_association": False,
            "track_management": False,
            "birth_frame_scored": False,
            "real_mask_semantics": "zero means unborn or permanently dead",
            "device": str(device),
            "total_elapsed_seconds": time.perf_counter() - start,
        },
        "summary": summarize(runs),
        "runs": runs,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    focus = {
        key: report["summary"].get(key)
        for key in (
            "location_rmse_prediction",
            "location_rmse_update",
            "velocity_rmse_prediction",
            "velocity_rmse_update",
            "location_rmse_measurement",
            "measurement_nis_mean",
            "measurement_nis_95_coverage",
        )
    }
    print(json.dumps({
        "output": str(output.resolve()),
        "protocol": report["protocol"],
        "focus": focus,
    }, indent=2))


if __name__ == "__main__":
    main()
