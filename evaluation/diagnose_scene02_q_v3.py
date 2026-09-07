"""Diagnose and counterfactually scale learned process noise on scene02.

The rollout uses the known-identity protocol from
``evaluate_scene02_known_identity_v3.py`` so changes in Q cannot alter BP
association or track lifecycle.  Model weights and measurements are shared
across every Q scale.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_scene02_known_identity_v3 import (
    build_runtime,
    build_scene_data,
    numeric_metrics,
    sha256_file,
)


class DiagnosticAccumulator:
    def __init__(self, config: dict):
        self.config = config
        self.values = defaultdict(list)
        self.subsets = defaultdict(lambda: defaultdict(list))

    @staticmethod
    def _append(destination, key, tensor, valid):
        selected = torch.as_tensor(tensor).detach().reshape(-1)[valid]
        selected = selected[torch.isfinite(selected)]
        if selected.numel():
            destination[key].extend(selected.cpu().tolist())

    def __call__(self, **values):
        outputs = values["outputs"]
        window = values["window"]
        valid = outputs.real_mask[:, 0].bool()
        if not valid.any():
            return
        scale = outputs.states.new_tensor([
            self.config["S_POS"], self.config["S_POS"],
            self.config["S_VEL"], self.config["S_VEL"],
        ])
        prediction = outputs.states[:, 0, 0]
        update = outputs.updated_state[:, 0]
        truth = outputs.target[:, 0]
        measurement = window.measurement_future[:, 0]
        prediction_error = (prediction - truth) * scale
        update_error = (update - truth) * scale
        # The trainer observer exposes the raw causal window while ``outputs``
        # comes from its normalized clone.  Compare raw measurement and raw
        # target directly; mixing the two coordinate systems is invalid.
        measurement_error = measurement - window.y[:, 0][:, [0, 2]]
        position_prediction = prediction_error[:, :2].norm(dim=-1)
        position_update = update_error[:, :2].norm(dim=-1)
        position_measurement = measurement_error.norm(dim=-1)
        velocity_prediction = prediction_error[:, 2:].norm(dim=-1)
        velocity_update = update_error[:, 2:].norm(dim=-1)
        truth_speed = (truth[:, 2:] * float(self.config["S_VEL"])).norm(dim=-1)

        covariance_prediction = outputs.covariances[:, 0, 0]
        covariance_update = outputs.updated_covariance
        transform = torch.diag(scale)
        covariance_prediction_physical = (
            transform.unsqueeze(0) @ covariance_prediction
            @ transform.unsqueeze(0)
        )
        covariance_update_physical = (
            transform.unsqueeze(0) @ covariance_update @ transform.unsqueeze(0)
        )
        eye = torch.eye(4, device=update.device, dtype=update.dtype).unsqueeze(0)
        state_nis = (
            update_error.unsqueeze(1)
            @ torch.linalg.solve(
                covariance_update_physical + 1e-6 * eye,
                update_error.unsqueeze(-1),
            )
        ).flatten()
        measurement_nis = outputs.nis[:, 0, 0]

        observation = covariance_prediction.new_zeros(2, 4)
        observation[0, 0] = observation[1, 1] = 1.0
        observation_normalized = observation @ transform
        gain_normalized = torch.linalg.solve(
            outputs.innovation_covariance,
            (covariance_prediction @ observation_normalized.T).transpose(1, 2),
        ).transpose(1, 2)
        # The EKF innovation is already in metres. K maps physical innovation
        # to normalized state; left multiplication by D gives the physical K.
        gain_physical = scale.view(1, 4, 1) * gain_normalized
        position_gain = torch.linalg.matrix_norm(
            gain_physical[:, :2], dim=(-2, -1)
        )
        velocity_gain = torch.linalg.matrix_norm(
            gain_physical[:, 2:], dim=(-2, -1)
        )

        model = values.get("model")
        if model is None:
            # The observer is wrapped with the model below; this branch keeps
            # the class independently usable in tests.
            q_state = covariance_prediction.new_zeros(
                covariance_prediction.size(0), 1, 4, 4
            )
        else:
            q_state = model.last_state_process_noise
        q_raw = (
            model.last_process_noise_used
            if model is not None else covariance_prediction.new_zeros(
                covariance_prediction.size(0), 1, 2, 2
            )
        )
        q_physical = (
            scale.view(1, 1, 4, 1) * q_state
            * scale.view(1, 1, 1, 4)
        )
        q_position_trace = q_physical[:, 0, :2, :2].diagonal(
            dim1=-2, dim2=-1
        ).sum(-1)
        q_velocity_trace = q_physical[:, 0, 2:, 2:].diagonal(
            dim1=-2, dim2=-1
        ).sum(-1)
        extra_q = outputs.q_position_noise[:, 0, 0]
        extra_position_trace = (
            extra_q[:, :2, :2].diagonal(dim1=-2, dim2=-1).sum(-1)
            * float(self.config["S_POS"]) ** 2
        )
        metrics = {
            "position_prediction_error_m": position_prediction,
            "position_update_error_m": position_update,
            "position_measurement_error_m": position_measurement,
            "velocity_prediction_error_mps": velocity_prediction,
            "velocity_update_error_mps": velocity_update,
            "truth_speed_mps": truth_speed,
            "measurement_nis": measurement_nis,
            "measurement_nis_below_95": (measurement_nis <= 5.991464547).float(),
            "state_nis": state_nis,
            "state_nis_below_95": (state_nis <= 9.487729037).float(),
            "position_gain_frobenius": position_gain,
            "velocity_gain_frobenius": velocity_gain,
            "prediction_position_cov_trace_m2": (
                covariance_prediction_physical[:, :2, :2]
                .diagonal(dim1=-2, dim2=-1).sum(-1)
            ),
            "update_position_cov_trace_m2": (
                covariance_update_physical[:, :2, :2]
                .diagonal(dim1=-2, dim2=-1).sum(-1)
            ),
            "state_q_position_trace_m2": q_position_trace,
            "state_q_velocity_trace_m2ps2": q_velocity_trace,
            "raw_acceleration_q_trace": q_raw[:, 0].diagonal(
                dim1=-2, dim2=-1
            ).sum(-1),
            "extra_q_position_trace_m2": extra_position_trace,
            "q_acceleration_multiplier": outputs.q_acceleration_multiplier[:, 0, 0],
            "q_innovation_score": outputs.q_innovation_score[:, 0, 0],
            "update_worse_than_measurement": (
                position_update > position_measurement
            ).float(),
            "update_better_than_prediction": (
                position_update < position_prediction
            ).float(),
        }
        high_speed = truth_speed > 34.95
        for key, tensor in metrics.items():
            self._append(self.values, key, tensor, valid)
            self._append(self.subsets["normal_speed"], key, tensor, valid & ~high_speed)
            self._append(self.subsets["above_train_max_speed"], key, tensor, valid & high_speed)

    @staticmethod
    def _summary(values: dict) -> dict:
        result = {}
        for key, entries in values.items():
            if not entries:
                continue
            array = np.asarray(entries, dtype=float)
            result[key] = {
                "mean": float(array.mean()),
                "rms": float(np.sqrt(np.mean(array ** 2))),
                "std": float(array.std()),
                "p50": float(np.percentile(array, 50)),
                "p95": float(np.percentile(array, 95)),
                "p99": float(np.percentile(array, 99)),
                "max": float(array.max()),
                "count": int(array.size),
            }
        for left, right, name in (
            ("q_innovation_score", "state_q_position_trace_m2", "corr_surprise_q_position"),
            ("truth_speed_mps", "state_q_position_trace_m2", "corr_speed_q_position"),
            ("measurement_nis", "state_q_position_trace_m2", "corr_nis_q_position"),
        ):
            if left in values and right in values and len(values[left]) == len(values[right]):
                x, y = np.asarray(values[left]), np.asarray(values[right])
                result[name] = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else 0.0
        return result

    def result(self) -> dict:
        return {
            "global": self._summary(self.values),
            "speed_subsets": {
                name: self._summary(values)
                for name, values in self.subsets.items()
            },
        }


def run_scale(
    model,
    trainer,
    trajectory_info,
    seeds,
    scale,
    position_variance,
    range_noise_std_m,
    azimuth_noise_std_deg,
    edge_threshold,
):
    model.process_noise_mode = "zero" if scale == 0.0 else "scaled"
    model.process_noise_scale = float(scale)
    model.decoder.q_position_variance_override_physical = (
        None if position_variance is None else float(position_variance)
    )
    runs = []
    merged = DiagnosticAccumulator(trainer.config)
    for run_index, seed in enumerate(seeds):
        scene = build_scene_data(
            trajectory_info,
            seed,
            range_noise_std_m,
            azimuth_noise_std_deg,
            edge_threshold,
        )
        accumulator = DiagnosticAccumulator(trainer.config)

        def observer(**values):
            values["model"] = model
            accumulator(**values)
            merged(**values)

        trainer.step_observer = observer
        metrics = trainer.validate(
            DataLoader([scene], batch_size=1, shuffle=False, num_workers=0),
            (
                f"scene02 Q scale={scale:g}, position-Q="
                f"{'learned' if position_variance is None else f'{position_variance:g} m^2'}, "
                f"run {run_index + 1}/{len(seeds)}"
            ),
        )
        trainer.step_observer = None
        runs.append({
            "monte_carlo_index": run_index,
            "seed": seed,
            "metrics": numeric_metrics(metrics),
            "diagnostics": accumulator.result(),
        })
    return {
        "process_noise_mode": model.process_noise_mode,
        "process_noise_scale": float(scale),
        "direct_position_variance_per_axis_m2": (
            None if position_variance is None else float(position_variance)
        ),
        "summary": merged.result(),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 4])
    parser.add_argument(
        "--position_variances", type=float, nargs="+", default=[0.0],
        help="fixed direct position-Q variance per axis in m^2",
    )
    parser.add_argument(
        "--include_learned_position", action="store_true",
        help="also evaluate the checkpoint's learned direct position-Q output",
    )
    parser.add_argument("--monte_carlo_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--range_noise_std_m", type=float, default=3.0)
    parser.add_argument("--azimuth_noise_std_deg", type=float, default=0.1)
    parser.add_argument("--edge_distance_threshold_m", type=float, default=150.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()
    if cli.monte_carlo_runs <= 0 or not cli.scales or not cli.position_variances:
        raise ValueError(
            "positive monte_carlo_runs and at least one Q scale/position variance are required"
        )
    if any(scale < 0 for scale in cli.scales) or any(
        variance < 0 for variance in cli.position_variances
    ):
        raise ValueError("Q scales and position variances must be non-negative")

    trajectory_path = Path(cli.trajectory).resolve()
    checkpoint_path = Path(cli.checkpoint).resolve()
    trajectory_info = torch.load(
        trajectory_path, map_location="cpu", weights_only=False
    )
    payload, args, config, model, trainer = build_runtime(
        checkpoint_path, torch.device(cli.device), float(trajectory_info["delta_T"])
    )
    seeds = [int(cli.seed + index) for index in range(cli.monte_carlo_runs)]
    start = time.perf_counter()
    position_variances = list(cli.position_variances)
    if cli.include_learned_position:
        position_variances.insert(0, None)
    variants = [
        run_scale(
            model, trainer, trajectory_info, seeds, scale, position_variance,
            cli.range_noise_std_m, cli.azimuth_noise_std_deg,
            cli.edge_distance_threshold_m,
        )
        for scale in cli.scales
        for position_variance in position_variances
    ]
    report = {
        "protocol": {
            "version": 1,
            "diagnostic": "known-identity learned-Q counterfactual scale scan",
            "trajectory": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "monte_carlo_runs": cli.monte_carlo_runs,
            "seed_start": cli.seed,
            "shared_measurements_across_scales": True,
            "range_noise_std_m": cli.range_noise_std_m,
            "azimuth_noise_std_deg": cli.azimuth_noise_std_deg,
            "learned_acceleration_q_scaled": True,
            "direct_position_q_counterfactual": True,
            "bp_association": False,
            "track_management": False,
            "total_elapsed_seconds": time.perf_counter() - start,
        },
        "variants": variants,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    focus = []
    for variant in variants:
        global_result = variant["summary"]["global"]
        focus.append({
            "scale": variant["process_noise_scale"],
            "position_variance_per_axis_m2": variant[
                "direct_position_variance_per_axis_m2"
            ],
            "position_update_error_m": global_result.get("position_update_error_m"),
            "position_gain_frobenius": global_result.get("position_gain_frobenius"),
            "measurement_nis": global_result.get("measurement_nis"),
            "state_q_position_trace_m2": global_result.get("state_q_position_trace_m2"),
            "extra_q_position_trace_m2": global_result.get("extra_q_position_trace_m2"),
        })
    print(json.dumps({"output": str(output.resolve()), "focus": focus}, indent=2))


if __name__ == "__main__":
    main()
