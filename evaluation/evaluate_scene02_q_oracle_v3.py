"""Local position-Q oracle and causal-NIS threshold diagnostic on scene02.

The online trajectory is rolled out with acceleration Q=8 and no direct
position Q.  At each scored time, candidate direct position variances are
re-evaluated from the same prior, current measurement and covariance.  Truth
is used only to score the non-deployable oracle/policy labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_scene02_known_identity_v3 import (
    build_runtime,
    build_scene_data,
    sha256_file,
)


class OracleCollector:
    def __init__(self, model, config, candidates):
        self.model = model
        self.config = config
        self.candidates = tuple(float(value) for value in candidates)
        self.surprise = []
        self.position_square = {value: [] for value in self.candidates}
        self.velocity_square = {value: [] for value in self.candidates}

    def __call__(self, **values):
        outputs = values["outputs"]
        window = values["window"]
        valid = outputs.real_mask[:, 0].bool()
        if not valid.any():
            return
        prediction = outputs.states[:, 0, 0]
        target = outputs.target[:, 0]
        covariance = outputs.covariances[:, 0, 0]
        innovation = outputs.measurement_innovation[:, 0]
        scale = prediction.new_tensor([
            self.config["S_POS"], self.config["S_POS"],
            self.config["S_VEL"], self.config["S_VEL"],
        ])
        h_bar = prediction.new_zeros(2, 4)
        h_bar[0, 0] = h_bar[1, 1] = float(self.config["S_POS"])
        base_state_s = h_bar.unsqueeze(0) @ covariance @ h_bar.T.unsqueeze(0)
        measurement_r = outputs.innovation_covariance - base_state_s
        measurement_r = 0.5 * (
            measurement_r + measurement_r.transpose(-1, -2)
        )
        eye2 = torch.eye(2, device=prediction.device, dtype=prediction.dtype)
        measurement_r = measurement_r + 1e-6 * eye2

        residual_normalized = window.residual / float(self.config["S_RES"])
        bundle = self.model._innovation_bundle(window, residual_normalized)
        causal = self.model.decoder._causal_nis_statistics(
            bundle, residual_normalized
        )
        self.surprise.extend(causal[valid, 0].detach().cpu().tolist())

        for physical_variance in self.candidates:
            candidate_covariance = covariance.clone()
            normalized_variance = (
                physical_variance / float(self.config["S_POS"]) ** 2
            )
            candidate_covariance[:, 0, 0] += normalized_variance
            candidate_covariance[:, 1, 1] += normalized_variance
            innovation_s = (
                h_bar.unsqueeze(0) @ candidate_covariance
                @ h_bar.T.unsqueeze(0) + measurement_r
            )
            pht = candidate_covariance @ h_bar.T
            gain = torch.linalg.solve(
                innovation_s, pht.transpose(1, 2)
            ).transpose(1, 2)
            update = prediction + (
                gain @ innovation.unsqueeze(-1)
            ).squeeze(-1)
            error = (update - target) * scale
            self.position_square[physical_variance].extend(
                error[valid, :2].square().sum(-1).detach().cpu().tolist()
            )
            self.velocity_square[physical_variance].extend(
                error[valid, 2:].square().sum(-1).detach().cpu().tolist()
            )

    def result(self):
        return {
            "surprise": np.asarray(self.surprise, dtype=np.float64),
            "position_square": {
                value: np.asarray(entries, dtype=np.float64)
                for value, entries in self.position_square.items()
            },
            "velocity_square": {
                value: np.asarray(entries, dtype=np.float64)
                for value, entries in self.velocity_square.items()
            },
        }


def summarize(position_square, velocity_square):
    count = int(len(position_square))
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_square))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_square))),
        "valid_node_steps": count,
    }


def concatenate(runs, key, candidate=None):
    if candidate is None:
        return np.concatenate([run[key] for run in runs])
    return np.concatenate([run[key][candidate] for run in runs])


def threshold_policy(calibration_runs, evaluation_runs, candidates):
    calibration_surprise = concatenate(calibration_runs, "surprise")
    quantiles = np.linspace(0.05, 0.95, 37)
    thresholds = np.unique(np.quantile(calibration_surprise, quantiles))
    best = None
    for low in candidates:
        for high in candidates:
            if high < low:
                continue
            low_error = concatenate(calibration_runs, "position_square", low)
            high_error = concatenate(calibration_runs, "position_square", high)
            for threshold in thresholds:
                selected = np.where(
                    calibration_surprise >= threshold, high_error, low_error
                )
                score = float(selected.mean())
                if best is None or score < best[0]:
                    best = (score, float(threshold), float(low), float(high))
    _, threshold, low, high = best
    evaluation_surprise = concatenate(evaluation_runs, "surprise")
    high_mask = evaluation_surprise >= threshold
    position = np.where(
        high_mask,
        concatenate(evaluation_runs, "position_square", high),
        concatenate(evaluation_runs, "position_square", low),
    )
    velocity = np.where(
        high_mask,
        concatenate(evaluation_runs, "velocity_square", high),
        concatenate(evaluation_runs, "velocity_square", low),
    )
    result = summarize(position, velocity)
    result.update({
        "calibrated_low_variance_m2": low,
        "calibrated_high_variance_m2": high,
        "calibrated_log1p_mean_nis_threshold": threshold,
        "high_selection_fraction": float(high_mask.mean()),
        "calibration_runs": len(calibration_runs),
        "heldout_runs": len(evaluation_runs),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", type=float, nargs="+", default=[0, 1, 4, 16])
    parser.add_argument("--monte_carlo_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--range_noise_std_m", type=float, default=3.0)
    parser.add_argument("--azimuth_noise_std_deg", type=float, default=0.1)
    parser.add_argument("--edge_distance_threshold_m", type=float, default=150.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    candidates = tuple(sorted(set(float(value) for value in args.candidates)))
    if len(candidates) < 2 or any(value < 0.0 for value in candidates):
        raise ValueError("provide at least two non-negative candidates")
    trajectory_path = Path(args.trajectory).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    trajectory_info = torch.load(
        trajectory_path, map_location="cpu", weights_only=False
    )
    payload, _, config, model, trainer = build_runtime(
        checkpoint_path, torch.device(args.device),
        float(trajectory_info["delta_T"]),
    )
    model.process_noise_mode = "scaled"
    model.process_noise_scale = 8.0
    model.decoder.q_position_variance_override_physical = 0.0
    runs = []
    for run_index in range(args.monte_carlo_runs):
        seed = int(args.seed + run_index)
        scene = build_scene_data(
            trajectory_info, seed, args.range_noise_std_m,
            args.azimuth_noise_std_deg, args.edge_distance_threshold_m,
        )
        collector = OracleCollector(model, config, candidates)
        trainer.step_observer = collector
        trainer.validate(
            DataLoader([scene], batch_size=1, shuffle=False, num_workers=0),
            f"scene02 position-Q oracle {run_index + 1}/{args.monte_carlo_runs}",
        )
        trainer.step_observer = None
        runs.append(collector.result())

    constants = {}
    for candidate in candidates:
        constants[str(candidate)] = summarize(
            concatenate(runs, "position_square", candidate),
            concatenate(runs, "velocity_square", candidate),
        )
    position_matrix = np.stack([
        concatenate(runs, "position_square", candidate)
        for candidate in candidates
    ], axis=1)
    velocity_matrix = np.stack([
        concatenate(runs, "velocity_square", candidate)
        for candidate in candidates
    ], axis=1)
    oracle_index = position_matrix.argmin(axis=1)
    row = np.arange(position_matrix.shape[0])
    oracle = summarize(
        position_matrix[row, oracle_index], velocity_matrix[row, oracle_index]
    )
    oracle["selection_fraction"] = {
        str(candidate): float(np.mean(oracle_index == index))
        for index, candidate in enumerate(candidates)
    }
    split = max(1, args.monte_carlo_runs // 2)
    heldout_constants = None
    if args.monte_carlo_runs >= 2:
        heldout_constants = {
            str(candidate): summarize(
                concatenate(runs[split:], "position_square", candidate),
                concatenate(runs[split:], "velocity_square", candidate),
            )
            for candidate in candidates
        }
    policy = (
        threshold_policy(runs[:split], runs[split:], candidates)
        if args.monte_carlo_runs >= 2 else None
    )
    report = {
        "protocol": {
            "version": 1,
            "diagnostic": "local direct-position-Q oracle from a fixed Q=8 rollout",
            "truth_used_only_for_candidate_scoring": True,
            "policy_feature": "strictly lagged log1p(mean NIS)",
            "trajectory": str(trajectory_path),
            "trajectory_sha256": sha256_file(trajectory_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "acceleration_q_scale": 8.0,
            "candidate_position_variances_per_axis_m2": list(candidates),
            "monte_carlo_runs": args.monte_carlo_runs,
            "seed_start": args.seed,
            "known_identity": True,
            "causal_online_rollout": True,
        },
        "constant_candidates": constants,
        "heldout_constant_candidates": heldout_constants,
        "oracle": oracle,
        "heldout_threshold_policy": policy,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
