"""Run the frozen paper scene02 protocol through the audited BP tracker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "paper_scene02_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "paper_scene02_20260823"
RUNNER = ROOT / "evaluation" / "run_bp_tracker_robustness_v3.py"
SUMMARIZER = ROOT / "evaluation" / "summarize_paper_scene02_v3.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    trajectory = Path(manifest["trajectory"]["path"])
    if sha256(trajectory) != manifest["trajectory"]["sha256"]:
        raise RuntimeError(f"Trajectory hash mismatch: {trajectory}")
    info = torch.load(trajectory, map_location="cpu", weights_only=False)
    expected = manifest["trajectory"]
    actual = {
        "frame_count": int(info["frame_max"]),
        "target_count": int(info["target_num"]),
        "delta_t_seconds": float(info["delta_T"]),
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise RuntimeError(f"Trajectory metadata mismatch for {key}: {value}")
    for model_id, model in manifest["models"].items():
        checkpoint = Path(model["checkpoint"])
        if sha256(checkpoint) != model["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for {model_id}: {checkpoint}")
    for label, baseline in manifest["frozen_historical_baselines"].items():
        source = Path(baseline["source"])
        if sha256(source) != baseline["sha256"]:
            raise RuntimeError(f"Historical asset hash mismatch for {label}: {source}")
    return manifest


def report_is_valid(path: Path, manifest: dict, model_id: str, runs: int) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report["protocol"]
        expected_model = manifest["models"][model_id]
        measurement = manifest["measurement_and_tracker"]
        fixed_q = manifest["fixed_process_noise"]
        ospa = manifest["ospa"]
        expected_seeds = [manifest["monte_carlo"]["seed_start"] + index for index in range(runs)]
        checks = (
            protocol["checkpoint_sha256"] == expected_model["sha256"],
            protocol["trajectory_sha256"] == manifest["trajectory"]["sha256"],
            protocol["actual_pd"] == measurement["actual_pd"],
            protocol["nominal_pd"] == measurement["nominal_pd"],
            protocol["actual_clutter_mean"] == measurement["actual_clutter_mean"],
            protocol["nominal_clutter_mean"] == measurement["nominal_clutter_mean"],
            protocol["range_noise_std_m"] == measurement["range_noise_std_m"],
            protocol["azimuth_noise_std_deg"] == measurement["azimuth_noise_std_deg"],
            protocol["surveillance_padding_m"] == measurement["surveillance_padding_m"],
            protocol["neural_process_noise_scale"] == fixed_q["acceleration_multiplier"],
            protocol["neural_position_q_variance_per_axis_m2"] == fixed_q["position_variance_per_axis_m2"],
            protocol["monte_carlo_runs"] == runs,
            protocol["seed_start"] == manifest["monte_carlo"]["seed_start"],
            protocol["ospa_cutoff_m"] == ospa["cutoff_m"],
            protocol["ospa_order"] == ospa["order"],
            protocol["evaluated_frames"] == "full_trajectory",
            len(report["runs"]) == runs,
            [run["seed"] for run in report["runs"]] == expected_seeds,
            all(run["frame_count"] == manifest["trajectory"]["frame_count"] for run in report["runs"]),
            all(len(run["per_frame"]) == manifest["trajectory"]["frame_count"] for run in report["runs"]),
        )
        def all_finite(value) -> bool:
            if isinstance(value, dict):
                return all(all_finite(item) for item in value.values())
            if isinstance(value, list):
                return all(all_finite(item) for item in value)
            if isinstance(value, float):
                return math.isfinite(value)
            return True

        return all(checks) and all_finite(report)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_status(output_root: Path, mode: str, states: dict, manifest_path: Path) -> None:
    status = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "mode": mode,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "models": states,
    }
    (output_root / f"{mode}_STATUS.json").write_text(
        json.dumps(status, indent=2, allow_nan=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--models", nargs="+", choices=("P2", "P3", "P4", "P5"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    canonical_models = ["P2", "P4"] if args.mode == "smoke" else ["P2", "P3", "P4", "P5"]
    models = args.models or canonical_models
    run_count = manifest["monte_carlo"][f"{args.mode}_runs"]
    mode_root = args.output.resolve() / args.mode
    mode_root.mkdir(parents=True, exist_ok=True)
    states = {
        model_id: (
            "complete"
            if report_is_valid(mode_root / f"{model_id}.json", manifest, model_id, run_count)
            else "pending"
        )
        for model_id in canonical_models
    }
    write_status(args.output.resolve(), args.mode, states, args.manifest.resolve())

    measurement = manifest["measurement_and_tracker"]
    ospa = manifest["ospa"]
    for model_id in models:
        report_path = mode_root / f"{model_id}.json"
        if report_is_valid(report_path, manifest, model_id, run_count):
            states[model_id] = "complete"
            write_status(args.output.resolve(), args.mode, states, args.manifest.resolve())
            print(f"[{model_id}] reuse valid report: {report_path}", flush=True)
            continue
        states[model_id] = "running"
        write_status(args.output.resolve(), args.mode, states, args.manifest.resolve())
        command = [
            sys.executable,
            str(RUNNER),
            "--checkpoint", manifest["models"][model_id]["checkpoint"],
            "--trajectory", manifest["trajectory"]["path"],
            "--output", str(report_path),
            "--actual_pd", str(measurement["actual_pd"]),
            "--actual_clutter_mean", str(measurement["actual_clutter_mean"]),
            "--nominal_pd", str(measurement["nominal_pd"]),
            "--nominal_clutter_mean", str(measurement["nominal_clutter_mean"]),
            "--monte_carlo", str(run_count),
            "--seed", str(manifest["monte_carlo"]["seed_start"]),
            "--device", args.device,
            "--range_noise_std_m", str(measurement["range_noise_std_m"]),
            "--azimuth_noise_std_deg", str(measurement["azimuth_noise_std_deg"]),
            "--surveillance_padding_m", str(measurement["surveillance_padding_m"]),
            "--ospa_cutoff_m", str(ospa["cutoff_m"]),
            "--ospa_order", str(ospa["order"]),
            "--neural_process_noise_scale", str(
                manifest["fixed_process_noise"]["acceleration_multiplier"]
            ),
            "--neural_position_q_variance_m2", str(
                manifest["fixed_process_noise"]["position_variance_per_axis_m2"]
            ),
        ]
        if model_id == "P4":
            plot_root = args.output.resolve() / "plots"
            plot_root.mkdir(parents=True, exist_ok=True)
            command.extend([
                "--plot_output", str(plot_root / f"{args.mode}_P4_tracking_mc0.png"),
                "--truth_plot_output", str(plot_root / "scene02_ground_truth.png"),
                "--plot_mc_index", "0",
            ])
        print(f"[{model_id}] start {args.mode} run ({run_count} MC)", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError:
            states[model_id] = "failed"
            write_status(args.output.resolve(), args.mode, states, args.manifest.resolve())
            raise
        if not report_is_valid(report_path, manifest, model_id, run_count):
            states[model_id] = "invalid_report"
            write_status(args.output.resolve(), args.mode, states, args.manifest.resolve())
            raise RuntimeError(f"Invalid completed report for {model_id}: {report_path}")
        states[model_id] = "complete"
        write_status(args.output.resolve(), args.mode, states, args.manifest.resolve())
        print(f"[{model_id}] complete: {report_path}", flush=True)

    if args.mode == "formal" and all(
        report_is_valid(mode_root / f"{model_id}.json", manifest, model_id, run_count)
        for model_id in canonical_models
    ):
        subprocess.run(
            [sys.executable, str(SUMMARIZER), "--manifest", str(args.manifest.resolve()), "--results", str(args.output.resolve())],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
