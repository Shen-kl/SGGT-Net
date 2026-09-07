"""Run the frozen P4 scene02 detection/clutter robustness protocol."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run_paper_scene02_protocol_v3 import load_manifest, sha256


DEFAULT_MANIFEST = ROOT / "results" / "paper_scene02_fixed_q_protocol_v2.json"
DEFAULT_OUTPUT = ROOT / "results" / "paper_scene02_fixed_q_20260824"
RUNNER = ROOT / "evaluation" / "run_bp_tracker_robustness_v3.py"


def condition_report_is_valid(
    path: Path, manifest: dict, condition: dict, runs: int
) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report["protocol"]
        model = manifest["models"][manifest["robustness"]["model"]]
        measurement = manifest["measurement_and_tracker"]
        fixed_q = manifest["fixed_process_noise"]
        ospa = manifest["ospa"]
        expected_seeds = [
            manifest["monte_carlo"]["seed_start"] + index for index in range(runs)
        ]
        checks = (
            protocol["checkpoint_sha256"] == model["sha256"],
            protocol["trajectory_sha256"] == manifest["trajectory"]["sha256"],
            protocol["actual_pd"] == condition["actual_pd"],
            protocol["actual_clutter_mean"] == condition["actual_clutter_mean"],
            protocol["nominal_pd"] == manifest["robustness"]["nominal_pd"],
            protocol["nominal_clutter_mean"]
            == manifest["robustness"]["nominal_clutter_mean"],
            protocol["range_noise_std_m"] == measurement["range_noise_std_m"],
            protocol["azimuth_noise_std_deg"]
            == measurement["azimuth_noise_std_deg"],
            protocol["surveillance_padding_m"]
            == measurement["surveillance_padding_m"],
            protocol["neural_process_noise_scale"]
            == fixed_q["acceleration_multiplier"],
            protocol["neural_position_q_variance_per_axis_m2"]
            == fixed_q["position_variance_per_axis_m2"],
            protocol["monte_carlo_runs"] == runs,
            protocol["seed_start"] == manifest["monte_carlo"]["seed_start"],
            protocol["ospa_cutoff_m"] == ospa["cutoff_m"],
            protocol["ospa_order"] == ospa["order"],
            protocol["evaluated_frames"] == "full_trajectory",
            len(report["runs"]) == runs,
            [run["seed"] for run in report["runs"]] == expected_seeds,
            all(
                run["frame_count"] == manifest["trajectory"]["frame_count"]
                for run in report["runs"]
            ),
            all(
                len(run["per_frame"]) == manifest["trajectory"]["frame_count"]
                for run in report["runs"]
            ),
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


def write_status(output: Path, mode: str, states: dict, manifest_path: Path) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "mode": mode,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "conditions": states,
    }
    (output / f"{mode}_robustness_STATUS.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Optional condition IDs. The base condition is reused from the main P4 report.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    output = args.output.resolve()
    mode_root = output / args.mode
    robustness_root = mode_root / "robustness"
    log_root = output / "logs"
    robustness_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runs = manifest["monte_carlo"][f"{args.mode}_runs"]
    conditions = manifest["robustness"]["conditions"]
    if args.conditions:
        requested = set(args.conditions)
        known = {condition["id"] for condition in conditions}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown robustness conditions: {sorted(unknown)}")
        conditions = [condition for condition in conditions if condition["id"] in requested]

    states = {}
    for condition in manifest["robustness"]["conditions"]:
        if condition.get("reuse_main_report"):
            main_report = mode_root / "P4.json"
            states[condition["id"]] = "reused_main" if main_report.exists() else "waiting_main"
        else:
            report_path = robustness_root / f"{condition['id']}.json"
            states[condition["id"]] = (
                "complete"
                if condition_report_is_valid(report_path, manifest, condition, runs)
                else "pending"
            )
    write_status(output, args.mode, states, manifest_path)

    model_id = manifest["robustness"]["model"]
    model = manifest["models"][model_id]
    measurement = manifest["measurement_and_tracker"]
    fixed_q = manifest["fixed_process_noise"]
    ospa = manifest["ospa"]
    for condition in conditions:
        condition_id = condition["id"]
        if condition.get("reuse_main_report"):
            main_report = mode_root / f"{model_id}.json"
            if not main_report.exists():
                raise RuntimeError(f"Main {model_id} report is required: {main_report}")
            states[condition_id] = "reused_main"
            write_status(output, args.mode, states, manifest_path)
            continue
        report_path = robustness_root / f"{condition_id}.json"
        if condition_report_is_valid(report_path, manifest, condition, runs):
            states[condition_id] = "complete"
            write_status(output, args.mode, states, manifest_path)
            print(f"[{condition_id}] reuse valid report: {report_path}", flush=True)
            continue

        states[condition_id] = "running"
        write_status(output, args.mode, states, manifest_path)
        command = [
            sys.executable,
            str(RUNNER),
            "--checkpoint", model["checkpoint"],
            "--trajectory", manifest["trajectory"]["path"],
            "--output", str(report_path),
            "--actual_pd", str(condition["actual_pd"]),
            "--actual_clutter_mean", str(condition["actual_clutter_mean"]),
            "--nominal_pd", str(manifest["robustness"]["nominal_pd"]),
            "--nominal_clutter_mean", str(manifest["robustness"]["nominal_clutter_mean"]),
            "--monte_carlo", str(runs),
            "--seed", str(manifest["monte_carlo"]["seed_start"]),
            "--device", args.device,
            "--range_noise_std_m", str(measurement["range_noise_std_m"]),
            "--azimuth_noise_std_deg", str(measurement["azimuth_noise_std_deg"]),
            "--surveillance_padding_m", str(measurement["surveillance_padding_m"]),
            "--ospa_cutoff_m", str(ospa["cutoff_m"]),
            "--ospa_order", str(ospa["order"]),
            "--neural_process_noise_scale", str(fixed_q["acceleration_multiplier"]),
            "--neural_position_q_variance_m2", str(
                fixed_q["position_variance_per_axis_m2"]
            ),
        ]
        stdout_path = log_root / f"{args.mode}_{condition_id}.stdout.log"
        stderr_path = log_root / f"{args.mode}_{condition_id}.stderr.log"
        print(f"[{condition_id}] start {args.mode} run ({runs} MC)", flush=True)
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
        except subprocess.CalledProcessError:
            states[condition_id] = "failed"
            write_status(output, args.mode, states, manifest_path)
            raise
        if not condition_report_is_valid(report_path, manifest, condition, runs):
            states[condition_id] = "invalid_report"
            write_status(output, args.mode, states, manifest_path)
            raise RuntimeError(f"Invalid completed report: {report_path}")
        states[condition_id] = "complete"
        write_status(output, args.mode, states, manifest_path)
        print(f"[{condition_id}] complete: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
