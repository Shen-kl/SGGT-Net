"""Run the frozen paired P4/P5 association-context robustness experiment."""

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
RUNNER = ROOT / "evaluation" / "run_bp_tracker_robustness_v3.py"
DEFAULT_PROTOCOL = ROOT / "results" / "association_context_decision_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "association_context_decision_20260903"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_report(protocol: dict, model_id: str, condition: dict, runs: int) -> dict:
    measurement = protocol["measurement_and_tracker"]
    fixed_q = protocol["fixed_process_noise"]
    return {
        "checkpoint_sha256": protocol["models"][model_id]["sha256"],
        "trajectory_sha256": protocol["trajectory"]["sha256"],
        "actual_pd": condition["actual_pd"],
        "actual_clutter_mean": condition["actual_clutter_mean"],
        "nominal_pd": measurement["nominal_pd"],
        "nominal_clutter_mean": measurement["nominal_clutter_mean"],
        "range_noise_std_m": measurement["range_noise_std_m"],
        "azimuth_noise_std_deg": measurement["azimuth_noise_std_deg"],
        "surveillance_padding_m": measurement["surveillance_padding_m"],
        "neural_process_noise_scale": fixed_q["acceleration_multiplier"],
        "neural_position_q_variance_per_axis_m2": fixed_q["position_variance_per_axis_m2"],
        "monte_carlo_runs": runs,
        "seed_start": protocol["monte_carlo"]["seed_start"],
        "ospa_cutoff_m": protocol["ospa"]["cutoff_m"],
        "ospa_order": protocol["ospa"]["order"],
        "evaluated_frames": "full_trajectory",
    }


def report_is_valid(
    path: Path, protocol: dict, model_id: str, condition: dict, runs: int
) -> bool:
    if not path.exists():
        return False
    try:
        report = read_json(path)
        actual = report["protocol"]
        if any(actual.get(key) != value for key, value in expected_report(
            protocol, model_id, condition, runs
        ).items()):
            return False
        expected_seeds = [
            protocol["monte_carlo"]["seed_start"] + index for index in range(runs)
        ]
        if [run["seed"] for run in report["runs"]] != expected_seeds:
            return False
        frame_count = protocol["trajectory"]["frame_count"]
        if any(
            run["frame_count"] != frame_count or len(run["per_frame"]) != frame_count
            for run in report["runs"]
        ):
            return False

        def finite(value) -> bool:
            if isinstance(value, dict):
                return all(finite(item) for item in value.values())
            if isinstance(value, list):
                return all(finite(item) for item in value)
            return not isinstance(value, float) or math.isfinite(value)

        return finite(report)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def source_for(
    protocol: dict, output: Path, mode: str, model_id: str, condition: dict
) -> tuple[Path, str]:
    reuse = protocol.get("formal_reused_reports", {}).get(
        f"{model_id}:{condition['id']}"
    )
    if mode == "formal" and reuse:
        return ROOT / reuse["path"], "reused"
    return output / mode / "raw" / model_id / f"{condition['id']}.json", "generated"


def write_status(
    output: Path, protocol_path: Path, mode: str, states: dict
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "mode": mode,
        "cells": states,
    }
    (output / f"STATUS_{mode}.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--models", nargs="+", choices=("P4", "P5"))
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    output = args.output.resolve()
    models = args.models or ["P4", "P5"]
    conditions = protocol["conditions"]
    if args.conditions:
        requested = set(args.conditions)
        known = {condition["id"] for condition in conditions}
        if requested - known:
            raise ValueError(f"unknown conditions: {sorted(requested - known)}")
        conditions = [c for c in conditions if c["id"] in requested]
    runs = protocol["monte_carlo"][f"{args.mode}_runs"]

    for path, expected_hash in (
        (Path(protocol["trajectory"]["path"]), protocol["trajectory"]["sha256"]),
        *[
            (Path(model["checkpoint"]), model["sha256"])
            for model in protocol["models"].values()
        ],
    ):
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"missing or hash-mismatched frozen input: {path}")

    states = {}
    for model_id in ("P4", "P5"):
        for condition in protocol["conditions"]:
            cell = f"{model_id}:{condition['id']}"
            source, origin = source_for(protocol, output, args.mode, model_id, condition)
            states[cell] = (
                origin if report_is_valid(source, protocol, model_id, condition, runs)
                else "pending"
            )
    write_status(output, protocol_path, args.mode, states)

    measurement = protocol["measurement_and_tracker"]
    fixed_q = protocol["fixed_process_noise"]
    for model_id in models:
        for condition in conditions:
            cell = f"{model_id}:{condition['id']}"
            report_path, origin = source_for(
                protocol, output, args.mode, model_id, condition
            )
            if report_is_valid(report_path, protocol, model_id, condition, runs):
                states[cell] = origin
                write_status(output, protocol_path, args.mode, states)
                print(f"[{cell}] {origin} valid report", flush=True)
                continue
            if origin == "reused":
                raise RuntimeError(f"invalid frozen reused report: {report_path}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            log_root = output / args.mode / "logs"
            log_root.mkdir(parents=True, exist_ok=True)
            states[cell] = "running"
            write_status(output, protocol_path, args.mode, states)
            command = [
                sys.executable,
                str(RUNNER),
                "--checkpoint", protocol["models"][model_id]["checkpoint"],
                "--trajectory", protocol["trajectory"]["path"],
                "--output", str(report_path),
                "--actual_pd", str(condition["actual_pd"]),
                "--actual_clutter_mean", str(condition["actual_clutter_mean"]),
                "--nominal_pd", str(measurement["nominal_pd"]),
                "--nominal_clutter_mean", str(measurement["nominal_clutter_mean"]),
                "--monte_carlo", str(runs),
                "--seed", str(protocol["monte_carlo"]["seed_start"]),
                "--device", args.device,
                "--range_noise_std_m", str(measurement["range_noise_std_m"]),
                "--azimuth_noise_std_deg", str(measurement["azimuth_noise_std_deg"]),
                "--surveillance_padding_m", str(measurement["surveillance_padding_m"]),
                "--ospa_cutoff_m", str(protocol["ospa"]["cutoff_m"]),
                "--ospa_order", str(protocol["ospa"]["order"]),
                "--neural_process_noise_scale", str(fixed_q["acceleration_multiplier"]),
                "--neural_position_q_variance_m2", str(fixed_q["position_variance_per_axis_m2"]),
            ]
            stdout_path = log_root / f"{model_id}_{condition['id']}.stdout.log"
            stderr_path = log_root / f"{model_id}_{condition['id']}.stderr.log"
            print(f"[{cell}] start {runs} paired MC runs", flush=True)
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
                        stderr_path.open("w", encoding="utf-8") as stderr_handle:
                    subprocess.run(
                        command, cwd=ROOT, check=True,
                        stdout=stdout_handle, stderr=stderr_handle,
                    )
            except subprocess.CalledProcessError:
                states[cell] = "failed"
                write_status(output, protocol_path, args.mode, states)
                raise
            if not report_is_valid(
                report_path, protocol, model_id, condition, runs
            ):
                states[cell] = "invalid_report"
                write_status(output, protocol_path, args.mode, states)
                raise RuntimeError(f"invalid completed report: {report_path}")
            states[cell] = "complete"
            write_status(output, protocol_path, args.mode, states)
            print(f"[{cell}] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
