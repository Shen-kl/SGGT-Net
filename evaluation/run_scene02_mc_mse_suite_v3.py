"""Run missing scene02 conditions and retain aggregate-only MC-MSE artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.run_paper_scene02_protocol_v3 import load_manifest, report_is_valid, sha256
from evaluation.run_scene02_fixed_q_robustness_v3 import condition_report_is_valid
from evaluation.summarize_scene02_mc_mse_suite_v3 import write_artifact_manifest


DEFAULT_MANIFEST = ROOT / "results" / "paper_scene02_mc_mse_protocol_v3.json"
DEFAULT_OUTPUT = ROOT / "results" / "paper_scene02_mc_mse_20260826"
DEFAULT_TEMP = ROOT / ".tmp" / "paper_scene02_mc_mse_20260826"
BP_RUNNER = ROOT / "evaluation" / "run_bp_tracker_robustness_v3.py"
SUMMARIZER = ROOT / "evaluation" / "summarize_scene02_mc_mse_suite_v3.py"
MODELS = ("P2", "P3", "P4", "P5")


def write_status(output: Path, manifest_path: Path, states: dict, temp_removed=False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "conditions": states,
        "temporary_raw_reports_removed": bool(temp_removed),
    }
    (output / "STATUS.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def validate_reused_sources(manifest: dict) -> None:
    runs = manifest["monte_carlo"]["formal_runs"]
    for model in MODELS:
        metadata = manifest["reused_source_reports"][model]
        source = ROOT / metadata["path"]
        if sha256(source) != metadata["sha256"]:
            raise RuntimeError(f"reused source hash mismatch: {source}")
        if not report_is_valid(source, manifest, model, runs):
            raise RuntimeError(f"reused source protocol mismatch: {source}")
    condition = next(
        item for item in manifest["robustness"]["conditions"]
        if item["id"] == "pd_0p99_clutter_10"
    )
    metadata = manifest["reused_source_reports"][condition["id"]]
    source = ROOT / metadata["path"]
    if sha256(source) != metadata["sha256"]:
        raise RuntimeError(f"reused source hash mismatch: {source}")
    if not condition_report_is_valid(source, manifest, condition, runs):
        raise RuntimeError(f"reused source protocol mismatch: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--conditions", nargs="+", help="Optional new condition IDs")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_reused_sources(manifest)
    output = args.output.resolve()
    temp = args.temp.resolve()
    allowed_temp_root = (ROOT / ".tmp").resolve()
    if temp == allowed_temp_root or allowed_temp_root not in temp.parents:
        raise RuntimeError(f"temporary raw-report path must be below {allowed_temp_root}: {temp}")
    temp.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    runs = manifest["monte_carlo"]["formal_runs"]
    new_conditions = [
        condition for condition in manifest["robustness"]["conditions"]
        if not condition.get("reuse")
    ]
    if args.conditions:
        requested = set(args.conditions)
        known = {condition["id"] for condition in new_conditions}
        if requested - known:
            raise ValueError(f"unknown or reusable condition IDs: {sorted(requested - known)}")
        new_conditions = [condition for condition in new_conditions if condition["id"] in requested]

    states = {
        condition["id"]: (
            "reused"
            if condition.get("reuse")
            else (
                "complete"
                if condition_report_is_valid(
                    temp / f"{condition['id']}.json", manifest, condition, runs
                )
                else "pending"
            )
        )
        for condition in manifest["robustness"]["conditions"]
    }
    write_status(output, manifest_path, states)
    measurement = manifest["measurement_and_tracker"]
    fixed_q = manifest["fixed_process_noise"]
    ospa = manifest["ospa"]
    model = manifest["models"][manifest["robustness"]["model"]]
    for condition in new_conditions:
        condition_id = condition["id"]
        report_path = temp / f"{condition_id}.json"
        if condition_report_is_valid(report_path, manifest, condition, runs):
            states[condition_id] = "complete"
            write_status(output, manifest_path, states)
            print(f"[{condition_id}] reuse valid temporary report", flush=True)
            continue
        states[condition_id] = "running"
        write_status(output, manifest_path, states)
        command = [
            sys.executable,
            str(BP_RUNNER),
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
            "--neural_position_q_variance_m2", str(fixed_q["position_variance_per_axis_m2"]),
        ]
        stdout_path = output / "logs" / f"{condition_id}.stdout.log"
        stderr_path = output / "logs" / f"{condition_id}.stderr.log"
        print(f"[{condition_id}] start ({runs} MC)", flush=True)
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                subprocess.run(
                    command, cwd=ROOT, check=True,
                    stdout=stdout_handle, stderr=stderr_handle,
                )
        except subprocess.CalledProcessError:
            states[condition_id] = "failed"
            write_status(output, manifest_path, states)
            raise
        if not condition_report_is_valid(report_path, manifest, condition, runs):
            states[condition_id] = "invalid_report"
            write_status(output, manifest_path, states)
            raise RuntimeError(f"invalid report after run: {report_path}")
        states[condition_id] = "complete"
        write_status(output, manifest_path, states)
        print(f"[{condition_id}] complete", flush=True)

    missing = [condition["id"] for condition in manifest["robustness"]["conditions"] if not condition.get("reuse") and not condition_report_is_valid(temp / f"{condition['id']}.json", manifest, condition, runs)]
    if missing:
        print(f"Partial run complete; aggregation waits for: {missing}", flush=True)
        return 0
    subprocess.run(
        [
            sys.executable,
            str(SUMMARIZER),
            "--manifest", str(manifest_path),
            "--output", str(output),
            "--temp", str(temp),
        ],
        cwd=ROOT,
        check=True,
    )
    shutil.rmtree(temp)
    states = {condition["id"]: "complete" for condition in manifest["robustness"]["conditions"]}
    write_status(output, manifest_path, states, temp_removed=True)
    # Refresh artifact hashes after the final status update without recreating
    # or retaining the temporary per-run reports.
    write_artifact_manifest(output, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
