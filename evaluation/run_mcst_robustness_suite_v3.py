"""Run the formal MCST-BP robustness suite on the simulation scenario.

Raw per-run trajectories are kept only under .tmp while a condition is being
evaluated.  Successful conditions retain aggregate Monte Carlo MSE curves,
global RMSE values, protocol metadata, and stdout/stderr logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BP_RUNNER = ROOT / "evaluation" / "run_bp_tracker_robustness_v3.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoint" / "mcst_v9_full_seed123" / "best_prior_position.pth"
DEFAULT_MCST_REPOSITORY = Path(r"E:\PycharmProjects\Neural Network\MCST_2D_update")
DEFAULT_TRAJECTORY = Path(
    r"E:\PycharmProjects\Neural Network\GNN_groupTargetTracking_v9_norm_without_struct"
    r"\generate_trajectory\scene02\trajectory01.pt"
)
DEFAULT_STANDARD_REPORT = ROOT / "results" / "mcst_v9_seed123" / "simulation_scenario_mc20.json"
DEFAULT_OUTPUT = ROOT / "results" / "mcst_v9_seed123" / "robustness"
DEFAULT_TEMP = ROOT / ".tmp" / "mcst_v9_seed123_robustness"

CONDITIONS = (
    {"id": "pd_0p99_clutter_3", "actual_pd": 0.99, "actual_clutter_mean": 3.0},
    {"id": "pd_0p90_clutter_3", "actual_pd": 0.90, "actual_clutter_mean": 3.0},
    {"id": "pd_0p80_clutter_3", "actual_pd": 0.80, "actual_clutter_mean": 3.0},
    {"id": "pd_0p99_clutter_2", "actual_pd": 0.99, "actual_clutter_mean": 2.0, "reuse": True},
    {"id": "pd_0p99_clutter_10", "actual_pd": 0.99, "actual_clutter_mean": 10.0},
    {"id": "pd_0p99_clutter_15", "actual_pd": 0.99, "actual_clutter_mean": 15.0},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def report_is_valid(
    path: Path,
    condition: dict,
    checkpoint_hash: str,
    monte_carlo: int,
    seed: int,
) -> bool:
    if not path.exists():
        return False
    try:
        report = load_json(path)
        protocol = report["protocol"]
        return all(
            (
                protocol.get("model_backend") == "mcst",
                protocol.get("checkpoint_sha256") == checkpoint_hash,
                float(protocol.get("actual_pd")) == float(condition["actual_pd"]),
                float(protocol.get("actual_clutter_mean"))
                == float(condition["actual_clutter_mean"]),
                float(protocol.get("nominal_pd")) == 0.99,
                float(protocol.get("nominal_clutter_mean")) == 2.0,
                float(protocol.get("range_noise_std_m")) == 3.0,
                float(protocol.get("azimuth_noise_std_deg")) == 0.01,
                int(protocol.get("monte_carlo_runs")) == monte_carlo,
                int(protocol.get("seed_start")) == seed,
                len(report.get("runs", [])) == monte_carlo,
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def aggregate_report(report: dict) -> tuple[dict[str, np.ndarray], dict]:
    runs = report["runs"]
    if not runs:
        raise ValueError("report contains no Monte Carlo runs")
    frame_count = len(runs[0]["per_frame"])
    if any(len(run["per_frame"]) != frame_count for run in runs):
        raise ValueError("Monte Carlo runs do not share a common frame count")

    ospa = np.asarray(
        [[frame["ospa"] for frame in run["per_frame"]] for run in runs],
        dtype=np.float64,
    )
    cardinality_error = np.asarray(
        [
            [frame["estimated_count"] - frame["truth_count"] for frame in run["per_frame"]]
            for run in runs
        ],
        dtype=np.float64,
    )
    position_sse = np.asarray(
        [[frame["position_sse"] for frame in run["per_frame"]] for run in runs],
        dtype=np.float64,
    )
    velocity_sse = np.asarray(
        [[frame["velocity_sse"] for frame in run["per_frame"]] for run in runs],
        dtype=np.float64,
    )
    matched_count = np.asarray(
        [[frame["matched_state_count"] for frame in run["per_frame"]] for run in runs],
        dtype=np.float64,
    )
    denominator = np.maximum(matched_count.sum(axis=0), 1.0)
    arrays = {
        "ospa_mc_mse_curve": np.mean(np.square(ospa), axis=0),
        "position_mc_mse_curve": position_sse.sum(axis=0) / denominator,
        "velocity_mc_mse_curve": velocity_sse.sum(axis=0) / denominator,
        "cardinality_mc_mse_curve": np.mean(np.square(cardinality_error), axis=0),
    }
    final = {
        "ospa_final_rmse": float(np.sqrt(np.mean(arrays["ospa_mc_mse_curve"]))),
        "position_final_rmse": float(
            np.sqrt(position_sse.sum() / max(float(matched_count.sum()), 1.0))
        ),
        "velocity_final_rmse": float(
            np.sqrt(velocity_sse.sum() / max(float(matched_count.sum()), 1.0))
        ),
        "cardinality_final_rmse": float(
            np.sqrt(np.mean(arrays["cardinality_mc_mse_curve"]))
        ),
    }
    return arrays, final


def condition_artifact_is_valid(
    json_path: Path,
    npz_path: Path,
    condition: dict,
    checkpoint_hash: str,
    monte_carlo: int,
    seed: int,
) -> bool:
    if not json_path.exists() or not npz_path.exists():
        return False
    try:
        payload = load_json(json_path)
        protocol = payload["protocol"]
        with np.load(npz_path) as arrays:
            curves_ok = all(
                key in arrays and arrays[key].shape == (100,)
                for key in (
                    "ospa_mc_mse_curve",
                    "position_mc_mse_curve",
                    "velocity_mc_mse_curve",
                    "cardinality_mc_mse_curve",
                )
            )
        return curves_ok and all(
            (
                protocol.get("checkpoint_sha256") == checkpoint_hash,
                float(protocol.get("actual_pd")) == float(condition["actual_pd"]),
                float(protocol.get("actual_clutter_mean"))
                == float(condition["actual_clutter_mean"]),
                int(protocol.get("monte_carlo_runs")) == monte_carlo,
                int(protocol.get("seed_start")) == seed,
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def save_aggregate(report: dict, condition: dict, output: Path) -> dict:
    arrays, final = aggregate_report(report)
    condition_id = condition["id"]
    npz_path = output / "curves" / f"{condition_id}.npz"
    json_path = output / "aggregate" / f"{condition_id}.json"
    np.savez_compressed(npz_path, **arrays)
    payload = {
        "condition": condition_id,
        "protocol": report["protocol"],
        "storage": {
            "raw_per_run_curves_saved": False,
            "curve": "per-frame Monte Carlo MSE",
            "final": "global RMSE",
            "curve_file": str(npz_path.resolve()),
        },
        "final_rmse": final,
        "run_summary": report["summary"],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_status(output: Path, states: dict[str, str], checkpoint_hash: str) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "conditions": states,
    }
    (output / "STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_combined_artifacts(output: Path) -> None:
    payloads = {
        condition["id"]: load_json(output / "aggregate" / f"{condition['id']}.json")
        for condition in CONDITIONS
    }
    columns = (
        "condition",
        "actual_pd",
        "actual_clutter_mean",
        "ospa_final_rmse",
        "position_final_rmse",
        "velocity_final_rmse",
        "cardinality_final_rmse",
        "mean_ospa",
        "estimated_count_mae",
        "milliseconds_per_frame",
    )
    rows = []
    for condition in CONDITIONS:
        payload = payloads[condition["id"]]
        run_summary = payload["run_summary"]
        rows.append(
            {
                "condition": condition["id"],
                "actual_pd": condition["actual_pd"],
                "actual_clutter_mean": condition["actual_clutter_mean"],
                **payload["final_rmse"],
                "mean_ospa": run_summary["mean_ospa"]["mean"],
                "estimated_count_mae": run_summary["estimated_count_mae"]["mean"],
                "milliseconds_per_frame": run_summary["milliseconds_per_frame"]["mean"],
            }
        )
    with (output / "robustness_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "robustness_summary.json").write_text(
        json.dumps({"conditions": rows}, indent=2), encoding="utf-8"
    )

    metric_specs = (
        ("ospa_mc_mse_curve", "OSPA RMSE", "m"),
        ("position_mc_mse_curve", "Matched position RMSE", "m"),
        ("velocity_mc_mse_curve", "Matched velocity RMSE", "m/s"),
        ("cardinality_mc_mse_curve", "Cardinality RMSE", "targets"),
    )
    groups = {
        "pd": CONDITIONS[:3],
        "clutter": CONDITIONS[3:],
    }
    for group_name, conditions in groups.items():
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for condition in conditions:
            with np.load(output / "curves" / f"{condition['id']}.npz") as loaded:
                arrays = {key: loaded[key].copy() for key, _, _ in metric_specs}
            if group_name == "pd":
                label = f"$P_D$={condition['actual_pd']:.2f}"
            else:
                label = f"clutter={condition['actual_clutter_mean']:g}"
            for axis, (key, title, unit) in zip(axes.flat, metric_specs):
                axis.plot(np.sqrt(arrays[key]), linewidth=1.4, label=label)
                axis.set_xlabel("Time step")
                axis.set_ylabel(unit)
                axis.set_title(title)
                axis.grid(alpha=0.25)
        for axis in axes.flat:
            axis.legend()
        figure.savefig(output / f"robustness_{group_name}_mc_rmse.png", dpi=220)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mcst_repository", type=Path, default=DEFAULT_MCST_REPOSITORY)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--standard_report", type=Path, default=DEFAULT_STANDARD_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--monte_carlo", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    trajectory = args.trajectory.resolve()
    standard_report = args.standard_report.resolve()
    output = args.output.resolve()
    temp = args.temp.resolve()
    allowed_temp = (ROOT / ".tmp").resolve()
    if temp == allowed_temp or allowed_temp not in temp.parents:
        raise ValueError(f"temporary path must be below {allowed_temp}: {temp}")
    for path in (checkpoint, trajectory, standard_report, BP_RUNNER):
        if not path.exists():
            raise FileNotFoundError(path)
    for directory in (output / "aggregate", output / "curves", output / "logs", temp):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = sha256(checkpoint)
    states = {condition["id"]: "pending" for condition in CONDITIONS}
    write_status(output, states, checkpoint_hash)
    for condition in CONDITIONS:
        condition_id = condition["id"]
        aggregate_json = output / "aggregate" / f"{condition_id}.json"
        aggregate_npz = output / "curves" / f"{condition_id}.npz"
        if condition_artifact_is_valid(
            aggregate_json,
            aggregate_npz,
            condition,
            checkpoint_hash,
            args.monte_carlo,
            args.seed,
        ):
            states[condition_id] = "complete"
            write_status(output, states, checkpoint_hash)
            print(f"[{condition_id}] reuse valid aggregate", flush=True)
            continue

        raw_report = standard_report if condition.get("reuse") else temp / f"{condition_id}.json"
        if not report_is_valid(
            raw_report, condition, checkpoint_hash, args.monte_carlo, args.seed
        ):
            if condition.get("reuse"):
                raise RuntimeError(f"standard report does not match formal protocol: {raw_report}")
            states[condition_id] = "running"
            write_status(output, states, checkpoint_hash)
            command = [
                sys.executable,
                str(BP_RUNNER),
                "--model_backend", "mcst",
                "--mcst_repository", str(args.mcst_repository.resolve()),
                "--checkpoint", str(checkpoint),
                "--trajectory", str(trajectory),
                "--output", str(raw_report),
                "--actual_pd", str(condition["actual_pd"]),
                "--actual_clutter_mean", str(condition["actual_clutter_mean"]),
                "--nominal_pd", "0.99",
                "--nominal_clutter_mean", "2",
                "--range_noise_std_m", "3",
                "--azimuth_noise_std_deg", "0.01",
                "--monte_carlo", str(args.monte_carlo),
                "--seed", str(args.seed),
                "--device", args.device,
                "--ospa_cutoff_m", "100",
                "--ospa_order", "1",
                "--neural_process_noise_scale", "6.874306",
                "--neural_position_q_variance_m2", "1.316616",
            ]
            stdout_path = output / "logs" / f"{condition_id}.stdout.log"
            stderr_path = output / "logs" / f"{condition_id}.stderr.log"
            print(f"[{condition_id}] start ({args.monte_carlo} MC)", flush=True)
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
                write_status(output, states, checkpoint_hash)
                raise
            if not report_is_valid(
                raw_report, condition, checkpoint_hash, args.monte_carlo, args.seed
            ):
                states[condition_id] = "invalid_report"
                write_status(output, states, checkpoint_hash)
                raise RuntimeError(f"invalid report after run: {raw_report}")

        report = load_json(raw_report)
        save_aggregate(report, condition, output)
        if not condition.get("reuse"):
            raw_report.unlink()
        states[condition_id] = "complete"
        write_status(output, states, checkpoint_hash)
        print(f"[{condition_id}] complete", flush=True)

    write_combined_artifacts(output)
    (output / "DONE").write_text(
        datetime.now(timezone.utc).astimezone().isoformat(), encoding="utf-8"
    )
    print(json.dumps(load_json(output / "robustness_summary.json"), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
