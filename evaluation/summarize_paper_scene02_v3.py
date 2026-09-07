"""Summarize current-model scene02 reports and frozen historical baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_ospa_rms(run: dict) -> float:
    values = np.asarray([frame["ospa"] for frame in run["per_frame"]], dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def run_metric_rmse(run: dict, sse_key: str) -> float:
    count = int(run["matched_state_count"])
    return float(np.sqrt(float(run[sse_key]) / count)) if count else float("nan")


def model_summary(report: dict) -> dict:
    runs = report["runs"]
    frame_matrix = np.asarray(
        [[frame["ospa"] for frame in run["per_frame"]] for run in runs], dtype=float
    )
    run_rms = np.asarray([run_ospa_rms(run) for run in runs], dtype=float)
    position = np.asarray([run_metric_rmse(run, "matched_position_sse") for run in runs])
    velocity = np.asarray([run_metric_rmse(run, "matched_velocity_sse") for run in runs])
    total_count = sum(int(run["matched_state_count"]) for run in runs)
    return {
        "monte_carlo_runs": len(runs),
        "legacy_ospa_curve_rms_m": float(np.sqrt(np.mean(np.square(frame_matrix.mean(axis=0))))),
        "run_ospa_rms_mean_m": float(run_rms.mean()),
        "run_ospa_rms_std_m": float(run_rms.std(ddof=1)),
        "matched_position_global_rmse_m": float(
            np.sqrt(sum(float(run["matched_position_sse"]) for run in runs) / total_count)
        ),
        "matched_velocity_global_rmse_mps": float(
            np.sqrt(sum(float(run["matched_velocity_sse"]) for run in runs) / total_count)
        ),
        "estimated_count_mae_mean": float(np.mean([run["estimated_count_mae"] for run in runs])),
        "milliseconds_per_frame_mean": float(np.mean([run["milliseconds_per_frame"] for run in runs])),
        "_run_ospa_rms": run_rms,
        "_run_position_rmse": position,
        "_run_velocity_rmse": velocity,
        "_measurement_counts": [
            [frame["measurement_count"] for frame in run["per_frame"]] for run in runs
        ],
        "_seeds": [run["seed"] for run in runs],
    }


def paired_bootstrap(reference: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    gain = reference[valid] - candidate[valid]
    rng = np.random.default_rng(seed)
    samples = gain[rng.integers(0, len(gain), size=(10000, len(gain)))].mean(axis=1)
    return {
        "paired_runs": int(len(gain)),
        "mean_gain": float(gain.mean()),
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "win_fraction": float(np.mean(gain > 0)),
    }


def public(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reports = {
        model_id: json.loads((args.results / "formal" / f"{model_id}.json").read_text(encoding="utf-8"))
        for model_id in ("P2", "P3", "P4", "P5")
    }
    summaries = {model_id: model_summary(report) for model_id, report in reports.items()}
    seeds = summaries["P4"]["_seeds"]
    counts = summaries["P4"]["_measurement_counts"]
    for model_id in ("P2", "P3", "P5"):
        if summaries[model_id]["_seeds"] != seeds:
            raise RuntimeError(f"Unpaired Monte Carlo seeds for {model_id}")
        if summaries[model_id]["_measurement_counts"] != counts:
            raise RuntimeError(f"Measurement realizations differ for {model_id}")

    bootstrap = {}
    for model_id in ("P2", "P3", "P5"):
        bootstrap[f"P4_over_{model_id}"] = {
            "ospa_rms_m": paired_bootstrap(
                summaries[model_id]["_run_ospa_rms"], summaries["P4"]["_run_ospa_rms"], 20260823
            ),
            "matched_position_rmse_m": paired_bootstrap(
                summaries[model_id]["_run_position_rmse"], summaries["P4"]["_run_position_rmse"], 20260824
            ),
            "matched_velocity_rmse_mps": paired_bootstrap(
                summaries[model_id]["_run_velocity_rmse"], summaries["P4"]["_run_velocity_rmse"], 20260825
            ),
        }

    output = args.results / "summary"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": manifest["protocol_name"],
        "trajectory_sha256": manifest["trajectory"]["sha256"],
        "current_models": {model_id: public(value) for model_id, value in summaries.items()},
        "paired_bootstrap": bootstrap,
        "historical_baselines": manifest["frozen_historical_baselines"],
        "historical_baseline_limitation": "Frozen descriptive values; original seeds and paired measurement realizations are unavailable.",
    }
    (output / "SUMMARY.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    with (output / "ablation_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *public(summaries["P2"]).keys()])
        writer.writeheader()
        for model_id in ("P2", "P3", "P4", "P5"):
            writer.writerow({"model": model_id, **public(summaries[model_id])})

    with (output / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "source", "ospa_rmse_m", "monte_carlo_runs", "paired_with_p4"),
        )
        writer.writeheader()
        writer.writerow({
            "method": "P4 complete CGIC",
            "source": "new frozen scene02 protocol",
            "ospa_rmse_m": summaries["P4"]["legacy_ospa_curve_rms_m"],
            "monte_carlo_runs": summaries["P4"]["monte_carlo_runs"],
            "paired_with_p4": True,
        })
        for method, baseline in manifest["frozen_historical_baselines"].items():
            writer.writerow({
                "method": method,
                "source": baseline["source"],
                "ospa_rmse_m": baseline["paper_reported_ospa_rmse_m"],
                "monte_carlo_runs": baseline["monte_carlo_runs"],
                "paired_with_p4": False,
            })
    print(json.dumps(payload["current_models"], indent=2))
    print(f"summary={output / 'SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
