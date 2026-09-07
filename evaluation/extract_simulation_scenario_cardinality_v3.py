"""Extract aggregate SGGT-Net cardinality curves from a formal tracker report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "results"
    / "paper_scene02_fixed_q_20260824"
    / "formal"
    / "P4.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "paper_simulation_scenario_cardinality_20260903"
OUTPUT_STEM = "SGGT_NET_CARDINALITY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"formal report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_count_matrices(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("formal report contains no Monte Carlo runs")

    estimated_rows: list[list[float]] = []
    truth_rows: list[list[int]] = []
    expected_frames: list[int] | None = None
    for run_index, run in enumerate(runs):
        frames = run.get("per_frame")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"run {run_index} contains no per-frame records")
        frame_indices = [int(frame["frame"]) for frame in frames]
        if expected_frames is None:
            expected_frames = frame_indices
        elif frame_indices != expected_frames:
            raise ValueError(f"run {run_index} uses a different frame sequence")
        estimated_rows.append([float(frame["estimated_count"]) for frame in frames])
        truth_rows.append([int(frame["truth_count"]) for frame in frames])

    estimated = np.asarray(estimated_rows, dtype=float)
    truth = np.asarray(truth_rows, dtype=int)
    if not np.all(np.isfinite(estimated)):
        raise ValueError("estimated cardinality contains NaN or infinity")
    if np.any(estimated < 0) or np.any(truth < 0):
        raise ValueError("cardinality cannot be negative")
    if not np.all(truth == truth[0]):
        raise ValueError("ground-truth cardinality differs across Monte Carlo runs")
    return estimated, truth[0]


def selected_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "checkpoint",
        "checkpoint_sha256",
        "trajectory",
        "trajectory_sha256",
        "actual_pd",
        "actual_clutter_mean",
        "nominal_pd",
        "nominal_clutter_mean",
        "range_noise_std_m",
        "azimuth_noise_std_deg",
        "monte_carlo_runs",
        "seed_start",
        "evaluated_frames",
    )
    return {key: protocol.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delta-t", type=float, default=2.0, help="frame interval in seconds")
    parser.add_argument("--method", default="SGGT-Net")
    args = parser.parse_args()
    if args.delta_t <= 0:
        raise ValueError("--delta-t must be positive")

    report_path = args.report.resolve()
    report = read_json(report_path)
    estimated, truth = extract_count_matrices(report)
    run_count, frame_count = estimated.shape
    protocol = report.get("protocol", {})
    declared_runs = protocol.get("monte_carlo_runs")
    if declared_runs is not None and int(declared_runs) != run_count:
        raise ValueError(
            f"protocol declares {declared_runs} runs, but report contains {run_count}"
        )

    frame_index = np.arange(frame_count, dtype=int)
    time_s = frame_index.astype(float) * args.delta_t
    error = estimated - truth[None, :]
    count_mean = estimated.mean(axis=0)
    count_std = estimated.std(axis=0, ddof=0)
    count_q025, count_q975 = np.quantile(estimated, (0.025, 0.975), axis=0)
    count_mse = np.mean(np.square(error), axis=0)
    count_mae = np.mean(np.abs(error), axis=0)

    arrays = {
        "frame_index": frame_index,
        "time_s": time_s,
        "ground_truth_count": truth,
        "estimated_count_mc_mean": count_mean,
        "estimated_count_mc_std": count_std,
        "estimated_count_mc_q025": count_q025,
        "estimated_count_mc_q975": count_q975,
        "count_error_mc_mse": count_mse,
        "count_error_mc_mae": count_mae,
    }
    summary = {
        "monte_carlo_runs": run_count,
        "frame_count": frame_count,
        "delta_t_seconds": args.delta_t,
        "global_count_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "global_count_mae": float(np.mean(np.abs(error))),
        "maximum_absolute_mean_bias": float(np.max(np.abs(count_mean - truth))),
    }
    payload = {
        "schema_version": 1,
        "method": args.method,
        "curve_definition": (
            "estimated_count_mc_mean is the arithmetic mean of the estimated "
            "target cardinality over Monte Carlo runs at each frame"
        ),
        "source_report": str(report_path),
        "source_report_sha256": sha256_file(report_path),
        "protocol": selected_protocol(protocol),
        "storage": {
            "raw_per_run_cardinality_saved": False,
            "aggregate_statistics_only": True,
        },
        "summary": summary,
        "arrays": {key: value.tolist() for key, value in arrays.items()},
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{OUTPUT_STEM}.json"
    csv_path = output / f"{OUTPUT_STEM}.csv"
    npz_path = output / f"{OUTPUT_STEM}.npz"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = tuple(arrays)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(frame_count):
            writer.writerow({key: arrays[key][frame] for key in fieldnames})
    np.savez_compressed(npz_path, **arrays)

    print(
        json.dumps(
            {
                "summary": summary,
                "generated": [str(json_path), str(csv_path), str(npz_path)],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
