"""Trajectory-paired bootstrap for fixed and dynamic causal correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EVENT_NAMES = ("all", "steady", "switch", "strong", "unclassified")
METRICS = (
    "position_prediction",
    "position_update",
    "velocity_prediction",
    "velocity_update",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _index(report: dict) -> dict:
    return {
        int(row.get("trajectory_id", row["trajectory_index"])): row["events"]
        for row in report["trajectories"]
    }


def _bootstrap_comparison(
    left: dict,
    right: dict,
    event: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict:
    ids = sorted(
        trajectory_id for trajectory_id in set(left) & set(right)
        if event in left[trajectory_id] and event in right[trajectory_id]
    )
    if not ids:
        raise ValueError(f"no paired trajectories for event={event}")
    left_sse, right_sse, counts, trajectory_gain = [], [], [], []
    sse_key = f"{metric}_sse"
    for trajectory_id in ids:
        left_row = left[trajectory_id][event]
        right_row = right[trajectory_id][event]
        left_count = int(left_row["count"])
        right_count = int(right_row["count"])
        if left_count != right_count:
            raise ValueError(
                f"count mismatch for trajectory={trajectory_id}, event={event}: "
                f"{left_count} != {right_count}"
            )
        left_value = float(left_row[sse_key])
        right_value = float(right_row[sse_key])
        counts.append(left_count)
        left_sse.append(left_value)
        right_sse.append(right_value)
        trajectory_gain.append(
            np.sqrt(left_value / left_count)
            - np.sqrt(right_value / right_count)
        )
    counts = np.asarray(counts, dtype=np.float64)
    left_sse = np.asarray(left_sse, dtype=np.float64)
    right_sse = np.asarray(right_sse, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(int(samples), len(ids)))
    bootstrap_count = counts[indices].sum(axis=1)
    differences = (
        np.sqrt(left_sse[indices].sum(axis=1) / bootstrap_count)
        - np.sqrt(right_sse[indices].sum(axis=1) / bootstrap_count)
    )
    left_point = float(np.sqrt(left_sse.sum() / counts.sum()))
    right_point = float(np.sqrt(right_sse.sum() / counts.sum()))
    trajectory_gain = np.asarray(trajectory_gain, dtype=np.float64)
    return {
        "trajectory_count": len(ids),
        "point_count": int(counts.sum()),
        "left_rmse": left_point,
        "right_rmse": right_point,
        "right_improvement": left_point - right_point,
        "improvement_95ci": _interval(differences),
        "probability_improvement_positive": float((differences > 0).mean()),
        "trajectory_win_fraction": float((trajectory_gain > 0).mean()),
        "trajectory_gain_median": float(np.median(trajectory_gain)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    paths = {name: Path(path).resolve() for name, path in args.report}
    if len(paths) != len(args.report):
        raise ValueError("report names must be unique")
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    splits = {report["protocol"]["split"] for report in reports.values()}
    ranges = {
        (
            report["protocol"]["trajectory_start"],
            report["protocol"]["trajectory_stop_exclusive"],
        )
        for report in reports.values()
    }
    if len(splits) != 1 or len(ranges) != 1:
        raise ValueError("all reports must use the same split and trajectory range")
    required = {"temporal", "fixed05", "full", "dynamic"}
    if set(reports) != required:
        raise ValueError(f"reports must be named exactly {sorted(required)}")
    indexed = {name: _index(report) for name, report in reports.items()}
    comparisons = {
        "dynamic_over_temporal": ("temporal", "dynamic"),
        "dynamic_over_fixed05": ("fixed05", "dynamic"),
        "full_over_dynamic": ("dynamic", "full"),
        "full_over_temporal": ("temporal", "full"),
    }
    output = {
        "protocol": {
            "version": 1,
            "split": next(iter(splits)),
            "trajectory_range": list(next(iter(ranges))),
            "bootstrap_unit": "trajectory",
            "paired_resampling": True,
            "pooled_rmse_from_resampled_sse": True,
            "bootstrap_samples": int(args.samples),
            "seed": int(args.seed),
            "confidence_interval": "percentile 95%",
            "reports": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
        },
        "comparisons": {},
    }
    for comparison_index, (name, (left, right)) in enumerate(comparisons.items()):
        event_results = {}
        for event_index, event in enumerate(EVENT_NAMES):
            metric_results = {}
            for metric_index, metric in enumerate(METRICS):
                metric_results[metric] = _bootstrap_comparison(
                    indexed[left], indexed[right], event, metric,
                    samples=args.samples,
                    seed=(
                        args.seed + 1000 * comparison_index
                        + 100 * event_index + metric_index
                    ),
                )
            event_results[event] = metric_results
        output["comparisons"][name] = event_results
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
