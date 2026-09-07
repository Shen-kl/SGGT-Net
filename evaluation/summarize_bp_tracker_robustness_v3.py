"""Pool BP robustness cells across trajectories and Monte Carlo rollouts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "rollout_count": int(len(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(value).resolve() for value in args.input]
    reports = [json.loads(path.read_text("utf-8")) for path in paths]
    condition = {
        (
            report["protocol"]["actual_pd"],
            report["protocol"]["actual_clutter_mean"],
            report["protocol"]["nominal_pd"],
            report["protocol"]["nominal_clutter_mean"],
            report["protocol"]["checkpoint_sha256"],
        )
        for report in reports
    }
    if len(condition) != 1:
        raise ValueError("all reports must use one condition and checkpoint")
    runs = [run for report in reports for run in report["runs"]]
    if not runs:
        raise ValueError("no Monte Carlo runs to summarize")
    frame_count = sum(int(run["frame_count"]) for run in runs)
    matched_count = sum(int(run["matched_state_count"]) for run in runs)

    def pooled_frame_mean(key):
        return float(sum(
            float(run[key]) * int(run["frame_count"]) for run in runs
        ) / frame_count)

    position_sse = sum(float(run["matched_position_sse"]) for run in runs)
    velocity_sse = sum(float(run["matched_velocity_sse"]) for run in runs)
    nis_sum = sum(float(run["matched_state_nis_sum"]) for run in runs)
    nll_sum = sum(float(run["matched_state_nll_sum"]) for run in runs)
    coverage_count = sum(
        int(run["matched_state_nis_95_count"]) for run in runs
    )
    actual_pd, actual_clutter, nominal_pd, nominal_clutter, checkpoint_hash = (
        next(iter(condition))
    )
    summary = {
        "mean_ospa": pooled_frame_mean("mean_ospa"),
        "mean_ospa_localization": pooled_frame_mean("mean_ospa_localization"),
        "mean_ospa_cardinality": pooled_frame_mean("mean_ospa_cardinality"),
        "estimated_count_mae": pooled_frame_mean("estimated_count_mae"),
        "mean_false_track_count": pooled_frame_mean("mean_false_track_count"),
        "mean_missed_target_count": pooled_frame_mean("mean_missed_target_count"),
        "matched_position_rmse": (
            math.sqrt(position_sse / matched_count) if matched_count else None
        ),
        "matched_velocity_rmse": (
            math.sqrt(velocity_sse / matched_count) if matched_count else None
        ),
        "matched_state_nis_mean": (
            nis_sum / matched_count if matched_count else None
        ),
        "matched_state_nll_mean": (
            nll_sum / matched_count if matched_count else None
        ),
        "matched_state_nis_95_coverage": (
            coverage_count / matched_count if matched_count else None
        ),
        "fragmentation_per_rollout": mean_std([
            run["fragmentation_count"] for run in runs
        ]),
        "identity_switches_per_rollout": mean_std([
            run["identity_switch_count"] for run in runs
        ]),
        "milliseconds_per_frame": (
            1000.0 * sum(
                run["elapsed_seconds"] for run in runs
            ) / frame_count
        ),
        "frame_count": frame_count,
        "matched_state_count": matched_count,
    }
    report = {
        "protocol": {
            "version": 1,
            "actual_pd": actual_pd,
            "actual_clutter_mean": actual_clutter,
            "nominal_pd": nominal_pd,
            "nominal_clutter_mean": nominal_clutter,
            "checkpoint_sha256": checkpoint_hash,
            "trajectory_count": len(paths),
            "rollout_count": len(runs),
            "pooling": {
                "ospa_and_counts": "frame-count weighted",
                "state_rmse_nis_nll": "matched-state sufficient statistics",
                "fragmentation_identity_switch": "mean/std per rollout",
            },
            "sources": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in paths
            ],
        },
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
