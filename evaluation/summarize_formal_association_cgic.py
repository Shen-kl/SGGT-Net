"""Summarize frozen association-CGIC reports across random seeds."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


VARIANTS = (
    "P2_temporal",
    "P3_own_only",
    "P4_full",
    "P5_no_association_context",
)
SPLITS = ("val", "iid_test", "ood_test", "val_association")
METRICS = (
    "location_rmse_prediction",
    "location_rmse_update",
    "velocity_rmse_prediction",
    "velocity_rmse_update",
)


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_root", default="results/formal_association_cgic"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument(
        "--output", default="results/formal_association_cgic/SUMMARY.json"
    )
    args = parser.parse_args()
    root = Path(args.result_root)
    reports = {}
    protocols = set()
    for seed in args.seeds:
        reports[seed] = {}
        for variant in VARIANTS:
            reports[seed][variant] = {}
            for split in SPLITS:
                path = root / f"seed_{seed}" / f"{variant}_{split}.json"
                if not path.exists():
                    raise SystemExit(f"missing formal report: {path}")
                report = json.loads(path.read_text("utf-8"))
                reports[seed][variant][split] = report
                protocol = report["protocol"]
                protocols.add(
                    (
                        split,
                        int(protocol["trajectory_start"]),
                        int(protocol["trajectory_stop_exclusive"]),
                        bool(protocol.get("association_augmentation", False)),
                        protocol.get("association_seed"),
                    )
                )
                for metric in METRICS:
                    value = float(report["global"][metric])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"non-finite {metric}: seed={seed}, "
                            f"variant={variant}, split={split}"
                        )

    summary = {
        "protocol": {
            "version": 1,
            "seeds": list(args.seeds),
            "seed_count": len(args.seeds),
            "variants": list(VARIANTS),
            "splits": list(SPLITS),
            "aggregation": "arithmetic mean/std across independently trained seeds",
            "paired_trajectory_bootstrap": (
                "stored separately in each seed directory"
            ),
            "observed_evaluation_protocols": [list(row) for row in sorted(protocols)],
        },
        "metrics": {},
        "gains_over_temporal": {},
        "incremental_peer_gain": {},
        "incremental_association_context_gain": {},
    }
    for split in SPLITS:
        summary["metrics"][split] = {}
        for variant in VARIANTS:
            summary["metrics"][split][variant] = {}
            for metric in METRICS:
                values = [
                    reports[seed][variant][split]["global"][metric]
                    for seed in args.seeds
                ]
                summary["metrics"][split][variant][metric] = mean_std(values)

        for destination, left, right in (
            ("gains_over_temporal", "P2_temporal", "P4_full"),
            ("incremental_peer_gain", "P3_own_only", "P4_full"),
            (
                "incremental_association_context_gain",
                "P5_no_association_context",
                "P4_full",
            ),
        ):
            summary[destination][split] = {}
            for metric in METRICS:
                gains = [
                    float(reports[seed][left][split]["global"][metric])
                    - float(reports[seed][right][split]["global"][metric])
                    for seed in args.seeds
                ]
                summary[destination][split][metric] = mean_std(gains)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

