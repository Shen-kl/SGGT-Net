"""Trajectory-paired bootstrap comparison for two V3 evaluation reports."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bootstrap_causal_activation_v3 import (
    EVENT_NAMES,
    METRICS,
    _bootstrap_comparison,
    _index,
    _sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left_name", required=True)
    parser.add_argument("--left_report", required=True)
    parser.add_argument("--right_name", required=True)
    parser.add_argument("--right_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.left_name == args.right_name:
        raise ValueError("comparison names must differ")
    if args.samples < 100:
        raise ValueError("bootstrap samples must be at least 100")

    paths = {
        args.left_name: Path(args.left_report).resolve(),
        args.right_name: Path(args.right_report).resolve(),
    }
    reports = {
        name: json.loads(path.read_text("utf-8"))
        for name, path in paths.items()
    }
    split_values = {report["protocol"]["split"] for report in reports.values()}
    ranges = {
        (
            int(report["protocol"]["trajectory_start"]),
            int(report["protocol"]["trajectory_stop_exclusive"]),
        )
        for report in reports.values()
    }
    association_protocols = {
        (
            bool(report["protocol"].get("association_augmentation", False)),
            report["protocol"].get("association_seed"),
        )
        for report in reports.values()
    }
    if len(split_values) != 1 or len(ranges) != 1:
        raise ValueError("reports must use the same split and trajectory range")
    if len(association_protocols) != 1:
        raise ValueError("reports must use the same association protocol")

    indexed = {name: _index(report) for name, report in reports.items()}
    comparison = {}
    for event_index, event in enumerate(EVENT_NAMES):
        if not all(
            any(event in row for row in rows.values())
            for rows in indexed.values()
        ):
            continue
        comparison[event] = {}
        for metric_index, metric in enumerate(METRICS):
            comparison[event][metric] = _bootstrap_comparison(
                indexed[args.left_name],
                indexed[args.right_name],
                event,
                metric,
                samples=args.samples,
                seed=args.seed + 100 * event_index + metric_index,
            )

    output = {
        "protocol": {
            "version": 1,
            "left_name": args.left_name,
            "right_name": args.right_name,
            "positive_improvement_means": (
                f"{args.right_name} has lower RMSE than {args.left_name}"
            ),
            "split": next(iter(split_values)),
            "trajectory_range": list(next(iter(ranges))),
            "association_protocol": list(next(iter(association_protocols))),
            "bootstrap_unit": "trajectory",
            "paired_resampling": True,
            "bootstrap_samples": int(args.samples),
            "seed": int(args.seed),
            "reports": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
        },
        "comparison": comparison,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
