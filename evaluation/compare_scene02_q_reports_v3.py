"""Paired trajectory-level comparison of dynamic and fixed scene02 Q reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "location_rmse_prediction",
    "location_rmse_update",
    "velocity_rmse_prediction",
    "velocity_rmse_update",
)


def select_variant(
    report, scale=None, learned_position=False, fixed_position_variance=0.0
):
    matches = []
    for variant in report["variants"]:
        if scale is not None and float(variant["process_noise_scale"]) != float(scale):
            continue
        is_learned = variant["direct_position_variance_per_axis_m2"] is None
        if is_learned != bool(learned_position):
            continue
        if not is_learned and float(
            variant["direct_position_variance_per_axis_m2"]
        ) != float(fixed_position_variance):
            continue
        matches.append(variant)
    if len(matches) != 1:
        raise ValueError(f"expected one matching variant, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--fixed_scale", type=float, default=8.0)
    parser.add_argument("--fixed_position_variance", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    dynamic_report = json.loads(Path(args.dynamic).read_text(encoding="utf-8"))
    fixed_report = json.loads(Path(args.fixed).read_text(encoding="utf-8"))
    dynamic = select_variant(dynamic_report, scale=1.0, learned_position=True)
    fixed = select_variant(
        fixed_report, scale=args.fixed_scale, learned_position=False,
        fixed_position_variance=args.fixed_position_variance,
    )
    dynamic_by_seed = {int(run["seed"]): run for run in dynamic["runs"]}
    fixed_by_seed = {int(run["seed"]): run for run in fixed["runs"]}
    seeds = sorted(set(dynamic_by_seed) & set(fixed_by_seed))
    if not seeds or not set(fixed_by_seed).issubset(set(dynamic_by_seed)):
        raise ValueError(
            "every fixed-report seed must be present in the dynamic report"
        )
    rng = np.random.default_rng(args.seed)
    metrics = {}
    for metric in METRICS:
        dynamic_values = np.asarray([
            dynamic_by_seed[seed]["metrics"][metric] for seed in seeds
        ], dtype=float)
        fixed_values = np.asarray([
            fixed_by_seed[seed]["metrics"][metric] for seed in seeds
        ], dtype=float)
        paired = dynamic_values - fixed_values
        sample_index = rng.integers(
            0, len(seeds), size=(args.bootstrap_samples, len(seeds))
        )
        bootstrap = paired[sample_index].mean(axis=1)
        metrics[metric] = {
            "dynamic_mean": float(dynamic_values.mean()),
            "fixed_mean": float(fixed_values.mean()),
            "paired_dynamic_minus_fixed_mean": float(paired.mean()),
            "paired_bootstrap_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "dynamic_better_runs": int((paired < 0.0).sum()),
            "paired_runs": len(seeds),
        }
    output = {
        "protocol": {
            "dynamic_report": str(Path(args.dynamic).resolve()),
            "fixed_report": str(Path(args.fixed).resolve()),
            "fixed_scale": args.fixed_scale,
            "fixed_position_variance": args.fixed_position_variance,
            "seeds": seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "difference_sign": "dynamic minus fixed; negative favors dynamic",
        },
        "metrics": metrics,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
