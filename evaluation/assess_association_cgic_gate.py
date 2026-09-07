"""Create the machine-readable 200/50 gate decision for formal CGIC runs."""

import argparse
import json
import math
from pathlib import Path

import torch


def finite_metrics(metrics):
    return all(
        math.isfinite(float(value))
        for value in metrics.values()
        if isinstance(value, (int, float))
    )


def position_score(metrics):
    return float(metrics["location_rmse_prediction"]) + float(
        metrics["location_rmse_update"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        default="checkpoint/dev_assoc_cgic_fixedval_200_50",
    )
    parser.add_argument(
        "--output",
        default="results/association_cgic_200_50_gate.json",
    )
    parser.add_argument("--minimum_prediction_gain_m", type=float, default=0.05)
    parser.add_argument("--maximum_clean_regression_m", type=float, default=0.05)
    parser.add_argument("--minimum_correction_rms_m", type=float, default=1e-4)
    parser.add_argument("--maximum_saturation_fraction", type=float, default=0.98)
    args = parser.parse_args()

    checkpoint_dir = Path(args.run_dir) / "GNN_groupTargetTracking"
    baseline_path = checkpoint_dir / "epoch_minus_one_metrics.json"
    best_path = checkpoint_dir / "best_association_position.pth"
    if not baseline_path.exists() or not best_path.exists():
        raise SystemExit("The development run has not produced gate artifacts")

    baseline = json.loads(baseline_path.read_text("utf-8"))
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    clean_initial = baseline["clean_validation"]
    association_initial = baseline["association_validation"]
    clean_best = best["metrics"]
    association_best = best["association_validation_metrics"]

    association_prediction_gain = (
        float(association_initial["location_rmse_prediction"])
        - float(association_best["location_rmse_prediction"])
    )
    association_update_gain = (
        float(association_initial["location_rmse_update"])
        - float(association_best["location_rmse_update"])
    )
    clean_prediction_gain = (
        float(clean_initial["location_rmse_prediction"])
        - float(clean_best["location_rmse_prediction"])
    )
    clean_update_gain = (
        float(clean_initial["location_rmse_update"])
        - float(clean_best["location_rmse_update"])
    )
    correction_rms = float(
        association_best["causal_innovation_position_rms_m"]
    )
    correction_max = float(
        association_best["causal_innovation_position_max_m"]
    )
    max_position = float(best.get("causal_innovation_max_position_m", 50.0))

    checks = {
        "protocol_is_200_50": (
            int(best["evaluation_protocol"]["trajectory_count"]) == 50
            and len(best["evaluation_protocol"]["trajectories"]) == 50
        ),
        "all_metrics_finite": finite_metrics(clean_best)
        and finite_metrics(association_best),
        "association_prediction_gain_at_least_threshold": (
            association_prediction_gain >= args.minimum_prediction_gain_m
        ),
        "association_update_not_worse": association_update_gain >= 0.0,
        "clean_prediction_not_materially_worse": (
            clean_prediction_gain >= -args.maximum_clean_regression_m
        ),
        "clean_update_not_materially_worse": (
            clean_update_gain >= -args.maximum_clean_regression_m
        ),
        "correction_nonzero": correction_rms >= args.minimum_correction_rms_m,
        "correction_not_saturated": (
            correction_max < args.maximum_saturation_fraction * max_position
        ),
    }
    report = {
        "approved_for_formal_training": all(checks.values()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint": str(best_path.resolve()),
        "checkpoint_epoch": int(best["epoch"]),
        "protocol_sha256": best["evaluation_protocol"]["protocol_sha256"],
        "thresholds": {
            "minimum_association_prediction_gain_m": args.minimum_prediction_gain_m,
            "maximum_clean_regression_m": args.maximum_clean_regression_m,
            "minimum_correction_rms_m": args.minimum_correction_rms_m,
            "maximum_saturation_fraction": args.maximum_saturation_fraction,
        },
        "gains_m": {
            "association_prediction": association_prediction_gain,
            "association_update": association_update_gain,
            "clean_prediction": clean_prediction_gain,
            "clean_update": clean_update_gain,
        },
        "correction": {
            "position_rms_m": correction_rms,
            "position_max_m": correction_max,
            "configured_max_position_m": max_position,
        },
        "checks": checks,
        "initial": {
            "clean_position_score": position_score(clean_initial),
            "association_position_score": position_score(association_initial),
        },
        "best": {
            "clean_position_score": position_score(clean_best),
            "association_position_score": position_score(association_best),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

