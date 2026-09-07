"""Select CV/IMM process-noise settings on validation data only."""

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data_v3 import GroupTargetTrackingDatasetV3
from evaluation.evaluate_physical_baseline_v3 import (
    bucket_report,
    merge_bucket,
    run_trajectory,
)


def evaluate(dataset, model, parameters, args):
    config = SimpleNamespace(
        model=model,
        input_length=args.input_length,
        delta_t=args.delta_t,
        cv_acceleration_std=parameters.get("cv_acceleration_std", 2.0),
        imm_low_acceleration_std=parameters.get(
            "imm_low_acceleration_std", 0.5
        ),
        imm_high_acceleration_std=parameters.get(
            "imm_high_acceleration_std", 5.0
        ),
        imm_transition=parameters.get(
            "imm_transition", [0.97, 0.03, 0.05, 0.95]
        ),
        imm_initial_probability=args.imm_initial_probability,
    )
    aggregate = defaultdict(float)
    stop = len(dataset) if args.max_trajectories <= 0 else min(
        len(dataset), args.max_trajectories
    )
    for trajectory_id in range(stop):
        merge_bucket(aggregate, run_trajectory(dataset[trajectory_id], config)["all"])
    metrics = bucket_report(aggregate)
    return {
        "parameters": parameters,
        "trajectory_count": stop,
        "prediction_position_rmse_m": metrics["position_prediction_rmse"],
        "update_position_rmse_m": metrics["position_update_rmse"],
        "position_score_m": (
            metrics["position_prediction_rmse"]
            + metrics["position_update_rmse"]
        ),
        "prediction_velocity_rmse_mps": metrics["velocity_prediction_rmse"],
        "update_velocity_rmse_mps": metrics["velocity_update_rmse"],
        "measurement_nis_mean": metrics["measurement_nis_mean"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--input_length", type=int, default=8)
    parser.add_argument("--delta_t", type=float, default=2.0)
    parser.add_argument(
        "--cv_acceleration_grid", type=float, nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--imm_low_acceleration_grid", type=float, nargs="+",
        default=[0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--imm_high_acceleration_grid", type=float, nargs="+",
        default=[3.0, 5.0, 8.0],
    )
    parser.add_argument(
        "--imm_stay_probability_grid", type=float, nargs="+",
        default=[0.90, 0.95, 0.98],
    )
    parser.add_argument(
        "--imm_initial_probability", type=float, nargs=2,
        default=[0.8, 0.2],
    )
    args = parser.parse_args()
    for values in (
        args.cv_acceleration_grid,
        args.imm_low_acceleration_grid,
        args.imm_high_acceleration_grid,
    ):
        if any(value <= 0 or not math.isfinite(value) for value in values):
            raise ValueError("all acceleration standard deviations must be positive")
    if any(not 0.5 <= value < 1.0 for value in args.imm_stay_probability_grid):
        raise ValueError("IMM stay probabilities must lie in [0.5, 1)")

    dataset = GroupTargetTrackingDatasetV3(
        args.data_dir, "val", load_structure=False,
        load_analysis_metadata=False,
    )
    cv_rows = [
        evaluate(dataset, "cv", {"cv_acceleration_std": value}, args)
        for value in args.cv_acceleration_grid
    ]
    imm_rows = []
    for low, high, stay in itertools.product(
        args.imm_low_acceleration_grid,
        args.imm_high_acceleration_grid,
        args.imm_stay_probability_grid,
    ):
        if low >= high:
            continue
        # Symmetric switching makes the single persistence hyperparameter
        # auditable and avoids a large post-hoc validation search.
        transition = [stay, 1.0 - stay, 1.0 - stay, stay]
        imm_rows.append(evaluate(dataset, "imm", {
            "imm_low_acceleration_std": low,
            "imm_high_acceleration_std": high,
            "imm_transition": transition,
        }, args))
    cv_rows.sort(key=lambda row: row["position_score_m"])
    imm_rows.sort(key=lambda row: row["position_score_m"])
    report = {
        "protocol": {
            "version": 1,
            "selection_split": "val",
            "test_splits_used_for_selection": False,
            "dataset_root": str(Path(args.data_dir).resolve()),
            "trajectory_count": cv_rows[0]["trajectory_count"],
            "selection_metric": "prediction position RMSE + update position RMSE",
            "tie_break": "first lexicographic grid occurrence",
        },
        "selected": {"cv": cv_rows[0], "imm": imm_rows[0]},
        "grid": {"cv": cv_rows, "imm": imm_rows},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
