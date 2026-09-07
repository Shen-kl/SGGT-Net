"""Summarize paired P4/P5 reports under the frozen robustness protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.run_association_context_robustness_v3 import (
    DEFAULT_OUTPUT,
    DEFAULT_PROTOCOL,
    read_json,
    report_is_valid,
    source_for,
)


METRICS = (
    "ospa_rmse_m",
    "matched_position_rmse_m",
    "matched_velocity_rmse_mps",
    "cardinality_rmse",
)


def run_metrics(report: dict) -> dict[str, np.ndarray]:
    values = {metric: [] for metric in METRICS}
    for run in report["runs"]:
        frames = run["per_frame"]
        values["ospa_rmse_m"].append(
            np.sqrt(np.mean([frame["ospa"] ** 2 for frame in frames]))
        )
        count = int(run["matched_state_count"])
        values["matched_position_rmse_m"].append(
            np.sqrt(float(run["matched_position_sse"]) / count)
        )
        values["matched_velocity_rmse_mps"].append(
            np.sqrt(float(run["matched_velocity_sse"]) / count)
        )
        values["cardinality_rmse"].append(np.sqrt(np.mean([
            (frame["estimated_count"] - frame["truth_count"]) ** 2
            for frame in frames
        ])))
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def pooled_metrics(report: dict) -> dict[str, float]:
    runs = report["runs"]
    frames = [frame for run in runs for frame in run["per_frame"]]
    matched_count = sum(int(run["matched_state_count"]) for run in runs)
    return {
        "ospa_rmse_m": float(np.sqrt(np.mean([frame["ospa"] ** 2 for frame in frames]))),
        "matched_position_rmse_m": float(np.sqrt(sum(
            float(run["matched_position_sse"]) for run in runs
        ) / matched_count)),
        "matched_velocity_rmse_mps": float(np.sqrt(sum(
            float(run["matched_velocity_sse"]) for run in runs
        ) / matched_count)),
        "cardinality_rmse": float(np.sqrt(np.mean([
            (frame["estimated_count"] - frame["truth_count"]) ** 2
            for frame in frames
        ]))),
    }


def verify_pair(p4: dict, p5: dict) -> None:
    if [run["seed"] for run in p4["runs"]] != [run["seed"] for run in p5["runs"]]:
        raise RuntimeError("P4/P5 reports do not use the same seeds")
    for left, right in zip(p4["runs"], p5["runs"]):
        left_counts = [frame["measurement_count"] for frame in left["per_frame"]]
        right_counts = [frame["measurement_count"] for frame in right["per_frame"]]
        if left_counts != right_counts:
            raise RuntimeError(
                f"measurement realization mismatch at seed {left['seed']}"
            )


def bootstrap_difference(
    difference: np.ndarray, rng: np.random.Generator, samples: int
) -> dict:
    draws = difference[
        rng.integers(0, len(difference), size=(samples, len(difference)))
    ].mean(axis=1)
    return {
        "observed_mean_p5_minus_p4": float(difference.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "paired_runs": int(len(difference)),
        "p4_win_fraction": float(np.mean(difference > 0.0)),
    }


def classify(interval: list[float]) -> str:
    if interval[0] > 0.0:
        return "P4_better"
    if interval[1] < 0.0:
        return "P5_better"
    return "inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    output = args.output.resolve()
    runs = protocol["monte_carlo"][f"{args.mode}_runs"]
    samples = int(protocol["comparison"]["bootstrap_samples"])
    seed = int(protocol["comparison"]["bootstrap_seed"])
    raw = {"P4": {}, "P5": {}}
    for model_id in ("P4", "P5"):
        for condition in protocol["conditions"]:
            source, _ = source_for(protocol, output, args.mode, model_id, condition)
            if not report_is_valid(source, protocol, model_id, condition, runs):
                raise RuntimeError(f"missing or invalid report: {source}")
            raw[model_id][condition["id"]] = read_json(source)

    condition_rows = []
    bootstrap = {}
    differences = {metric: [] for metric in METRICS}
    for condition_index, condition in enumerate(protocol["conditions"]):
        condition_id = condition["id"]
        p4 = raw["P4"][condition_id]
        p5 = raw["P5"][condition_id]
        verify_pair(p4, p5)
        pooled = {"P4": pooled_metrics(p4), "P5": pooled_metrics(p5)}
        run_values = {"P4": run_metrics(p4), "P5": run_metrics(p5)}
        bootstrap[condition_id] = {}
        for metric_index, metric in enumerate(METRICS):
            difference = run_values["P5"][metric] - run_values["P4"][metric]
            differences[metric].append(difference)
            result = bootstrap_difference(
                difference,
                np.random.default_rng(seed + 100 * condition_index + metric_index),
                samples,
            )
            result["classification"] = classify(result["ci95"])
            bootstrap[condition_id][metric] = result
            condition_rows.append({
                "condition": condition_id,
                "actual_pd": condition["actual_pd"],
                "actual_clutter_mean": condition["actual_clutter_mean"],
                "metric": metric,
                "P4": pooled["P4"][metric],
                "P5": pooled["P5"][metric],
                "P5_minus_P4_pooled": pooled["P5"][metric] - pooled["P4"][metric],
                **result,
            })

    macro = {}
    for metric_index, metric in enumerate(METRICS):
        arrays = differences[metric]
        rng = np.random.default_rng(seed + 10000 + metric_index)
        draws = np.stack([
            values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
            for values in arrays
        ]).mean(axis=0)
        observed = float(np.mean([values.mean() for values in arrays]))
        interval = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        macro[metric] = {
            "observed_equal_condition_mean_p5_minus_p4": observed,
            "ci95": interval,
            "classification": classify(interval),
            "conditions": len(arrays),
            "paired_runs_per_condition": runs,
        }

    high_ambiguity = ("pd_0p80_clutter_3", "pd_0p99_clutter_15")
    primary = protocol["comparison"]["primary_metrics"]
    supported_degradation = any(
        bootstrap[condition][metric]["classification"] == "P4_better"
        for condition in high_ambiguity for metric in primary
    )
    positive_macro = [
        metric for metric in primary if macro[metric]["classification"] == "P4_better"
    ]
    negative_macro = [
        metric for metric in primary if macro[metric]["classification"] == "P5_better"
    ]
    retain = bool((positive_macro and not negative_macro) or supported_degradation)
    decision = {
        "result": "retain_explicit_association_context" if retain else "select_simpler_P5_architecture",
        "positive_macro_primary_metrics": positive_macro,
        "negative_macro_primary_metrics": negative_macro,
        "supported_P5_degradation_in_high_ambiguity_condition": supported_degradation,
        "scope": (
            "Only the explicit detection-mass, entropy, and candidate-spread feature encoder is at issue. "
            "BP association and innovation/covariance quantities required by the tracker are not removed."
        ),
    }
    summary = {
        "protocol": str(protocol_path),
        "difference_definition": protocol["comparison"]["difference_definition"],
        "condition_bootstrap": bootstrap,
        "equal_condition_macro_bootstrap": macro,
        "decision": decision,
    }
    summary_root = output / args.mode / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "association_context_comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (summary_root / "association_context_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=condition_rows[0].keys())
        writer.writeheader()
        writer.writerows(condition_rows)
    print(json.dumps({"macro": macro, "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
