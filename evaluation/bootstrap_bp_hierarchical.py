"""Paired hierarchical bootstrap for the formal BP Pd-by-clutter matrix.

The sampling hierarchy is scene -> paired Monte Carlo realization.  P2 and P4
always share the same resampled scene and random seed.  This prevents treating
60 repeated rollouts from only three trajectories as 60 independent scenes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BP_ROOT = ROOT / "results" / "bp_tracker_robustness"
OUTPUT_ROOT = ROOT / "results" / "paper_revision_20260819"
MODELS = ("seed123_P2_temporal", "seed123_P4_full")
PDS = (0.75, 0.85, 0.95)
CLUTTERS = (10, 20, 30)
SCENES = (1, 2, 3)
METRICS = (
    "matched_position_rmse",
    "matched_velocity_rmse",
    "mean_ospa",
    "matched_state_nis_95_coverage",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def scene_path(model: str, pd: float, clutter: int, scene: int) -> Path:
    pd_tag = str(pd).replace(".", "p")
    return BP_ROOT / model / f"pd_{pd_tag}_clutter_{clutter}_scene{scene:02d}.json"


def load_pair(pd: float, clutter: int, scene: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    paths = [scene_path(model, pd, clutter, scene) for model in MODELS]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    protocols = [report["protocol"] for report in reports]
    for protocol in protocols:
        if float(protocol["actual_pd"]) != pd or int(protocol["actual_clutter_mean"]) != clutter:
            raise ValueError(f"Condition mismatch in scene {scene}: Pd={pd}, clutter={clutter}")
        if int(protocol["monte_carlo_runs"]) != 20:
            raise ValueError(f"Expected 20 Monte Carlo runs in {paths}")
    if protocols[0]["trajectory_sha256"] != protocols[1]["trajectory_sha256"]:
        raise ValueError(f"P2/P4 trajectory mismatch for scene {scene}")
    runs = [report["runs"] for report in reports]
    pair_keys = [
        [(int(run["monte_carlo_index"]), int(run["seed"])) for run in model_runs]
        for model_runs in runs
    ]
    if pair_keys[0] != pair_keys[1]:
        raise ValueError(f"P2/P4 Monte Carlo pairing mismatch for scene {scene}")
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}
    return runs[0], runs[1], hashes


def aggregate(runs: list[dict[str, Any]]) -> dict[str, float]:
    matched_count = sum(float(run["matched_state_count"]) for run in runs)
    frame_count = sum(float(run["frame_count"]) for run in runs)
    if matched_count <= 0 or frame_count <= 0:
        raise ValueError("Invalid zero sufficient-statistic count")
    return {
        "matched_position_rmse": math.sqrt(
            sum(float(run["matched_position_sse"]) for run in runs) / matched_count
        ),
        "matched_velocity_rmse": math.sqrt(
            sum(float(run["matched_velocity_sse"]) for run in runs) / matched_count
        ),
        "mean_ospa": sum(float(run["mean_ospa"]) * float(run["frame_count"]) for run in runs)
        / frame_count,
        "matched_state_nis_95_coverage": sum(
            float(run["matched_state_nis_95_count"]) for run in runs
        )
        / matched_count,
    }


def improvement(metric: str, p2_value: float, p4_value: float) -> float:
    if metric == "matched_state_nis_95_coverage":
        return p4_value - p2_value
    return p2_value - p4_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    sources: dict[str, str] = {}
    condition_reports: list[dict[str, Any]] = []
    condition_samples: dict[tuple[float, int, str], list[float]] = {}
    for pd in PDS:
        for clutter in CLUTTERS:
            scene_pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
            for scene in SCENES:
                p2_runs, p4_runs, hashes = load_pair(pd, clutter, scene)
                scene_pairs.append((p2_runs, p4_runs))
                sources.update(hashes)

            observed_p2 = aggregate([run for pair in scene_pairs for run in pair[0]])
            observed_p4 = aggregate([run for pair in scene_pairs for run in pair[1]])
            samples = {metric: [] for metric in METRICS}
            for _ in range(args.bootstrap_samples):
                sampled_p2: list[dict[str, Any]] = []
                sampled_p4: list[dict[str, Any]] = []
                for _scene_draw in SCENES:
                    scene_index = rng.randrange(len(scene_pairs))
                    p2_runs, p4_runs = scene_pairs[scene_index]
                    for _run_draw in range(len(p2_runs)):
                        run_index = rng.randrange(len(p2_runs))
                        sampled_p2.append(p2_runs[run_index])
                        sampled_p4.append(p4_runs[run_index])
                aggregate_p2 = aggregate(sampled_p2)
                aggregate_p4 = aggregate(sampled_p4)
                for metric in METRICS:
                    samples[metric].append(
                        improvement(metric, aggregate_p2[metric], aggregate_p4[metric])
                    )

            metric_reports: dict[str, Any] = {}
            for metric in METRICS:
                observed_gain = improvement(metric, observed_p2[metric], observed_p4[metric])
                values = samples[metric]
                run_gains = [
                    improvement(metric, float(p2[metric]), float(p4[metric]))
                    for pair in scene_pairs
                    for p2, p4 in zip(pair[0], pair[1])
                ]
                scene_gains = []
                for p2_runs, p4_runs in scene_pairs:
                    p2_scene = aggregate(p2_runs)
                    p4_scene = aggregate(p4_runs)
                    scene_gains.append(improvement(metric, p2_scene[metric], p4_scene[metric]))
                metric_reports[metric] = {
                    "P2": observed_p2[metric],
                    "P4": observed_p4[metric],
                    "P4_improvement": observed_gain,
                    "improvement_95ci": [percentile(values, 0.025), percentile(values, 0.975)],
                    "probability_improvement_positive": sum(value > 0 for value in values) / len(values),
                    "paired_run_win_fraction": sum(value > 0 for value in run_gains) / len(run_gains),
                    "paired_scene_win_fraction": sum(value > 0 for value in scene_gains) / len(scene_gains),
                    "paired_scene_gains": scene_gains,
                }
                condition_samples[(pd, clutter, metric)] = values
            condition_reports.append({"pd": pd, "clutter_mean": clutter, "metrics": metric_reports})

    macro: dict[str, Any] = {}
    for metric in METRICS:
        observed_condition_gains = [
            report["metrics"][metric]["P4_improvement"] for report in condition_reports
        ]
        macro_samples = [
            sum(condition_samples[(pd, clutter, metric)][index] for pd in PDS for clutter in CLUTTERS)
            / (len(PDS) * len(CLUTTERS))
            for index in range(args.bootstrap_samples)
        ]
        macro[metric] = {
            "condition_macro_improvement": sum(observed_condition_gains) / len(observed_condition_gains),
            "improvement_95ci": [percentile(macro_samples, 0.025), percentile(macro_samples, 0.975)],
            "probability_improvement_positive": sum(value > 0 for value in macro_samples)
            / len(macro_samples),
            "condition_win_fraction": sum(value > 0 for value in observed_condition_gains)
            / len(observed_condition_gains),
        }

    payload = {
        "protocol": {
            "version": 1,
            "date": "2026-08-19",
            "comparison": "P4_full over P2_temporal",
            "positive_improvement": {
                "error_metrics": "P2 minus P4",
                "nis_95_coverage": "P4 minus P2",
            },
            "sampling_unit": "scene then paired Monte Carlo realization",
            "scene_count_per_condition": len(SCENES),
            "monte_carlo_runs_per_scene": 20,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "condition_macro": "unweighted mean over the nine fixed Pd-by-clutter conditions",
        },
        "conditions": condition_reports,
        "macro": macro,
        "source_sha256": dict(sorted(sources.items())),
    }
    statistics = OUTPUT_ROOT / "statistics"
    tables = OUTPUT_ROOT / "tables"
    statistics.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    output_json = statistics / "bp_hierarchical_bootstrap.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_rows = []
    for report in condition_reports:
        for metric, values in report["metrics"].items():
            csv_rows.append(
                {
                    "scope": "condition",
                    "pd": report["pd"],
                    "clutter_mean": report["clutter_mean"],
                    "metric": metric,
                    "P2": values["P2"],
                    "P4": values["P4"],
                    "P4_improvement": values["P4_improvement"],
                    "ci95_low": values["improvement_95ci"][0],
                    "ci95_high": values["improvement_95ci"][1],
                    "probability_improvement_positive": values["probability_improvement_positive"],
                    "win_fraction": values["paired_scene_win_fraction"],
                }
            )
    for metric, values in macro.items():
        csv_rows.append(
            {
                "scope": "macro",
                "pd": "",
                "clutter_mean": "",
                "metric": metric,
                "P2": "",
                "P4": "",
                "P4_improvement": values["condition_macro_improvement"],
                "ci95_low": values["improvement_95ci"][0],
                "ci95_high": values["improvement_95ci"][1],
                "probability_improvement_positive": values["probability_improvement_positive"],
                "win_fraction": values["condition_win_fraction"],
            }
        )
    with (tables / "bp_hierarchical_bootstrap.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = list(csv_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps({"output": str(output_json), "macro": macro}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
