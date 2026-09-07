"""Summarize and plot the fixed-Q scene02 ablation and robustness suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "paper_scene02_fixed_q_protocol_v2.json"
DEFAULT_RESULTS = ROOT / "results" / "paper_scene02_fixed_q_20260824"
MODEL_IDS = ("P2", "P3", "P4", "P5")
SCALAR_FRAME_METRICS = (
    "ospa",
    "ospa_localization",
    "ospa_cardinality",
    "ospa_full_state",
    "truth_count",
    "estimated_count",
    "measurement_count",
    "matched_count",
    "false_track_count",
    "missed_target_count",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_ospa_rms(run: dict) -> float:
    values = np.asarray([frame["ospa"] for frame in run["per_frame"]], dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def run_state_rmse(run: dict, sse_key: str) -> float:
    count = int(run["matched_state_count"])
    return float(np.sqrt(float(run[sse_key]) / count)) if count else np.nan


def distribution(values) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "std": None, "p50": None, "p95": None}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def report_summary(report: dict) -> dict:
    runs = report["runs"]
    frame_ospa = np.asarray(
        [[frame["ospa"] for frame in run["per_frame"]] for run in runs], dtype=float
    )
    total_matched = sum(int(run["matched_state_count"]) for run in runs)
    position_sse = sum(float(run["matched_position_sse"]) for run in runs)
    velocity_sse = sum(float(run["matched_velocity_sse"]) for run in runs)
    nis_sum = sum(float(run["matched_state_nis_sum"]) for run in runs)
    nll_sum = sum(float(run["matched_state_nll_sum"]) for run in runs)
    nis_95_count = sum(int(run["matched_state_nis_95_count"]) for run in runs)
    return {
        "monte_carlo_runs": len(runs),
        "legacy_ospa_curve_rms_m": float(
            np.sqrt(np.mean(np.square(frame_ospa.mean(axis=0))))
        ),
        "run_ospa_rms_m": distribution([run_ospa_rms(run) for run in runs]),
        "matched_position_global_rmse_m": (
            float(np.sqrt(position_sse / total_matched)) if total_matched else None
        ),
        "run_position_rmse_m": distribution(
            [run_state_rmse(run, "matched_position_sse") for run in runs]
        ),
        "matched_velocity_global_rmse_mps": (
            float(np.sqrt(velocity_sse / total_matched)) if total_matched else None
        ),
        "run_velocity_rmse_mps": distribution(
            [run_state_rmse(run, "matched_velocity_sse") for run in runs]
        ),
        "estimated_count_mae": distribution(
            [run["estimated_count_mae"] for run in runs]
        ),
        "mean_false_track_count": distribution(
            [run["mean_false_track_count"] for run in runs]
        ),
        "mean_missed_target_count": distribution(
            [run["mean_missed_target_count"] for run in runs]
        ),
        "fragmentation_count": distribution(
            [run["fragmentation_count"] for run in runs]
        ),
        "identity_switch_count": distribution(
            [run["identity_switch_count"] for run in runs]
        ),
        "matched_state_nis_mean": nis_sum / total_matched if total_matched else None,
        "matched_state_nll_mean": nll_sum / total_matched if total_matched else None,
        "matched_state_nis_95_coverage": (
            nis_95_count / total_matched if total_matched else None
        ),
        "matched_state_count": total_matched,
        "milliseconds_per_frame": distribution(
            [run["milliseconds_per_frame"] for run in runs]
        ),
    }


def paired_bootstrap(reference, candidate, seed: int, samples: int = 20000) -> dict:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(candidate)
    gain = reference[valid] - candidate[valid]
    if not len(gain):
        return {"paired_runs": 0, "mean_gain": None, "ci95": [None, None], "win_fraction": None}
    rng = np.random.default_rng(seed)
    resampled = gain[rng.integers(0, len(gain), size=(samples, len(gain)))].mean(axis=1)
    return {
        "paired_runs": int(len(gain)),
        "mean_gain": float(gain.mean()),
        "ci95": [float(value) for value in np.quantile(resampled, [0.025, 0.975])],
        "win_fraction": float(np.mean(gain > 0)),
    }


def paired_report(reference: dict, candidate: dict, seed: int) -> dict:
    reference_runs = reference["runs"]
    candidate_runs = candidate["runs"]
    if [run["seed"] for run in reference_runs] != [run["seed"] for run in candidate_runs]:
        raise RuntimeError("Paired comparison has different Monte Carlo seeds")
    reference_counts = [
        [frame["measurement_count"] for frame in run["per_frame"]]
        for run in reference_runs
    ]
    candidate_counts = [
        [frame["measurement_count"] for frame in run["per_frame"]]
        for run in candidate_runs
    ]
    if reference_counts != candidate_counts:
        raise RuntimeError("Paired comparison has different measurement realizations")
    return {
        "ospa_rms_m": paired_bootstrap(
            [run_ospa_rms(run) for run in reference_runs],
            [run_ospa_rms(run) for run in candidate_runs],
            seed,
        ),
        "position_rmse_m": paired_bootstrap(
            [run_state_rmse(run, "matched_position_sse") for run in reference_runs],
            [run_state_rmse(run, "matched_position_sse") for run in candidate_runs],
            seed + 1,
        ),
        "velocity_rmse_mps": paired_bootstrap(
            [run_state_rmse(run, "matched_velocity_sse") for run in reference_runs],
            [run_state_rmse(run, "matched_velocity_sse") for run in candidate_runs],
            seed + 2,
        ),
    }


def aggregate_curves(report: dict) -> dict:
    runs = report["runs"]
    frame_count = len(runs[0]["per_frame"])
    curves = {}
    for metric in SCALAR_FRAME_METRICS:
        matrix = np.asarray(
            [[frame[metric] for frame in run["per_frame"]] for run in runs], dtype=float
        )
        curves[metric] = {
            "mean": matrix.mean(axis=0),
            "std": matrix.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(frame_count),
            "valid_runs": np.full(frame_count, len(runs), dtype=int),
        }
    for label, sse_key in (
        ("matched_position_rmse", "position_sse"),
        ("matched_velocity_rmse", "velocity_sse"),
    ):
        matrix = np.full((len(runs), frame_count), np.nan, dtype=float)
        sse = np.zeros((len(runs), frame_count), dtype=float)
        counts = np.zeros((len(runs), frame_count), dtype=float)
        for run_index, run in enumerate(runs):
            for frame_index, frame in enumerate(run["per_frame"]):
                count = int(frame["matched_state_count"])
                sse[run_index, frame_index] = float(frame[sse_key])
                counts[run_index, frame_index] = count
                if count:
                    matrix[run_index, frame_index] = np.sqrt(float(frame[sse_key]) / count)
        valid = np.sum(np.isfinite(matrix), axis=0)
        mean = np.divide(
            np.nansum(matrix, axis=0), valid, out=np.full(frame_count, np.nan), where=valid > 0
        )
        centered = np.where(np.isfinite(matrix), matrix - mean, 0.0)
        std = np.sqrt(
            np.divide(
                np.sum(centered ** 2, axis=0),
                valid - 1,
                out=np.zeros(frame_count),
                where=valid > 1,
            )
        )
        total_count = counts.sum(axis=0)
        global_rmse = np.sqrt(
            np.divide(sse.sum(axis=0), total_count, out=np.full(frame_count, np.nan), where=total_count > 0)
        )
        curves[label] = {
            "mean": mean,
            "std": std,
            "global_rmse": global_rmse,
            "valid_runs": valid,
        }
    return curves


def json_values(array) -> list:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(array)]


def write_curve_artifacts(
    curve_root: Path, main_curves: dict, robustness_curves: dict, historical: dict
) -> None:
    curve_root.mkdir(parents=True, exist_ok=True)
    long_path = curve_root / "curves_long.csv"
    fields = ("suite", "condition", "model", "frame", "metric", "mean", "std", "valid_runs")
    with long_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for suite, groups in (("main", main_curves), ("robustness", robustness_curves)):
            for condition, model_curves in groups.items():
                model = condition if suite == "main" else "P4"
                for metric, values in model_curves.items():
                    for frame, mean in enumerate(values["mean"]):
                        writer.writerow({
                            "suite": suite,
                            "condition": "standard" if suite == "main" else condition,
                            "model": model,
                            "frame": frame,
                            "metric": metric,
                            "mean": "" if not np.isfinite(mean) else float(mean),
                            "std": "" if not np.isfinite(values["std"][frame]) else float(values["std"][frame]),
                            "valid_runs": int(values["valid_runs"][frame]),
                        })
        for model, values in historical.items():
            for frame, mean in enumerate(values["mean"]):
                writer.writerow({
                    "suite": "historical",
                    "condition": "legacy_paper_protocol",
                    "model": model,
                    "frame": frame,
                    "metric": "ospa",
                    "mean": float(mean),
                    "std": float(values["std"][frame]),
                    "valid_runs": values["runs"],
                })

    payload = {"main": {}, "robustness": {}, "historical": {}}
    arrays = {}
    for suite, groups in (("main", main_curves), ("robustness", robustness_curves)):
        for group, curves in groups.items():
            payload[suite][group] = {}
            for metric, values in curves.items():
                payload[suite][group][metric] = {
                    key: json_values(array) for key, array in values.items()
                }
                safe = f"{suite}__{group}__{metric}"
                for key, array in values.items():
                    arrays[f"{safe}__{key}"] = np.asarray(array)
    for model, values in historical.items():
        payload["historical"][model] = {
            "protocol_compatible_with_current": False,
            "runs": values["runs"],
            "ospa_mean": json_values(values["mean"]),
            "ospa_std": json_values(values["std"]),
        }
        arrays[f"historical__{model}__ospa_mean"] = values["mean"]
        arrays[f"historical__{model}__ospa_std"] = values["std"]
    (curve_root / "CURVES.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(curve_root / "curves.npz", **arrays)


def load_historical(manifest: dict) -> dict:
    values = {}
    for model, metadata in manifest["frozen_historical_baselines"].items():
        with Path(metadata["source"]).open("rb") as handle:
            raw = pickle.load(handle)
        matrix = np.asarray(raw["ospa_error_set_monte"], dtype=float)
        values[model] = {
            "mean": matrix.mean(axis=0),
            "std": matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1]),
            "runs": len(matrix),
        }
    return values


def plot_band(axis, curve: dict, label: str, color=None) -> None:
    mean = np.asarray(curve["mean"], dtype=float)
    std = np.asarray(curve["std"], dtype=float)
    frames = np.arange(len(mean))
    line = axis.plot(frames, mean, label=label, color=color, linewidth=1.7)[0]
    axis.fill_between(
        frames, np.maximum(mean - std, 0.0), mean + std,
        color=line.get_color(), alpha=0.12,
    )


def finish_plot(axis, ylabel: str) -> None:
    axis.set_xlabel("Frame")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def save_plots(
    plot_root: Path, manifest: dict, main_curves: dict, robustness_curves: dict, historical: dict
) -> None:
    plot_root.mkdir(parents=True, exist_ok=True)
    labels = {model: manifest["models"][model]["label"] for model in MODEL_IDS}
    for metric, ylabel, filename in (
        ("ospa", "OSPA / m", "ablation_ospa_curve.png"),
        ("matched_position_rmse", "Matched position RMSE / m", "ablation_position_rmse_curve.png"),
        ("matched_velocity_rmse", "Matched velocity RMSE / (m/s)", "ablation_velocity_rmse_curve.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model in MODEL_IDS:
            plot_band(axis, main_curves[model][metric], f"{model}: {labels[model]}")
        finish_plot(axis, ylabel)
        figure.savefig(plot_root / filename, dpi=220)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plot_band(axis, main_curves["P4"]["ospa"], "P4 current protocol", "black")
    for model, curve in historical.items():
        plot_band(axis, {"mean": curve["mean"], "std": curve["std"]}, f"{model} (legacy protocol)")
    finish_plot(axis, "OSPA / m")
    figure.savefig(plot_root / "comparison_ospa_curve_protocol_qualified.png", dpi=220)
    plt.close(figure)

    pd_conditions = ["pd_0p99_clutter_2", "pd_0p95_clutter_2", "pd_0p85_clutter_2"]
    clutter_conditions = ["pd_0p99_clutter_2", "pd_0p99_clutter_10", "pd_0p99_clutter_20"]
    for conditions, metric, ylabel, filename in (
        (pd_conditions, "ospa", "OSPA / m", "robustness_pd_ospa_curve.png"),
        (clutter_conditions, "ospa", "OSPA / m", "robustness_clutter_ospa_curve.png"),
        (pd_conditions, "matched_position_rmse", "Matched position RMSE / m", "robustness_pd_position_curve.png"),
        (clutter_conditions, "matched_position_rmse", "Matched position RMSE / m", "robustness_clutter_position_curve.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for condition in conditions:
            metadata = next(item for item in manifest["robustness"]["conditions"] if item["id"] == condition)
            label = f"Pd={metadata['actual_pd']:.2f}, clutter={metadata['actual_clutter_mean']:g}"
            plot_band(axis, robustness_curves[condition][metric], label)
        finish_plot(axis, ylabel)
        figure.savefig(plot_root / filename, dpi=220)
        plt.close(figure)


def flat_summary(model: str, summary: dict) -> dict:
    return {
        "model": model,
        "monte_carlo_runs": summary["monte_carlo_runs"],
        "ospa_curve_rms_m": summary["legacy_ospa_curve_rms_m"],
        "run_ospa_rms_mean_m": summary["run_ospa_rms_m"]["mean"],
        "run_ospa_rms_std_m": summary["run_ospa_rms_m"]["std"],
        "position_global_rmse_m": summary["matched_position_global_rmse_m"],
        "velocity_global_rmse_mps": summary["matched_velocity_global_rmse_mps"],
        "estimated_count_mae_mean": summary["estimated_count_mae"]["mean"],
        "false_track_count_mean": summary["mean_false_track_count"]["mean"],
        "missed_target_count_mean": summary["mean_missed_target_count"]["mean"],
        "fragmentation_mean": summary["fragmentation_count"]["mean"],
        "identity_switch_mean": summary["identity_switch_count"]["mean"],
        "nis_mean": summary["matched_state_nis_mean"],
        "nis_95_coverage": summary["matched_state_nis_95_coverage"],
        "nll_mean": summary["matched_state_nll_mean"],
        "milliseconds_per_frame_mean": summary["milliseconds_per_frame"]["mean"],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(
    results: Path, protocol_name: str, protocol_path: Path
) -> None:
    records = []
    for relative_root in ("formal", "summary", "curves", "plots", "logs"):
        for path in sorted((results / relative_root).rglob("*")):
            if not path.is_file() or path.name == "ARTIFACT_MANIFEST.json":
                continue
            records.append({
                "path": path.relative_to(results).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    for name in ("RESULTS_README.md", "formal_STATUS.json", "formal_robustness_STATUS.json"):
        path = results / name
        if path.exists():
            records.append({
                "path": path.relative_to(results).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    (results / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "protocol": protocol_name,
                "protocol_manifest": {
                    "path": str(protocol_path.resolve()),
                    "sha256": file_sha256(protocol_path),
                },
                "artifact_count": len(records),
                "artifacts": records,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    results = args.results.resolve()
    reports = {
        model: read_json(results / "formal" / f"{model}.json") for model in MODEL_IDS
    }
    conditions = manifest["robustness"]["conditions"]
    robustness_reports = {}
    for condition in conditions:
        if condition.get("reuse_main_report"):
            robustness_reports[condition["id"]] = reports[manifest["robustness"]["model"]]
        else:
            robustness_reports[condition["id"]] = read_json(
                results / "formal" / "robustness" / f"{condition['id']}.json"
            )

    summaries = {model: report_summary(report) for model, report in reports.items()}
    robustness_summaries = {
        condition: report_summary(report) for condition, report in robustness_reports.items()
    }
    bootstrap = {
        f"P4_over_{model}": paired_report(reports[model], reports["P4"], 20260824 + index * 10)
        for index, model in enumerate(("P2", "P3", "P5"))
    }
    main_curves = {model: aggregate_curves(report) for model, report in reports.items()}
    robustness_curves = {
        condition: aggregate_curves(report) for condition, report in robustness_reports.items()
    }
    historical = load_historical(manifest)

    summary_root = results / "summary"
    curve_root = results / "curves"
    plot_root = results / "plots"
    summary_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": manifest["protocol_name"],
        "measurement_and_tracker": manifest["measurement_and_tracker"],
        "fixed_process_noise": manifest["fixed_process_noise"],
        "ablation": summaries,
        "paired_bootstrap": bootstrap,
        "robustness_definition": manifest["robustness"],
        "robustness": robustness_summaries,
        "historical_baselines": manifest["frozen_historical_baselines"],
        "historical_baseline_limitation": (
            "Frozen descriptive scene02 values from the prior paper protocol. They are not "
            "measurement-paired and are not protocol-compatible with the current Pd=0.99, "
            "clutter=2, azimuth-noise=0.01-degree experiment."
        ),
    }
    (summary_root / "SUMMARY.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    ablation_rows = [flat_summary(model, summaries[model]) for model in MODEL_IDS]
    with (summary_root / "ablation_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)

    robustness_rows = []
    for condition in conditions:
        row = flat_summary("P4", robustness_summaries[condition["id"]])
        row = {
            "condition": condition["id"],
            "actual_pd": condition["actual_pd"],
            "actual_clutter_mean": condition["actual_clutter_mean"],
            "nominal_pd": manifest["robustness"]["nominal_pd"],
            "nominal_clutter_mean": manifest["robustness"]["nominal_clutter_mean"],
            **row,
        }
        robustness_rows.append(row)
    with (summary_root / "robustness_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(robustness_rows[0]))
        writer.writeheader()
        writer.writerows(robustness_rows)

    with (summary_root / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "method", "ospa_rmse_m", "monte_carlo_runs", "protocol_compatible_with_current",
            "measurement_paired_with_p4", "source",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "method": "P4 complete CGIC",
            "ospa_rmse_m": summaries["P4"]["legacy_ospa_curve_rms_m"],
            "monte_carlo_runs": summaries["P4"]["monte_carlo_runs"],
            "protocol_compatible_with_current": True,
            "measurement_paired_with_p4": True,
            "source": str(results / "formal" / "P4.json"),
        })
        for method, metadata in manifest["frozen_historical_baselines"].items():
            writer.writerow({
                "method": method,
                "ospa_rmse_m": metadata["paper_reported_ospa_rmse_m"],
                "monte_carlo_runs": metadata["monte_carlo_runs"],
                "protocol_compatible_with_current": False,
                "measurement_paired_with_p4": False,
                "source": metadata["source"],
            })

    write_curve_artifacts(curve_root, main_curves, robustness_curves, historical)
    save_plots(plot_root, manifest, main_curves, robustness_curves, historical)
    write_artifact_manifest(results, manifest["protocol_name"], args.manifest)
    print(json.dumps({model: flat_summary(model, summaries[model]) for model in MODEL_IDS}, indent=2))
    print(f"summary={summary_root / 'SUMMARY.json'}")
    print(f"curves={curve_root / 'CURVES.json'}")
    print(f"plots={plot_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
