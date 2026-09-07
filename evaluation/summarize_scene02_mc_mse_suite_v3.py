"""Create aggregate-only Monte Carlo MSE artifacts for the scene02 suite."""

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
DEFAULT_MANIFEST = ROOT / "results" / "paper_scene02_mc_mse_protocol_v3.json"
DEFAULT_OUTPUT = ROOT / "results" / "paper_scene02_mc_mse_20260826"
DEFAULT_TEMP = ROOT / ".tmp" / "paper_scene02_mc_mse_20260826"
MODELS = ("P2", "P3", "P4", "P5")
DIRECT_METRICS = (
    "ospa",
    "ospa_localization",
    "ospa_cardinality",
    "ospa_full_state",
    "false_track_count",
    "missed_target_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def direct_mse_curve(report: dict, key: str) -> dict:
    matrix = np.asarray(
        [[frame[key] for frame in run["per_frame"]] for run in report["runs"]],
        dtype=float,
    )
    curve = np.mean(np.square(matrix), axis=0)
    return {
        "mc_mse_curve": curve,
        "final_rmse": float(np.sqrt(np.mean(curve))),
    }


def count_error_curve(report: dict) -> dict:
    matrix = np.asarray(
        [
            [frame["estimated_count"] - frame["truth_count"] for frame in run["per_frame"]]
            for run in report["runs"]
        ],
        dtype=float,
    )
    curve = np.mean(np.square(matrix), axis=0)
    return {"mc_mse_curve": curve, "final_rmse": float(np.sqrt(np.mean(curve)))}


def matched_mse_curve(report: dict, sse_key: str) -> dict:
    runs = report["runs"]
    frame_count = len(runs[0]["per_frame"])
    sse = np.zeros(frame_count, dtype=float)
    count = np.zeros(frame_count, dtype=float)
    for run in runs:
        for frame_index, frame in enumerate(run["per_frame"]):
            sse[frame_index] += float(frame[sse_key])
            count[frame_index] += int(frame["matched_state_count"])
    curve = np.divide(
        sse, count, out=np.full(frame_count, np.nan, dtype=float), where=count > 0
    )
    return {
        "mc_mse_curve": curve,
        "final_rmse": float(np.sqrt(sse.sum() / count.sum())),
    }


def aggregate_report(report: dict, source_path: Path, protocol_name: str) -> dict:
    curves = {key: direct_mse_curve(report, key) for key in DIRECT_METRICS}
    curves["count_error"] = count_error_curve(report)
    curves["matched_position"] = matched_mse_curve(report, "position_sse")
    curves["matched_velocity"] = matched_mse_curve(report, "velocity_sse")
    runs = report["runs"]
    matched = sum(int(run["matched_state_count"]) for run in runs)
    summary = {
        "monte_carlo_runs": len(runs),
        "frame_count": len(runs[0]["per_frame"]),
        "ospa_final_rmse_m": curves["ospa"]["final_rmse"],
        "position_final_rmse_m": curves["matched_position"]["final_rmse"],
        "velocity_final_rmse_mps": curves["matched_velocity"]["final_rmse"],
        "count_final_rmse": curves["count_error"]["final_rmse"],
        "estimated_count_mae_mean": float(
            np.mean([run["estimated_count_mae"] for run in runs])
        ),
        "false_track_count_mean": float(
            np.mean([run["mean_false_track_count"] for run in runs])
        ),
        "missed_target_count_mean": float(
            np.mean([run["mean_missed_target_count"] for run in runs])
        ),
        "fragmentation_mean": float(np.mean([run["fragmentation_count"] for run in runs])),
        "identity_switch_mean": float(
            np.mean([run["identity_switch_count"] for run in runs])
        ),
        "matched_state_nis_mean": (
            sum(float(run["matched_state_nis_sum"]) for run in runs) / matched
            if matched else None
        ),
        "matched_state_nis_95_coverage": (
            sum(int(run["matched_state_nis_95_count"]) for run in runs) / matched
            if matched else None
        ),
        "milliseconds_per_frame_mean": float(
            np.mean([run["milliseconds_per_frame"] for run in runs])
        ),
    }
    return {
        "protocol": protocol_name,
        "source_report": str(source_path.resolve()),
        "source_report_sha256": sha256_file(source_path),
        "source_protocol": report["protocol"],
        "storage": {
            "raw_per_run_curves_saved": False,
            "curve": "per-frame Monte Carlo MSE",
            "final": "global RMSE",
        },
        "summary": summary,
        "curves": {
            metric: {
                "mc_mse_curve": [
                    float(value) if np.isfinite(value) else None
                    for value in values["mc_mse_curve"]
                ],
                "final_rmse": values["final_rmse"],
            }
            for metric, values in curves.items()
        },
    }


def run_metric(report: dict, metric: str) -> np.ndarray:
    if metric == "ospa":
        return np.asarray(
            [
                np.sqrt(np.mean([frame["ospa"] ** 2 for frame in run["per_frame"]]))
                for run in report["runs"]
            ],
            dtype=float,
        )
    sse_key = "matched_position_sse" if metric == "position" else "matched_velocity_sse"
    return np.asarray(
        [
            np.sqrt(float(run[sse_key]) / int(run["matched_state_count"]))
            if int(run["matched_state_count"]) else np.nan
            for run in report["runs"]
        ],
        dtype=float,
    )


def paired_bootstrap(reference: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    gain = reference[valid] - candidate[valid]
    rng = np.random.default_rng(seed)
    samples = gain[
        rng.integers(0, len(gain), size=(20000, len(gain)))
    ].mean(axis=1)
    return {
        "paired_runs": int(len(gain)),
        "mean_gain": float(gain.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "win_fraction": float(np.mean(gain > 0)),
    }


def paired_comparison(reference: dict, candidate: dict, seed: int) -> dict:
    if [run["seed"] for run in reference["runs"]] != [
        run["seed"] for run in candidate["runs"]
    ]:
        raise RuntimeError("paired reports use different seeds")
    reference_counts = [
        [frame["measurement_count"] for frame in run["per_frame"]]
        for run in reference["runs"]
    ]
    candidate_counts = [
        [frame["measurement_count"] for frame in run["per_frame"]]
        for run in candidate["runs"]
    ]
    if reference_counts != candidate_counts:
        raise RuntimeError("paired reports use different measurement realizations")
    return {
        metric: paired_bootstrap(run_metric(reference, metric), run_metric(candidate, metric), seed + index)
        for index, metric in enumerate(("ospa", "position", "velocity"))
    }


def historical_curves(manifest: dict) -> dict:
    curves = {}
    for method, metadata in manifest["frozen_historical_baselines"].items():
        source = Path(metadata["source"])
        if sha256_file(source) != metadata["sha256"]:
            raise RuntimeError(f"historical source hash mismatch: {source}")
        with source.open("rb") as handle:
            raw = pickle.load(handle)
        matrix = np.asarray(raw["ospa_error_set_monte"], dtype=float)
        mse = np.mean(np.square(matrix), axis=0)
        curves[method] = {
            "mc_mse_curve": mse,
            "final_rmse": float(np.sqrt(np.mean(mse))),
            "monte_carlo_runs": len(matrix),
        }
    return curves


def array_from_aggregate(aggregate: dict, metric: str) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else value for value in aggregate["curves"][metric]["mc_mse_curve"]],
        dtype=float,
    )


def write_curves(
    output: Path, main: dict, robustness: dict, historical: dict
) -> None:
    curve_root = output / "curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    payload = {"main": {}, "robustness": {}, "historical": {}}
    arrays = {}
    rows = []
    for suite, groups in (("main", main), ("robustness", robustness)):
        for group, aggregate in groups.items():
            payload[suite][group] = aggregate["curves"]
            for metric, values in aggregate["curves"].items():
                curve = np.asarray(
                    [np.nan if value is None else value for value in values["mc_mse_curve"]],
                    dtype=float,
                )
                prefix = f"{suite}__{group}__{metric}"
                arrays[f"{prefix}__mc_mse_curve"] = curve
                arrays[f"{prefix}__final_rmse"] = np.asarray(values["final_rmse"])
                for frame, mse in enumerate(curve):
                    rows.append({
                        "suite": suite,
                        "group": group,
                        "metric": metric,
                        "frame": frame,
                        "mc_mse": "" if not np.isfinite(mse) else float(mse),
                    })
    for method, values in historical.items():
        payload["historical"][method] = {
            "mc_mse_curve": [float(value) for value in values["mc_mse_curve"]],
            "final_rmse": values["final_rmse"],
            "monte_carlo_runs": values["monte_carlo_runs"],
        }
        prefix = f"historical__{method}__ospa"
        arrays[f"{prefix}__mc_mse_curve"] = values["mc_mse_curve"]
        arrays[f"{prefix}__final_rmse"] = np.asarray(values["final_rmse"])
        for frame, mse in enumerate(values["mc_mse_curve"]):
            rows.append({
                "suite": "historical",
                "group": method,
                "metric": "ospa",
                "frame": frame,
                "mc_mse": float(mse),
            })
    (curve_root / "MC_MSE_CURVES.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (curve_root / "mc_mse_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("suite", "group", "metric", "frame", "mc_mse"))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(curve_root / "mc_mse_curves.npz", **arrays)


def plot_curve(axis, aggregate: dict, metric: str, label: str) -> None:
    mse = array_from_aggregate(aggregate, metric)
    axis.plot(np.arange(len(mse)), np.sqrt(mse), label=label, linewidth=1.8)


def finish_plot(axis, ylabel: str) -> None:
    axis.set_xlabel("Frame")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def write_plots(output: Path, manifest: dict, main: dict, robustness: dict, historical: dict) -> None:
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in (
        ("ospa", "Monte Carlo RMSE OSPA / m", "ablation_ospa_mc_rmse.png"),
        ("matched_position", "Monte Carlo position RMSE / m", "ablation_position_mc_rmse.png"),
        ("matched_velocity", "Monte Carlo velocity RMSE / (m/s)", "ablation_velocity_mc_rmse.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model in MODELS:
            plot_curve(axis, main[model], metric, f"{model}: {manifest['models'][model]['label']}")
        finish_plot(axis, ylabel)
        figure.savefig(plot_root / filename, dpi=220)
        plt.close(figure)

    pd_conditions = ("pd_0p99_clutter_3", "pd_0p90_clutter_3", "pd_0p80_clutter_3")
    clutter_conditions = ("pd_0p99_clutter_2", "pd_0p99_clutter_10", "pd_0p99_clutter_15")
    metadata = {item["id"]: item for item in manifest["robustness"]["conditions"]}
    for conditions, metric, ylabel, filename in (
        (pd_conditions, "ospa", "Monte Carlo RMSE OSPA / m", "robustness_pd_ospa_mc_rmse.png"),
        (pd_conditions, "matched_position", "Monte Carlo position RMSE / m", "robustness_pd_position_mc_rmse.png"),
        (clutter_conditions, "ospa", "Monte Carlo RMSE OSPA / m", "robustness_clutter_ospa_mc_rmse.png"),
        (clutter_conditions, "matched_position", "Monte Carlo position RMSE / m", "robustness_clutter_position_mc_rmse.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for condition in conditions:
            item = metadata[condition]
            label = f"Pd={item['actual_pd']:.2f}, clutter={item['actual_clutter_mean']:g}"
            plot_curve(axis, robustness[condition], metric, label)
        finish_plot(axis, ylabel)
        figure.savefig(plot_root / filename, dpi=220)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plot_curve(axis, main["P4"], "ospa", "P4 current protocol")
    for method, values in historical.items():
        axis.plot(
            np.arange(len(values["mc_mse_curve"])),
            np.sqrt(values["mc_mse_curve"]),
            label=f"{method} (legacy protocol)",
            linewidth=1.5,
        )
    finish_plot(axis, "Monte Carlo RMSE OSPA / m")
    figure.savefig(plot_root / "comparison_ospa_mc_rmse_protocol_qualified.png", dpi=220)
    plt.close(figure)


def flat_row(group: str, aggregate: dict) -> dict:
    summary = aggregate["summary"]
    return {"group": group, **summary}


def write_tables(
    output: Path, manifest: dict, main: dict, robustness: dict, historical: dict, bootstrap: dict
) -> None:
    summary_root = output / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": manifest["protocol_name"],
        "curve_storage": manifest["curve_storage"],
        "measurement_and_tracker": manifest["measurement_and_tracker"],
        "fixed_process_noise": manifest["fixed_process_noise"],
        "ablation": {model: aggregate["summary"] for model, aggregate in main.items()},
        "paired_bootstrap": bootstrap,
        "robustness": {
            condition: aggregate["summary"] for condition, aggregate in robustness.items()
        },
        "historical": {
            method: {
                "final_ospa_rmse_m": values["final_rmse"],
                "monte_carlo_runs": values["monte_carlo_runs"],
                "protocol_compatible_with_current": False,
            }
            for method, values in historical.items()
        },
        "historical_limitation": "legacy paper observation protocol; not paired or protocol-matched",
    }
    (summary_root / "SUMMARY.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = [flat_row(model, main[model]) for model in MODELS]
    with (summary_root / "ablation_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    condition_metadata = {item["id"]: item for item in manifest["robustness"]["conditions"]}
    rows = []
    for condition, aggregate in robustness.items():
        item = condition_metadata[condition]
        rows.append({
            "condition": condition,
            "actual_pd": item["actual_pd"],
            "actual_clutter_mean": item["actual_clutter_mean"],
            "nominal_pd": manifest["robustness"]["nominal_pd"],
            "nominal_clutter_mean": manifest["robustness"]["nominal_clutter_mean"],
            **aggregate["summary"],
        })
    with (summary_root / "robustness_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (summary_root / "comparison_table.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("method", "ospa_final_rmse_m", "monte_carlo_runs", "protocol_compatible_with_current")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "method": "P4 complete CGIC",
            "ospa_final_rmse_m": main["P4"]["summary"]["ospa_final_rmse_m"],
            "monte_carlo_runs": main["P4"]["summary"]["monte_carlo_runs"],
            "protocol_compatible_with_current": True,
        })
        for method, values in historical.items():
            writer.writerow({
                "method": method,
                "ospa_final_rmse_m": values["final_rmse"],
                "monte_carlo_runs": values["monte_carlo_runs"],
                "protocol_compatible_with_current": False,
            })


def write_readme(output: Path, manifest: dict, main: dict, robustness: dict) -> None:
    lines = [
        "# scene02 aggregate Monte Carlo MSE suite (2026-08-26)",
        "",
        "Only per-frame Monte Carlo MSE curves and their final global RMSE are retained.",
        "No per-run error curves are stored in this result directory.",
        "",
        "## Ablation",
        "",
        "| Model | OSPA RMSE (m) | Position RMSE (m) | Velocity RMSE (m/s) | Count RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        summary = main[model]["summary"]
        lines.append(
            f"| {model} | {summary['ospa_final_rmse_m']:.4f} | "
            f"{summary['position_final_rmse_m']:.4f} | {summary['velocity_final_rmse_mps']:.4f} | "
            f"{summary['count_final_rmse']:.4f} |"
        )
    lines.extend([
        "",
        "## Robustness",
        "",
        "| Condition | OSPA RMSE (m) | Position RMSE (m) | Velocity RMSE (m/s) | Count RMSE |",
        "|---|---:|---:|---:|---:|",
    ])
    for condition in manifest["robustness"]["conditions"]:
        summary = robustness[condition["id"]]["summary"]
        lines.append(
            f"| Pd={condition['actual_pd']:.2f}, clutter={condition['actual_clutter_mean']:g} | "
            f"{summary['ospa_final_rmse_m']:.4f} | {summary['position_final_rmse_m']:.4f} | "
            f"{summary['velocity_final_rmse_mps']:.4f} | {summary['count_final_rmse']:.4f} |"
        )
    lines.extend([
        "",
        "The tracker nominal operating point remains Pd=0.99 and clutter=2 in every robustness condition.",
        "Historical baseline curves use the legacy paper observation protocol and are descriptive only.",
        "",
        "## Files",
        "",
        "- `aggregate/`: aggregate-only reports with no per-run curves.",
        "- `summary/`: ablation, comparison, robustness, and bootstrap statistics.",
        "- `curves/MC_MSE_CURVES.json`: plot-ready aggregate curves.",
        "- `curves/mc_mse_curves.csv`: one row per aggregate curve frame.",
        "- `curves/mc_mse_curves.npz`: only `mc_mse_curve` and `final_rmse` arrays.",
        "- `plots/`: figures generated from square-root MSE curves.",
    ])
    (output / "RESULTS_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact_manifest(output: Path, manifest_path: Path) -> None:
    records = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "ARTIFACT_MANIFEST.json":
            continue
        records.append({
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    (output / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "protocol_manifest": str(manifest_path.resolve()),
                "protocol_manifest_sha256": sha256_file(manifest_path),
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_main = {}
    main_sources = {}
    for model in MODELS:
        metadata = manifest["reused_source_reports"][model]
        source = ROOT / metadata["path"]
        if sha256_file(source) != metadata["sha256"]:
            raise RuntimeError(f"reused source hash mismatch: {source}")
        raw_main[model] = read_json(source)
        main_sources[model] = source
    raw_robustness = {}
    robustness_sources = {}
    for condition in manifest["robustness"]["conditions"]:
        condition_id = condition["id"]
        if condition.get("reuse") == "main_P4":
            raw_robustness[condition_id] = raw_main["P4"]
            robustness_sources[condition_id] = main_sources["P4"]
        elif condition.get("reuse") == "previous_formal_report":
            metadata = manifest["reused_source_reports"][condition_id]
            source = ROOT / metadata["path"]
            if sha256_file(source) != metadata["sha256"]:
                raise RuntimeError(f"reused robustness hash mismatch: {source}")
            raw_robustness[condition_id] = read_json(source)
            robustness_sources[condition_id] = source
        else:
            source = args.temp.resolve() / f"{condition_id}.json"
            raw_robustness[condition_id] = read_json(source)
            robustness_sources[condition_id] = source

    aggregate_root = output / "aggregate"
    main_aggregate = {}
    robustness_aggregate = {}
    for model in MODELS:
        aggregate = aggregate_report(raw_main[model], main_sources[model], manifest["protocol_name"])
        main_aggregate[model] = aggregate
        target = aggregate_root / "main" / f"{model}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")
    for condition_id, report in raw_robustness.items():
        aggregate = aggregate_report(
            report, robustness_sources[condition_id], manifest["protocol_name"]
        )
        robustness_aggregate[condition_id] = aggregate
        target = aggregate_root / "robustness" / f"{condition_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8")

    bootstrap = {
        f"P4_over_{model}": paired_comparison(raw_main[model], raw_main["P4"], 20260826 + index * 10)
        for index, model in enumerate(("P2", "P3", "P5"))
    }
    historical = historical_curves(manifest)
    write_tables(output, manifest, main_aggregate, robustness_aggregate, historical, bootstrap)
    write_curves(output, main_aggregate, robustness_aggregate, historical)
    write_plots(output, manifest, main_aggregate, robustness_aggregate, historical)
    write_readme(output, manifest, main_aggregate, robustness_aggregate)
    write_artifact_manifest(output, args.manifest)
    print(json.dumps({
        "ablation": {model: value["summary"] for model, value in main_aggregate.items()},
        "robustness": {key: value["summary"] for key, value in robustness_aggregate.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
