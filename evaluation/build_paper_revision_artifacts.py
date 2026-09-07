"""Audit formal CGIC results and build paper-ready tabular artifacts.

This script is intentionally read-only with respect to checkpoints and raw result
files.  It fails closed: missing files, non-finite numbers, inconsistent sample
counts, or checkpoint/source hash mismatches prevent a passing audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "results" / "formal_association_cgic"
BP_ROOT = ROOT / "results" / "bp_tracker_robustness"
DEFAULT_OUTPUT = ROOT / "results" / "paper_revision_20260819"

SPLITS = ("val", "iid_test", "ood_test")
EVENTS = ("all", "steady", "switch", "strong", "unclassified")
METRICS = (
    "position_prediction",
    "position_update",
    "velocity_prediction",
    "velocity_update",
)
MODEL_FILES = {
    "P0_cv": (FORMAL_ROOT, "P0_cv"),
    "P1_imm": (FORMAL_ROOT, "P1_imm"),
    "P2_temporal": (FORMAL_ROOT / "seed_123", "P2_temporal"),
    "P3_own_only": (FORMAL_ROOT / "seed_123", "P3_own_only"),
    "P4_full": (FORMAL_ROOT / "seed_123", "P4_full"),
    "P5_no_association_context": (
        FORMAL_ROOT / "seed_123",
        "P5_no_association_context",
    ),
}
LEARNED_MODELS = tuple(name for name in MODEL_FILES if name.startswith("P2") or name.startswith("P3") or name.startswith("P4") or name.startswith("P5"))
EXPECTED_TRAJECTORIES = {
    "val": 200,
    "iid_test": 300,
    "ood_test": 400,
    "val_association": 200,
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.sources: dict[str, str] = {}

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def source(self, path: Path) -> None:
        if not path.exists():
            self.error(f"Missing source: {path}")
            return
        self.sources[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, audit: Audit) -> dict[str, Any]:
    audit.source(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        audit.error(f"Cannot parse {path}: {exc}")
        return {}
    check_finite(data, str(path), audit)
    return data


def check_finite(value: Any, location: str, audit: Audit) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        audit.error(f"Non-finite value at {location}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{location}.{key}", audit)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_finite(item, f"{location}[{index}]", audit)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_value(block: dict[str, Any], metric: str) -> float | None:
    aliases = {
        "position_prediction": ("position_prediction_rmse", "location_rmse_prediction"),
        "position_update": ("position_update_rmse", "location_rmse_update"),
        "velocity_prediction": ("velocity_prediction_rmse", "velocity_rmse_prediction"),
        "velocity_update": ("velocity_update_rmse", "velocity_rmse_update"),
    }
    for key in aliases[metric]:
        if key in block:
            return float(block[key])
    return None


def all_count(report: dict[str, Any]) -> int:
    global_block = report.get("global", {})
    if "count" in global_block:
        return int(global_block["count"])
    if "evaluated_prediction_points" in global_block:
        return int(global_block["evaluated_prediction_points"])
    return int(report.get("events", {}).get("all", {}).get("count", 0))


def event_block(report: dict[str, Any], event: str) -> dict[str, Any]:
    if event == "all":
        return report.get("events", {}).get("all", report.get("global", {}))
    return report.get("events", {}).get(event, {})


def audit_tracking_report(
    path: Path,
    model_name: str,
    split: str,
    report: dict[str, Any],
    audit: Audit,
) -> None:
    if not report:
        return
    protocol = report.get("protocol", {})
    expected_trajectories = EXPECTED_TRAJECTORIES[split]
    declared = int(protocol.get("trajectory_count", -1))
    actual = len(report.get("trajectories", []))
    if declared != expected_trajectories or actual != expected_trajectories:
        audit.error(
            f"{path}: trajectory count declared/actual/expected = "
            f"{declared}/{actual}/{expected_trajectories}"
        )
    count = all_count(report)
    if count <= 0:
        audit.error(f"{path}: no evaluated points")
    events = report.get("events", {})
    event_sum = sum(int(events.get(name, {}).get("count", 0)) for name in EVENTS[1:])
    if event_sum != count:
        audit.error(f"{path}: event count sum {event_sum} != all count {count}")
    for metric in METRICS:
        if metric_value(event_block(report, "all"), metric) is None:
            audit.error(f"{path}: missing global metric {metric}")

    checkpoint = protocol.get("checkpoint")
    expected_hash = protocol.get("checkpoint_sha256")
    if checkpoint and expected_hash:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            audit.error(f"{path}: checkpoint does not exist: {checkpoint_path}")
        else:
            actual_hash = sha256(checkpoint_path)
            if actual_hash != expected_hash:
                audit.error(
                    f"{path}: checkpoint hash mismatch {actual_hash} != {expected_hash}"
                )
    if model_name in LEARNED_MODELS and protocol.get("association_augmentation", False) != split.endswith("association"):
        audit.error(f"{path}: association flag does not match split label {split}")


def tracking_rows(audit: Audit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for model_name, (folder, stem) in MODEL_FILES.items():
        available_splits = list(SPLITS)
        if model_name in LEARNED_MODELS:
            available_splits.append("val_association")
        for split in available_splits:
            path = folder / f"{stem}_{split}.json"
            report = load_json(path, audit)
            audit_tracking_report(path, model_name, split, report, audit)
            if not report:
                continue
            block = event_block(report, "all")
            protocol = report.get("protocol", {})
            row = {
                "model": model_name,
                "seed": 123 if model_name in LEARNED_MODELS else "deterministic",
                "split": split,
                "trajectory_count": protocol.get("trajectory_count"),
                "evaluated_points": all_count(report),
                "position_prediction_rmse_m": metric_value(block, "position_prediction"),
                "position_update_rmse_m": metric_value(block, "position_update"),
                "velocity_prediction_rmse_mps": metric_value(block, "velocity_prediction"),
                "velocity_update_rmse_mps": metric_value(block, "velocity_update"),
                "milliseconds_per_valid_node_step": report.get("timing", {}).get(
                    "milliseconds_per_valid_node_step"
                ),
                "checkpoint_epoch": protocol.get("checkpoint_epoch"),
                "checkpoint_sha256": protocol.get("checkpoint_sha256"),
            }
            main_rows.append(row)
            for event in EVENTS:
                event_data = event_block(report, event)
                if not event_data:
                    continue
                event_rows.append(
                    {
                        "model": model_name,
                        "seed": row["seed"],
                        "split": split,
                        "event": event,
                        "count": event_data.get("count", row["evaluated_points"] if event == "all" else None),
                        "position_prediction_rmse_m": metric_value(event_data, "position_prediction"),
                        "position_update_rmse_m": metric_value(event_data, "position_update"),
                        "velocity_prediction_rmse_mps": metric_value(event_data, "velocity_prediction"),
                        "velocity_update_rmse_mps": metric_value(event_data, "velocity_update"),
                        "paired_temporal_position_rmse_m": event_data.get("paired_temporal_position_rmse"),
                        "paired_corrected_position_rmse_m": event_data.get("paired_expert_position_rmse"),
                        "paired_position_improvement_percent": event_data.get("paired_position_improvement_percent"),
                        "paired_position_win_rate": event_data.get("paired_expert_position_win_rate"),
                        "position_correction_norm_mean_m": event_data.get("paired_position_correction_norm_mean"),
                        "position_ideal_norm_mean_m": event_data.get("paired_position_ideal_norm_mean"),
                        "position_correction_cosine_mean": event_data.get("paired_position_correction_cosine_mean"),
                        "velocity_correction_norm_mean_mps": event_data.get("paired_velocity_correction_norm_mean"),
                        "velocity_ideal_norm_mean_mps": event_data.get("paired_velocity_ideal_norm_mean"),
                        "velocity_correction_cosine_mean": event_data.get("paired_velocity_correction_cosine_mean"),
                    }
                )
    return main_rows, event_rows


def bootstrap_rows(audit: Audit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    folder = FORMAL_ROOT / "seed_123"
    comparisons = ("P2_temporal", "P3_own_only", "P5_no_association_context")
    for baseline in comparisons:
        for split in (*SPLITS, "val_association"):
            path = folder / f"bootstrap_P4_full_over_{baseline}_{split}.json"
            report = load_json(path, audit)
            if not report:
                continue
            protocol = report.get("protocol", {})
            if protocol.get("bootstrap_unit") != "trajectory" or not protocol.get("paired_resampling"):
                audit.error(f"{path}: bootstrap is not paired at trajectory level")
            if int(protocol.get("bootstrap_samples", 0)) != 10000:
                audit.error(f"{path}: expected 10000 bootstrap samples")
            for event, metrics in report.get("comparison", {}).items():
                for metric, values in metrics.items():
                    ci = values.get("improvement_95ci", [None, None])
                    rows.append(
                        {
                            "right_model": "P4_full",
                            "left_model": baseline,
                            "split": split,
                            "event": event,
                            "metric": metric,
                            "trajectory_count": values.get("trajectory_count"),
                            "point_count": values.get("point_count"),
                            "left_rmse": values.get("left_rmse"),
                            "right_rmse": values.get("right_rmse"),
                            "right_improvement": values.get("right_improvement"),
                            "ci95_low": ci[0],
                            "ci95_high": ci[1],
                            "probability_improvement_positive": values.get(
                                "probability_improvement_positive"
                            ),
                            "trajectory_win_fraction": values.get("trajectory_win_fraction"),
                            "trajectory_gain_median": values.get("trajectory_gain_median"),
                        }
                    )
    return rows


def nis_rows(audit: Audit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    folder = FORMAL_ROOT / "seed_123"
    for model in ("P2_temporal", "P4_full"):
        for split in (*SPLITS, "val_association"):
            path = folder / f"{model}_nis_{split}.json"
            report = load_json(path, audit)
            if not report:
                continue
            if float(report.get("reconstruction_max_error_m", math.inf)) > 1e-4:
                audit.error(f"{path}: NIS reconstruction error exceeds 1e-4 m")
            for event, event_data in report.get("events", {}).items():
                for source, key in (
                    ("current", "current_nis_calibration"),
                    ("causal_history", "causal_history_nis_calibration"),
                ):
                    calibration = event_data.get(key, {})
                    coverage = calibration.get("coverage", {})
                    rows.append(
                        {
                            "model": model,
                            "split": split,
                            "event": event,
                            "nis_source": source,
                            "count": calibration.get("count"),
                            "observed_mean": calibration.get("observed_mean"),
                            "mean_ratio": calibration.get("mean_ratio"),
                            "pit_mean": calibration.get("pit_mean"),
                            "pit_ks_uniform": calibration.get("pit_ks_uniform"),
                            "coverage_mae": calibration.get("coverage_mae"),
                            "coverage_p50": coverage.get("p50", {}).get("observed_below"),
                            "coverage_p90": coverage.get("p90", {}).get("observed_below"),
                            "coverage_p95": coverage.get("p95", {}).get("observed_below"),
                            "coverage_p99": coverage.get("p99", {}).get("observed_below"),
                        }
                    )
    return rows


def bp_rows(audit: Audit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("seed123_P2_temporal", "seed123_P4_full"):
        for pd in (0.75, 0.85, 0.95):
            pd_tag = str(pd).replace(".", "p")
            for clutter in (10, 20, 30):
                folder = BP_ROOT / model
                path = folder / f"pd_{pd_tag}_clutter_{clutter}_SUMMARY.json"
                report = load_json(path, audit)
                if not report:
                    continue
                protocol = report.get("protocol", {})
                summary = report.get("summary", {})
                if int(protocol.get("trajectory_count", -1)) != 3 or int(protocol.get("rollout_count", -1)) != 60:
                    audit.error(f"{path}: expected 3 trajectories and 60 rollouts")
                for source in protocol.get("sources", []):
                    source_path = Path(source["path"])
                    audit.source(source_path)
                    if source_path.exists() and sha256(source_path) != source.get("sha256"):
                        audit.error(f"{path}: BP scene source hash mismatch: {source_path}")
                rows.append(
                    {
                        "model": model.replace("seed123_", ""),
                        "seed": 123,
                        "pd": pd,
                        "clutter_mean": clutter,
                        "trajectory_count": protocol.get("trajectory_count"),
                        "rollout_count": protocol.get("rollout_count"),
                        "frame_count": summary.get("frame_count"),
                        "matched_state_count": summary.get("matched_state_count"),
                        "matched_position_rmse_m": summary.get("matched_position_rmse"),
                        "matched_velocity_rmse_mps": summary.get("matched_velocity_rmse"),
                        "ospa": summary.get("mean_ospa"),
                        "ospa_localization": summary.get("mean_ospa_localization"),
                        "ospa_cardinality": summary.get("mean_ospa_cardinality"),
                        "estimated_count_mae": summary.get("estimated_count_mae"),
                        "false_track_count": summary.get("mean_false_track_count"),
                        "missed_target_count": summary.get("mean_missed_target_count"),
                        "nis_mean": summary.get("matched_state_nis_mean"),
                        "nis_95_coverage": summary.get("matched_state_nis_95_coverage"),
                        "fragmentation_per_rollout": summary.get("fragmentation_per_rollout", {}).get("mean"),
                        "identity_switches_per_rollout": summary.get("identity_switches_per_rollout", {}).get("mean"),
                        "milliseconds_per_frame": summary.get("milliseconds_per_frame"),
                    }
                )
    return rows


def complexity_rows(audit: Audit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("P2_temporal", "P4_full"):
        path = FORMAL_ROOT / f"scaling_seed123_{model}.json"
        report = load_json(path, audit)
        if not report:
            continue
        metadata = report.get("model", {})
        for item in report.get("scaling", []):
            rows.append(
                {
                    "model": model,
                    "parameter_count": metadata.get("parameter_count"),
                    "trainable_parameter_count": metadata.get("trainable_parameter_count"),
                    **item,
                }
            )
    return rows


def add_relative_gains(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["split"]): row for row in main_rows}
    gains: list[dict[str, Any]] = []
    for split in (*SPLITS, "val_association"):
        right = lookup.get(("P4_full", split))
        if not right:
            continue
        for left_model in ("P0_cv", "P1_imm", "P2_temporal", "P3_own_only", "P5_no_association_context"):
            left = lookup.get((left_model, split))
            if not left:
                continue
            for metric in (
                "position_prediction_rmse_m",
                "position_update_rmse_m",
                "velocity_prediction_rmse_mps",
                "velocity_update_rmse_mps",
            ):
                left_value = float(left[metric])
                right_value = float(right[metric])
                gains.append(
                    {
                        "right_model": "P4_full",
                        "left_model": left_model,
                        "split": split,
                        "metric": metric,
                        "left_value": left_value,
                        "right_value": right_value,
                        "absolute_improvement": left_value - right_value,
                        "relative_improvement_percent": 100.0 * (left_value - right_value) / left_value,
                    }
                )
    return gains


def build_markdown(
    main_rows: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    bp: list[dict[str, Any]],
    audit: Audit,
) -> str:
    lookup = {(row["model"], row["split"]): row for row in main_rows}
    lines = [
        "# Formal Result Audit (2026-08-19)",
        "",
        f"Status: **{'PASS' if not audit.errors else 'FAIL'}**",
        "",
        "## Scope",
        "",
        f"- Tracking reports: {len(main_rows)}",
        f"- Trajectory-level bootstrap rows: {len(bootstrap)}",
        f"- BP robustness condition rows: {len(bp)}",
        f"- Hashed raw sources: {len(audit.sources)}",
        "- Formal learned-model claims currently use seed 123 only.",
        "",
        "## Main tracking result (seed 123)",
        "",
        "| Split | Model | Pos. pred. (m) | Pos. update (m) | Vel. pred. (m/s) | Vel. update (m/s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in (*SPLITS, "val_association"):
        for model in MODEL_FILES:
            row = lookup.get((model, split))
            if not row:
                continue
            lines.append(
                f"| {split} | {model} | {row['position_prediction_rmse_m']:.3f} | "
                f"{row['position_update_rmse_m']:.3f} | {row['velocity_prediction_rmse_mps']:.3f} | "
                f"{row['velocity_update_rmse_mps']:.3f} |"
            )

    macro: dict[str, dict[str, float]] = {}
    for model in ("P2_temporal", "P4_full"):
        selected = [row for row in bp if row["model"] == model]
        macro[model] = {
            key: sum(float(row[key]) for row in selected) / len(selected)
            for key in ("matched_position_rmse_m", "matched_velocity_rmse_mps", "ospa", "nis_95_coverage")
        }
    lines.extend(
        [
            "",
            "## BP robustness macro-average (9 Pd/clutter conditions)",
            "",
            "| Model | Matched pos. RMSE (m) | Matched vel. RMSE (m/s) | OSPA | NIS 95% coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("P2_temporal", "P4_full"):
        values = macro.get(model, {})
        if values:
            lines.append(
                f"| {model} | {values['matched_position_rmse_m']:.3f} | "
                f"{values['matched_velocity_rmse_mps']:.3f} | {values['ospa']:.3f} | "
                f"{values['nis_95_coverage']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- P4 improves position prediction consistently relative to P2 in the current seed-123 IID/OOD/validation evaluations; use the paired bootstrap table for uncertainty.",
            "- P4 versus P5 is small and mixed, so association-context benefit must not be claimed as established.",
            "- BP robustness is a difficult detection/clutter stress test. The current matched-state NIS coverage is not calibrated to 95%, and P4 slightly worsens its macro-average coverage.",
            "- Complexity replication is timing-only and excludes loading, data I/O, BP association, and measurement update.",
            "- The zero F1 field in learned reports is a disabled legacy structural metric, not evidence of structure-prediction failure.",
            "",
            "## Audit messages",
            "",
        ]
    )
    if audit.errors:
        lines.extend(f"- ERROR: {message}" for message in audit.errors)
    if audit.warnings:
        lines.extend(f"- WARNING: {message}" for message in audit.warnings)
    if not audit.errors and not audit.warnings:
        lines.append("- No audit errors or warnings.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    tables = output / "tables"
    audit = Audit()

    protocol_path = ROOT / "results" / "formal_association_cgic_protocol_v1.json"
    audit.source(protocol_path)
    tracking, events = tracking_rows(audit)
    bootstrap = bootstrap_rows(audit)
    nis = nis_rows(audit)
    bp = bp_rows(audit)
    complexity = complexity_rows(audit)
    gains = add_relative_gains(tracking)

    expected_counts = {
        "tracking_reports": 22,
        "bootstrap_rows": 12 * 5 * 4,
        "nis_rows": 8 * 5 * 2,
        "bp_conditions": 18,
        "complexity_rows": 8,
    }
    actual_counts = {
        "tracking_reports": len(tracking),
        "bootstrap_rows": len(bootstrap),
        "nis_rows": len(nis),
        "bp_conditions": len(bp),
        "complexity_rows": len(complexity),
    }
    for key, expected in expected_counts.items():
        if actual_counts[key] != expected:
            audit.error(f"{key}: expected {expected}, found {actual_counts[key]}")

    write_csv(
        tables / "main_tracking.csv",
        tracking,
        (
            "model", "seed", "split", "trajectory_count", "evaluated_points",
            "position_prediction_rmse_m", "position_update_rmse_m",
            "velocity_prediction_rmse_mps", "velocity_update_rmse_mps",
            "milliseconds_per_valid_node_step", "checkpoint_epoch", "checkpoint_sha256",
        ),
    )
    write_csv(
        tables / "event_tracking.csv",
        events,
        (
            "model", "seed", "split", "event", "count",
            "position_prediction_rmse_m", "position_update_rmse_m",
            "velocity_prediction_rmse_mps", "velocity_update_rmse_mps",
            "paired_temporal_position_rmse_m", "paired_corrected_position_rmse_m",
            "paired_position_improvement_percent", "paired_position_win_rate",
            "position_correction_norm_mean_m", "position_ideal_norm_mean_m",
            "position_correction_cosine_mean", "velocity_correction_norm_mean_mps",
            "velocity_ideal_norm_mean_mps", "velocity_correction_cosine_mean",
        ),
    )
    write_csv(
        tables / "p4_relative_gains.csv",
        gains,
        (
            "right_model", "left_model", "split", "metric", "left_value",
            "right_value", "absolute_improvement", "relative_improvement_percent",
        ),
    )
    write_csv(
        tables / "paired_bootstrap.csv",
        bootstrap,
        (
            "right_model", "left_model", "split", "event", "metric",
            "trajectory_count", "point_count", "left_rmse", "right_rmse",
            "right_improvement", "ci95_low", "ci95_high",
            "probability_improvement_positive", "trajectory_win_fraction",
            "trajectory_gain_median",
        ),
    )
    write_csv(
        tables / "nis_calibration.csv",
        nis,
        (
            "model", "split", "event", "nis_source", "count", "observed_mean",
            "mean_ratio", "pit_mean", "pit_ks_uniform", "coverage_mae",
            "coverage_p50", "coverage_p90", "coverage_p95", "coverage_p99",
        ),
    )
    write_csv(
        tables / "bp_robustness.csv",
        bp,
        (
            "model", "seed", "pd", "clutter_mean", "trajectory_count", "rollout_count",
            "frame_count", "matched_state_count", "matched_position_rmse_m",
            "matched_velocity_rmse_mps", "ospa", "ospa_localization", "ospa_cardinality",
            "estimated_count_mae", "false_track_count", "missed_target_count", "nis_mean",
            "nis_95_coverage", "fragmentation_per_rollout", "identity_switches_per_rollout",
            "milliseconds_per_frame",
        ),
    )
    write_csv(
        tables / "complexity_scaling.csv",
        complexity,
        (
            "model", "parameter_count", "trainable_parameter_count", "target_count",
            "edge_count_last_frame", "mean_ms", "std_ms", "median_ms", "p95_ms",
            "mean_us_per_target", "peak_cuda_memory_mib",
        ),
    )

    output.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "date": "2026-08-19",
        "status": "pass" if not audit.errors else "fail",
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "source_sha256": dict(sorted(audit.sources.items())),
    }
    (output / "RESULT_AUDIT.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "RESULT_AUDIT.md").write_text(
        build_markdown(tracking, bootstrap, bp, audit), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **actual_counts, "errors": len(audit.errors)}, indent=2))
    return 1 if audit.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
