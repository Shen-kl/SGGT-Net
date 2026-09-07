"""Render audited paper CSV/JSON artifacts as deterministic LaTeX tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "results" / "paper_revision_20260819"
TABLE_ROOT = ARTIFACT_ROOT / "tables"
OUTPUT_ROOT = ARTIFACT_ROOT / "latex_tables"
DOC_OUTPUT_ROOT = ROOT / "docs" / "paper_revision" / "paper_tables"


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLE_ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(name: str, lines: list[str]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DOC_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    (OUTPUT_ROOT / name).write_text(content, encoding="utf-8")
    (DOC_OUTPUT_ROOT / name).write_text(content, encoding="utf-8")


def num(row: dict[str, str], key: str, digits: int = 3) -> str:
    return f"{float(row[key]):.{digits}f}"


def build_tracking_tables(main: list[dict[str, str]]) -> None:
    lookup = {(row["model"], row["split"]): row for row in main}
    models = (
        ("P0_cv", "CV"),
        ("P1_imm", "IMM"),
        ("P2_temporal", "P2 Temporal"),
        ("P3_own_only", "P3 own-only"),
        ("P4_full", "P4 full CGIC"),
        ("P5_no_association_context", "P5 w/o assoc. context"),
    )
    for kind, metric_prefix, unit in (
        ("position", "position", "m"),
        ("velocity", "velocity", "m/s"),
    ):
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{Known-identity {kind} RMSE ({unit}). Prior is evaluated before the current measurement update; posterior is evaluated after it. P0/P1 are deterministic, while P2--P5 currently report training seed 123.}}",
            rf"\label{{tab:{kind}_main}}",
            r"\begin{tabular}{lrrrrrr}",
            r"\hline",
            r"& \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{IID test} & \multicolumn{2}{c}{OOD test} \\",
            r"Method & Prior & Posterior & Prior & Posterior & Prior & Posterior \\",
            r"\hline",
        ]
        for model, label in models:
            values = []
            for split in ("val", "iid_test", "ood_test"):
                row = lookup[(model, split)]
                values.extend(
                    (
                        num(row, f"{metric_prefix}_prediction_rmse_{'m' if kind == 'position' else 'mps'}"),
                        num(row, f"{metric_prefix}_update_rmse_{'m' if kind == 'position' else 'mps'}"),
                    )
                )
            lines.append(label + " & " + " & ".join(values) + r" \\")
        lines.extend((r"\hline", r"\end{tabular}", r"\end{table*}"))
        write(f"table_{kind}_main.tex", lines)


def build_event_table(events: list[dict[str, str]], bootstrap: list[dict[str, str]]) -> None:
    lookup = {
        (row["model"], row["event"]): row
        for row in events
        if row["split"] == "val" and row["model"] in ("P2_temporal", "P4_full")
    }
    bootstrap_lookup = {
        row["event"]: row
        for row in bootstrap
        if row["left_model"] == "P2_temporal"
        and row["split"] == "val"
        and row["metric"] == "position_prediction"
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Validation position-prior RMSE by maneuver event and trajectory-paired bootstrap improvement of P4 over P2.}",
        r"\label{tab:event_position}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Event & Points & P2 & P4 & Gain [95\% CI] \\",
        r"\hline",
    ]
    for event in ("steady", "switch", "strong"):
        p2, p4, boot = lookup[("P2_temporal", event)], lookup[("P4_full", event)], bootstrap_lookup[event]
        lines.append(
            f"{event.capitalize()} & {int(float(p2['count']))} & "
            f"{num(p2, 'position_prediction_rmse_m')} & {num(p4, 'position_prediction_rmse_m')} & "
            f"{float(boot['right_improvement']):.3f} "
            f"[{float(boot['ci95_low']):.3f}, {float(boot['ci95_high']):.3f}] " + r"\\"
        )
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    write("table_event_position.tex", lines)


def build_bp_tables(bp: list[dict[str, str]], hierarchical: dict[str, object]) -> None:
    lookup = {
        (row["model"], float(row["pd"]), int(float(row["clutter_mean"]))): row
        for row in bp
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete BP-tracker stress test. Positive gains mean lower error for P4. Each condition contains three trajectories and 20 paired Monte Carlo runs per trajectory.}",
        r"\label{tab:bp_matrix}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\hline",
        r"Actual $P_D$ & Clutter & P2 pos. & P4 pos. & Pos. gain & P2 OSPA & P4 OSPA & OSPA gain \\",
        r"\hline",
    ]
    for pd in (0.75, 0.85, 0.95):
        for clutter in (10, 20, 30):
            p2 = lookup[("P2_temporal", pd, clutter)]
            p4 = lookup[("P4_full", pd, clutter)]
            pos_gain = float(p2["matched_position_rmse_m"]) - float(p4["matched_position_rmse_m"])
            ospa_gain = float(p2["ospa"]) - float(p4["ospa"])
            lines.append(
                f"{pd:.2f} & {clutter} & {num(p2, 'matched_position_rmse_m')} & "
                f"{num(p4, 'matched_position_rmse_m')} & {pos_gain:+.3f} & "
                f"{num(p2, 'ospa')} & {num(p4, 'ospa')} & {ospa_gain:+.3f} " + r"\\"
            )
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table*}"))
    write("table_bp_conditions.tex", lines)

    labels = {
        "matched_position_rmse": ("Matched position RMSE", "m"),
        "matched_velocity_rmse": ("Matched velocity RMSE", "m/s"),
        "mean_ospa": ("OSPA", ""),
        "matched_state_nis_95_coverage": ("NIS 95\% coverage", ""),
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{BP condition-macro improvements from hierarchical paired bootstrap (10,000 samples). Coverage uses P4 minus P2; other metrics use P2 minus P4.}",
        r"\label{tab:bp_bootstrap}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Metric & Gain [95\% CI] & $\Pr(\mathrm{gain}>0)$ \\",
        r"\hline",
    ]
    for key, (label, unit) in labels.items():
        values = hierarchical["macro"][key]
        gain = float(values["condition_macro_improvement"])
        low, high = (float(value) for value in values["improvement_95ci"])
        suffix = f" {unit}" if unit else ""
        lines.append(
            f"{label} & {gain:+.3f}{suffix} [{low:+.3f}, {high:+.3f}] & "
            f"{float(values['probability_improvement_positive']):.3f} " + r"\\"
        )
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    write("table_bp_hierarchical.tex", lines)


def build_complexity_table(complexity: list[dict[str, str]]) -> None:
    lookup = {
        (row["model"], int(float(row["target_count"]))): row for row in complexity
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prediction-interface scaling on an RTX 3060. Timing excludes model loading, data I/O, BP association, and measurement update.}",
        r"\label{tab:complexity}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"& \multicolumn{2}{c}{P2 Temporal} & \multicolumn{2}{c}{P4 full} \\",
        r"Targets & ms & MiB & ms & MiB \\",
        r"\hline",
    ]
    for target in (5, 10, 20, 40):
        p2, p4 = lookup[("P2_temporal", target)], lookup[("P4_full", target)]
        lines.append(
            f"{target} & {num(p2, 'mean_ms')} & {num(p2, 'peak_cuda_memory_mib', 2)} & "
            f"{num(p4, 'mean_ms')} & {num(p4, 'peak_cuda_memory_mib', 2)} " + r"\\"
        )
    lines.extend(
        (
            r"\hline",
            r"\multicolumn{5}{l}{Parameters: P2 317,540; P4 335,209 (+17,669).} \\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        )
    )
    write("table_complexity.tex", lines)


def main() -> int:
    audit = json.loads((ARTIFACT_ROOT / "RESULT_AUDIT.json").read_text(encoding="utf-8"))
    if audit["status"] != "pass":
        raise RuntimeError("Refusing to render LaTeX tables from a failed audit")
    main_rows = read_csv("main_tracking.csv")
    events = read_csv("event_tracking.csv")
    bootstrap = read_csv("paired_bootstrap.csv")
    bp = read_csv("bp_robustness.csv")
    complexity = read_csv("complexity_scaling.csv")
    hierarchical = json.loads(
        (ARTIFACT_ROOT / "statistics" / "bp_hierarchical_bootstrap.json").read_text(
            encoding="utf-8"
        )
    )
    build_tracking_tables(main_rows)
    build_event_table(events, bootstrap)
    build_bp_tables(bp, hierarchical)
    build_complexity_table(complexity)
    print(json.dumps({"tables": sorted(path.name for path in OUTPUT_ROOT.glob("*.tex"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
