"""Build deterministic, dependency-free SVG figures from audited paper tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "results" / "paper_revision_20260819"
TABLE_ROOT = ARTIFACT_ROOT / "tables"
FIGURE_ROOT = ARTIFACT_ROOT / "figures"
STAT_ROOT = ARTIFACT_ROOT / "statistics"

BLUE = "#2563eb"
BLUE_DARK = "#1e40af"
RED = "#dc2626"
GRAY = "#64748b"
LIGHT = "#e2e8f0"
BLACK = "#0f172a"


def rows(name: str) -> list[dict[str, str]]:
    with (TABLE_ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#0f172a}.small{font-size:13px}.tick{font-size:12px;fill:#475569}.label{font-size:15px;font-weight:600}.title{font-size:19px;font-weight:700}</style>',
    ]


def text(x: float, y: float, value: str, css: str = "small", anchor: str = "start", fill: str | None = None) -> str:
    color = f' fill="{fill}"' if fill else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}"{color}>{escape(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = LIGHT, width: float = 1, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dashed}/>'


def mix_with_white(color: str, strength: float) -> str:
    strength = min(1.0, max(0.0, strength))
    rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(255 + (channel - 255) * strength) for channel in rgb)
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def save_svg(name: str, content: list[str]) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    content.append("</svg>")
    (FIGURE_ROOT / name).write_text("\n".join(content) + "\n", encoding="utf-8")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_bp_heatmap(bp: list[dict[str, str]]) -> dict[str, object]:
    lookup = {
        (row["model"], f(row, "pd"), int(f(row, "clutter_mean"))): row for row in bp
    }
    conditions: list[dict[str, float]] = []
    for pd in (0.75, 0.85, 0.95):
        for clutter in (10, 20, 30):
            p2 = lookup[("P2_temporal", pd, clutter)]
            p4 = lookup[("P4_full", pd, clutter)]
            conditions.append(
                {
                    "pd": pd,
                    "clutter": clutter,
                    "position_gain_m": f(p2, "matched_position_rmse_m") - f(p4, "matched_position_rmse_m"),
                    "velocity_gain_mps": f(p2, "matched_velocity_rmse_mps") - f(p4, "matched_velocity_rmse_mps"),
                    "ospa_gain": f(p2, "ospa") - f(p4, "ospa"),
                    "coverage_change": f(p4, "nis_95_coverage") - f(p2, "nis_95_coverage"),
                }
            )

    width, height = 1000, 470
    doc = svg_start(width, height, "P4 improvement over P2 under detection and clutter stress")
    doc.append(text(500, 32, "P4 improvement over P2 under BP association stress", "title", "middle"))
    doc.append(text(500, 55, "Positive values indicate lower error for P4; each cell pools 3 scenes × 20 Monte Carlo rollouts", "tick", "middle"))
    panels = (("position_gain_m", "Matched position RMSE gain (m)"), ("ospa_gain", "OSPA gain"))
    for panel_index, (key, panel_title) in enumerate(panels):
        x0 = 95 + panel_index * 490
        y0 = 115
        cell_w, cell_h = 105, 78
        values = [float(item[key]) for item in conditions]
        scale = max(abs(value) for value in values)
        doc.append(text(x0 + 1.5 * cell_w, 88, panel_title, "label", "middle"))
        for column, clutter in enumerate((10, 20, 30)):
            doc.append(text(x0 + (column + 0.5) * cell_w, y0 - 14, str(clutter), "small", "middle"))
        doc.append(text(x0 + 1.5 * cell_w, y0 - 38, "Mean clutter count", "tick", "middle"))
        for row_index, pd in enumerate((0.75, 0.85, 0.95)):
            doc.append(text(x0 - 17, y0 + (row_index + 0.55) * cell_h, f"{pd:.2f}", "small", "end"))
            for column, clutter in enumerate((10, 20, 30)):
                item = next(item for item in conditions if item["pd"] == pd and item["clutter"] == clutter)
                value = float(item[key])
                base = BLUE if value >= 0 else RED
                fill = mix_with_white(base, 0.2 + 0.75 * abs(value) / scale)
                x = x0 + column * cell_w
                y = y0 + row_index * cell_h
                doc.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="white" stroke-width="3"/>')
                label_fill = "white" if abs(value) / scale > 0.55 else BLACK
                doc.append(text(x + cell_w / 2, y + cell_h / 2 + 5, f"{value:+.3f}", "label", "middle", label_fill))
        doc.append(text(x0 - 66, y0 + 1.5 * cell_h, "Pd", "label", "middle"))
    doc.append(text(500, 410, "Blue = improvement; red = regression. The only common regression is Pd=0.75, clutter=30.", "small", "middle"))
    save_svg("fig_bp_gain_heatmap.svg", doc)

    def macro(key: str, model: str) -> float:
        selected = [row for row in bp if row["model"] == model]
        return sum(f(row, key) for row in selected) / len(selected)

    return {
        "conditions": conditions,
        "position_win_conditions": sum(item["position_gain_m"] > 0 for item in conditions),
        "ospa_win_conditions": sum(item["ospa_gain"] > 0 for item in conditions),
        "macro": {
            model: {
                key: macro(key, model)
                for key in (
                    "matched_position_rmse_m",
                    "matched_velocity_rmse_mps",
                    "ospa",
                    "nis_95_coverage",
                )
            }
            for model in ("P2_temporal", "P4_full")
        },
    }


def build_bp_hierarchical_figure(report: dict[str, object]) -> dict[str, object]:
    macro = report["macro"]
    panels = (
        ("matched_position_rmse", "Position RMSE", "m"),
        ("matched_velocity_rmse", "Velocity RMSE", "m/s"),
        ("mean_ospa", "OSPA", ""),
        ("matched_state_nis_95_coverage", "NIS 95% coverage", ""),
    )
    width, height = 1050, 500
    doc = svg_start(width, height, "Hierarchical BP bootstrap summary")
    doc.append(text(525, 32, "BP robustness: hierarchical paired bootstrap", "title", "middle"))
    doc.append(text(525, 55, "Scene → paired Monte Carlo resampling, 10,000 samples; condition-macro improvement", "tick", "middle"))
    for index, (key, title, unit) in enumerate(panels):
        values = macro[key]
        gain = float(values["condition_macro_improvement"])
        low, high = (float(value) for value in values["improvement_95ci"])
        panel_x = 70 + (index % 2) * 505
        panel_y = 105 + (index // 2) * 180
        plot_x, plot_w = panel_x + 55, 380
        max_abs = max(abs(low), abs(high), abs(gain)) * 1.2
        if max_abs == 0:
            max_abs = 1.0
        xmap = lambda value: plot_x + (value + max_abs) / (2 * max_abs) * plot_w
        doc.append(text(panel_x + 245, panel_y - 15, title, "label", "middle"))
        doc.append(line(plot_x, panel_y + 62, plot_x + plot_w, panel_y + 62, GRAY, 1.2))
        doc.append(line(xmap(0), panel_y + 22, xmap(0), panel_y + 102, BLACK, 1.2, "5 4"))
        doc.append(line(xmap(low), panel_y + 62, xmap(high), panel_y + 62, BLUE_DARK, 4))
        doc.append(line(xmap(low), panel_y + 53, xmap(low), panel_y + 71, BLUE_DARK, 2))
        doc.append(line(xmap(high), panel_y + 53, xmap(high), panel_y + 71, BLUE_DARK, 2))
        point_color = BLUE if gain >= 0 else RED
        doc.append(f'<circle cx="{xmap(gain):.1f}" cy="{panel_y + 62:.1f}" r="7" fill="{point_color}"/>')
        suffix = f" {unit}" if unit else ""
        doc.append(text(panel_x + 245, panel_y + 125, f"{gain:+.3f}{suffix} [{low:+.3f}, {high:+.3f}]", "small", "middle"))
        doc.append(text(panel_x + 245, panel_y + 146, f"P(benefit > 0) = {float(values['probability_improvement_positive']):.3f}", "tick", "middle"))
    doc.append(text(525, 472, "Positive means lower error for RMSE/OSPA and higher coverage for NIS. Position uncertainty remains wide with only three scenes per condition.", "small", "middle"))
    save_svg("fig_bp_hierarchical_bootstrap.svg", doc)
    return macro


def build_trajectory_analysis() -> dict[str, object]:
    report_root = ROOT / "results" / "formal_association_cgic" / "seed_123"
    splits = ("val", "iid_test", "ood_test", "val_association")
    output_rows: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    for split in splits:
        reports = {
            model: json.loads((report_root / f"{model}_{split}.json").read_text(encoding="utf-8"))
            for model in ("P2_temporal", "P4_full")
        }
        trajectories = {
            model: {int(item["trajectory_id"]): item for item in report["trajectories"]}
            for model, report in reports.items()
        }
        if trajectories["P2_temporal"].keys() != trajectories["P4_full"].keys():
            raise ValueError(f"Unpaired trajectory identifiers for {split}")
        for trajectory_id in sorted(trajectories["P2_temporal"]):
            for event in ("all", "steady", "switch", "strong", "unclassified"):
                blocks = {
                    model: trajectories[model][trajectory_id]["events"].get(event, {})
                    for model in trajectories
                }
                count = int(blocks["P2_temporal"].get("count", 0))
                if count <= 0 or int(blocks["P4_full"].get("count", 0)) != count:
                    continue
                row: dict[str, object] = {
                    "split": split,
                    "trajectory_id": trajectory_id,
                    "event": event,
                    "count": count,
                }
                for metric, sse_key in (
                    ("position_prediction", "position_prediction_sse"),
                    ("position_update", "position_update_sse"),
                    ("velocity_prediction", "velocity_prediction_sse"),
                    ("velocity_update", "velocity_update_sse"),
                ):
                    p2 = math.sqrt(float(blocks["P2_temporal"][sse_key]) / count)
                    p4 = math.sqrt(float(blocks["P4_full"][sse_key]) / count)
                    row[f"P2_{metric}_rmse"] = p2
                    row[f"P4_{metric}_rmse"] = p4
                    row[f"{metric}_gain"] = p2 - p4
                output_rows.append(row)

        all_rows = [row for row in output_rows if row["split"] == split and row["event"] == "all"]
        ordered = sorted(all_rows, key=lambda row: float(row["position_prediction_gain"]))
        median_gain = percentile([float(row["position_prediction_gain"]) for row in ordered], 0.5)
        median_row = min(ordered, key=lambda row: abs(float(row["position_prediction_gain"]) - median_gain))
        selections[split] = {
            "worst": ordered[0],
            "median": median_row,
            "best": ordered[-1],
            "positive_trajectory_fraction": sum(float(row["position_prediction_gain"]) > 0 for row in ordered) / len(ordered),
        }

    table = TABLE_ROOT / "trajectory_position_gains.csv"
    fields = list(output_rows[0])
    with table.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    STAT_ROOT.mkdir(parents=True, exist_ok=True)
    (STAT_ROOT / "trajectory_case_manifest.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )

    all_by_split = {
        split: sorted(
            float(row["position_prediction_gain"])
            for row in output_rows
            if row["split"] == split and row["event"] == "all"
        )
        for split in splits
    }
    combined = [value for values in all_by_split.values() for value in values]
    xmin, xmax = percentile(combined, 0.01), percentile(combined, 0.99)
    pad = 0.08 * (xmax - xmin)
    xmin, xmax = xmin - pad, xmax + pad
    width, height = 1000, 480
    doc = svg_start(width, height, "Trajectory-level distribution of P4 position-prediction gain")
    doc.append(text(500, 32, "Trajectory-level position-prediction gain", "title", "middle"))
    doc.append(text(500, 55, "Empirical CDF of paired P2 − P4 trajectory RMSE; x-axis clipped to pooled 1st–99th percentiles", "tick", "middle"))
    x0, y0, plot_w, plot_h = 105, 95, 800, 300
    xmap = lambda value: x0 + (value - xmin) / (xmax - xmin) * plot_w
    ymap = lambda value: y0 + plot_h - value * plot_h
    for index in range(6):
        probability = index / 5
        y = ymap(probability)
        doc.append(line(x0, y, x0 + plot_w, y, LIGHT))
        doc.append(text(x0 - 10, y + 4, f"{probability:.1f}", "tick", "end"))
    if xmin <= 0 <= xmax:
        doc.append(line(xmap(0), y0, xmap(0), y0 + plot_h, BLACK, 1.5, "6 4"))
    colors = {"val": GRAY, "iid_test": BLUE, "ood_test": RED, "val_association": "#7c3aed"}
    for split in splits:
        values = all_by_split[split]
        points = []
        for index, value in enumerate(values):
            clipped = min(xmax, max(xmin, value))
            points.append((xmap(clipped), ymap((index + 1) / len(values))))
        doc.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" stroke="{colors[split]}" stroke-width="2.5"/>')
    for index in range(6):
        value = xmin + index * (xmax - xmin) / 5
        x = xmap(value)
        doc.append(text(x, y0 + plot_h + 23, f"{value:.1f}", "tick", "middle"))
    doc.append(text(500, 432, "P2 − P4 position-prediction RMSE (m)", "small", "middle"))
    legend_x = 220
    for index, split in enumerate(splits):
        x = legend_x + index * 155
        doc.append(line(x, 455, x + 22, 455, colors[split], 3))
        doc.append(text(x + 28, 459, split, "tick"))
    save_svg("fig_trajectory_gain_cdf.svg", doc)
    return selections


def build_event_gain(bootstrap: list[dict[str, str]]) -> dict[str, object]:
    selected = [
        row
        for row in bootstrap
        if row["left_model"] == "P2_temporal"
        and row["split"] == "val"
        and row["event"] in ("all", "steady", "switch", "strong")
        and row["metric"] in ("position_prediction", "position_update")
    ]
    event_order = ("all", "steady", "switch", "strong")
    width, height = 1000, 500
    doc = svg_start(width, height, "Trajectory-bootstrap position gains by motion event")
    doc.append(text(500, 32, "Position improvement by motion event (P4 over P2)", "title", "middle"))
    doc.append(text(500, 55, "Paired trajectory bootstrap, validation split, 10,000 resamples; whiskers show 95% CI", "tick", "middle"))
    panels = (("position_prediction", "Prior prediction"), ("position_update", "Posterior update"))
    summaries: dict[str, object] = {}
    for panel_index, (metric, label) in enumerate(panels):
        x0 = 135 + panel_index * 485
        y0 = 115
        plot_w, plot_h = 350, 245
        panel_rows = {row["event"]: row for row in selected if row["metric"] == metric}
        lows = [min(0.0, f(row, "ci95_low")) for row in panel_rows.values()]
        highs = [f(row, "ci95_high") for row in panel_rows.values()]
        xmin = min(lows) - 0.1
        xmax = max(highs) + 0.15
        xmap = lambda value: x0 + (value - xmin) / (xmax - xmin) * plot_w
        doc.append(text(x0 + plot_w / 2, 88, label, "label", "middle"))
        doc.append(line(xmap(0), y0 - 10, xmap(0), y0 + plot_h, GRAY, 1.5, "5 4"))
        for tick_index in range(5):
            value = xmin + tick_index * (xmax - xmin) / 4
            x = xmap(value)
            doc.append(line(x, y0 + plot_h, x, y0 + plot_h + 6, GRAY))
            doc.append(text(x, y0 + plot_h + 23, f"{value:.1f}", "tick", "middle"))
        for index, event in enumerate(event_order):
            row = panel_rows[event]
            y = y0 + 28 + index * 55
            value = f(row, "right_improvement")
            low, high = f(row, "ci95_low"), f(row, "ci95_high")
            doc.append(text(x0 - 14, y + 4, event, "small", "end"))
            doc.append(line(xmap(low), y, xmap(high), y, BLUE_DARK, 3))
            doc.append(line(xmap(low), y - 6, xmap(low), y + 6, BLUE_DARK, 2))
            doc.append(line(xmap(high), y - 6, xmap(high), y + 6, BLUE_DARK, 2))
            doc.append(f'<circle cx="{xmap(value):.1f}" cy="{y:.1f}" r="6" fill="{BLUE}"/>')
            doc.append(text(xmap(high) + 8, y + 4, f"{value:+.2f} m", "tick"))
        doc.append(text(x0 + plot_w / 2, y0 + plot_h + 48, "RMSE reduction (m)", "small", "middle"))
        summaries[metric] = {
            event: {
                "improvement_m": f(panel_rows[event], "right_improvement"),
                "ci95": [f(panel_rows[event], "ci95_low"), f(panel_rows[event], "ci95_high")],
                "trajectory_win_fraction": f(panel_rows[event], "trajectory_win_fraction"),
            }
            for event in event_order
        }
    doc.append(text(500, 468, "Strong-maneuver gains exceed steady-motion gains for both prior prediction and posterior update.", "small", "middle"))
    save_svg("fig_event_position_gain.svg", doc)
    return summaries


def build_nis_figure(nis: list[dict[str, str]]) -> dict[str, object]:
    selected = [
        row
        for row in nis
        if row["split"] == "val"
        and row["nis_source"] == "current"
        and row["event"] in ("all", "steady", "switch", "strong")
    ]
    lookup = {(row["model"], row["event"]): row for row in selected}
    width, height = 1000, 500
    doc = svg_start(width, height, "NIS calibration by motion event")
    doc.append(text(500, 32, "NIS calibration by motion event", "title", "middle"))
    doc.append(text(500, 55, "Validation split; measurement dimension is two", "tick", "middle"))
    events = ("all", "steady", "switch", "strong")
    panels = (("mean_ratio", "Mean NIS / expected mean", 1.0, 0.75, 1.15), ("coverage_p95", "Observed 95% coverage", 0.95, 0.90, 1.00))
    for panel_index, (key, label, target, ymin, ymax) in enumerate(panels):
        x0 = 105 + panel_index * 490
        y0, plot_w, plot_h = 105, 360, 270
        ymap = lambda value: y0 + plot_h - (value - ymin) / (ymax - ymin) * plot_h
        doc.append(text(x0 + plot_w / 2, 84, label, "label", "middle"))
        for index in range(5):
            value = ymin + index * (ymax - ymin) / 4
            y = ymap(value)
            doc.append(line(x0, y, x0 + plot_w, y, LIGHT))
            doc.append(text(x0 - 10, y + 4, f"{value:.2f}", "tick", "end"))
        doc.append(line(x0, ymap(target), x0 + plot_w, ymap(target), BLACK, 1.5, "6 4"))
        group_w = plot_w / len(events)
        bar_w = 27
        for event_index, event in enumerate(events):
            center = x0 + (event_index + 0.5) * group_w
            for model_index, (model, color) in enumerate((("P2_temporal", GRAY), ("P4_full", BLUE))):
                value = f(lookup[(model, event)], key)
                x = center + (model_index - 0.5) * (bar_w + 5)
                y = ymap(value)
                base = ymap(ymin)
                doc.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w}" height="{base - y:.1f}" fill="{color}"/>')
            doc.append(text(center, y0 + plot_h + 22, event, "tick", "middle"))
        if panel_index == 0:
            doc.append(f'<rect x="{x0 + 8}" y="{y0 + 10}" width="14" height="14" fill="{GRAY}"/>')
            doc.append(text(x0 + 28, y0 + 22, "P2", "tick"))
            doc.append(f'<rect x="{x0 + 75}" y="{y0 + 10}" width="14" height="14" fill="{BLUE}"/>')
            doc.append(text(x0 + 95, y0 + 22, "P4", "tick"))
    doc.append(text(500, 460, "P4 is well calibrated on clean validation, but this does not carry over to the BP clutter/missed-detection stress test.", "small", "middle"))
    save_svg("fig_nis_calibration.svg", doc)
    return {
        model: {
            event: {
                "mean_ratio": f(lookup[(model, event)], "mean_ratio"),
                "coverage_p95": f(lookup[(model, event)], "coverage_p95"),
            }
            for event in events
        }
        for model in ("P2_temporal", "P4_full")
    }


def build_complexity_figure(complexity: list[dict[str, str]]) -> dict[str, object]:
    lookup = {(row["model"], int(f(row, "target_count"))): row for row in complexity}
    targets = (5, 10, 20, 40)
    width, height = 1000, 470
    doc = svg_start(width, height, "Controlled model prediction scaling")
    doc.append(text(500, 32, "Controlled prediction-interface scaling", "title", "middle"))
    doc.append(text(500, 55, "Disconnected replication of a real five-target causal window; BP, I/O, loading, and measurement update excluded", "tick", "middle"))
    panels = (("mean_ms", "Mean latency (ms)", 40.0, 52.0), ("peak_cuda_memory_mib", "Peak CUDA memory (MiB)", 0.0, 55.0))
    for panel_index, (key, label, ymin, ymax) in enumerate(panels):
        x0 = 100 + panel_index * 490
        y0, plot_w, plot_h = 105, 355, 250
        xmap = lambda value: x0 + (math.log2(value) - math.log2(5)) / 3 * plot_w
        ymap = lambda value: y0 + plot_h - (value - ymin) / (ymax - ymin) * plot_h
        doc.append(text(x0 + plot_w / 2, 84, label, "label", "middle"))
        for index in range(5):
            value = ymin + index * (ymax - ymin) / 4
            y = ymap(value)
            doc.append(line(x0, y, x0 + plot_w, y, LIGHT))
            doc.append(text(x0 - 10, y + 4, f"{value:.0f}", "tick", "end"))
        for target in targets:
            x = xmap(target)
            doc.append(text(x, y0 + plot_h + 22, str(target), "tick", "middle"))
        for model, color in (("P2_temporal", GRAY), ("P4_full", BLUE)):
            points = [(xmap(target), ymap(f(lookup[(model, target)], key))) for target in targets]
            doc.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" stroke="{color}" stroke-width="3"/>')
            for x, y in points:
                doc.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        doc.append(text(x0 + plot_w / 2, y0 + plot_h + 48, "Target count", "small", "middle"))
    doc.append(f'<rect x="410" y="420" width="14" height="14" fill="{GRAY}"/>')
    doc.append(text(430, 432, "P2 Temporal", "tick"))
    doc.append(f'<rect x="535" y="420" width="14" height="14" fill="{BLUE}"/>')
    doc.append(text(555, 432, "P4 Full", "tick"))
    save_svg("fig_complexity_scaling.svg", doc)
    return {
        model: {
            "parameter_count": int(f(lookup[(model, 5)], "parameter_count")),
            "scaling": {
                str(target): {
                    "mean_ms": f(lookup[(model, target)], "mean_ms"),
                    "peak_cuda_memory_mib": f(lookup[(model, target)], "peak_cuda_memory_mib"),
                }
                for target in targets
            },
        }
        for model in ("P2_temporal", "P4_full")
    }


def tracking_claims(main: list[dict[str, str]], bootstrap: list[dict[str, str]]) -> dict[str, object]:
    lookup = {(row["model"], row["split"]): row for row in main}
    split_claims: dict[str, object] = {}
    for split in ("val", "iid_test", "ood_test", "val_association"):
        p2, p4 = lookup[("P2_temporal", split)], lookup[("P4_full", split)]
        metrics: dict[str, object] = {}
        for metric in (
            "position_prediction_rmse_m",
            "position_update_rmse_m",
            "velocity_prediction_rmse_mps",
            "velocity_update_rmse_mps",
        ):
            left, right = f(p2, metric), f(p4, metric)
            metrics[metric] = {
                "P2": left,
                "P4": right,
                "absolute_improvement": left - right,
                "relative_improvement_percent": 100 * (left - right) / left,
            }
        split_claims[split] = metrics

    ablation: dict[str, object] = {}
    for baseline in ("P2_temporal", "P3_own_only", "P5_no_association_context"):
        ablation[baseline] = {}
        for split in ("val", "iid_test", "ood_test", "val_association"):
            row = next(
                row
                for row in bootstrap
                if row["left_model"] == baseline
                and row["split"] == split
                and row["event"] == "all"
                and row["metric"] == "position_prediction"
            )
            ablation[baseline][split] = {
                "improvement_m": f(row, "right_improvement"),
                "ci95": [f(row, "ci95_low"), f(row, "ci95_high")],
                "trajectory_win_fraction": f(row, "trajectory_win_fraction"),
            }
    return {"P4_over_P2": split_claims, "position_prediction_ablation": ablation}


def main() -> int:
    audit = json.loads((ARTIFACT_ROOT / "RESULT_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Refusing to build figures from a failed result audit")
    main_rows = rows("main_tracking.csv")
    bootstrap = rows("paired_bootstrap.csv")
    bp = rows("bp_robustness.csv")
    nis = rows("nis_calibration.csv")
    complexity = rows("complexity_scaling.csv")
    bp_hierarchical = json.loads(
        (ARTIFACT_ROOT / "statistics" / "bp_hierarchical_bootstrap.json").read_text(
            encoding="utf-8"
        )
    )

    claims = tracking_claims(main_rows, bootstrap)
    claims["event_position_gain"] = build_event_gain(bootstrap)
    claims["bp_robustness"] = build_bp_heatmap(bp)
    claims["bp_hierarchical_bootstrap"] = build_bp_hierarchical_figure(bp_hierarchical)
    claims["nis_calibration"] = build_nis_figure(nis)
    claims["complexity"] = build_complexity_figure(complexity)
    claims["trajectory_cases"] = build_trajectory_analysis()
    claims["claim_limits"] = [
        "Learned-model formal evidence currently uses training seed 123 only.",
        "The P4-versus-P5 association-context effect is small and mixed across splits.",
        "BP matched-state NIS coverage is poor and is not improved by P4 on macro-average.",
        "Scaling measurements cover the prediction interface only, not end-to-end tracking.",
    ]
    STAT_ROOT.mkdir(parents=True, exist_ok=True)
    (STAT_ROOT / "core_claims.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "figures": sorted(path.name for path in FIGURE_ROOT.glob("*.svg")),
                "statistics": "statistics/core_claims.json",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
