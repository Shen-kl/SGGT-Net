"""Redraw the three simulation-scenario paper figures from saved MC-MSE curves.

This script is deliberately independent of the Monte Carlo evaluation pipeline.  It
only reads the aggregate curves already stored in ``MC_MSE_CURVES.json``; changing a
legend therefore never reruns tracking or changes any reported metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURVES = (
    ROOT
    / "results"
    / "paper_scene02_mc_mse_20260826"
    / "curves"
    / "MC_MSE_CURVES.json"
)
DEFAULT_LEGENDS = Path(__file__).with_name("scene02_figure_legends_v3.json")
DEFAULT_OUTPUT = ROOT / "docs" / "paper_revision" / "figures"

FIGURE_FILENAMES = {
    "comparison": "scene02_comparison_ospa_mc_rmse_protocol_qualified.png",
    "robustness_pd": "scene02_robustness_pd_ospa_mc_rmse.png",
    "robustness_clutter": "scene02_robustness_clutter_ospa_mc_rmse.png",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_label_override(raw: str) -> tuple[str, str, str]:
    """Parse SECTION.KEY=LABEL used by the repeatable --label option."""
    target, separator, label = raw.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError(
            "label override must use SECTION.KEY=LABEL, for example "
            "comparison.P4=SGGT-Net"
        )
    section, dot, key = target.partition(".")
    if not dot or not section or not key:
        raise argparse.ArgumentTypeError(
            "label target must contain a section and key separated by a dot"
        )
    return section, key, label


def apply_label_overrides(
    config: dict[str, Any], overrides: list[tuple[str, str, str]]
) -> None:
    for section, key, label in overrides:
        labels = config.get(section)
        if not isinstance(labels, dict):
            raise ValueError(f"unknown legend section: {section}")
        if key not in labels:
            raise ValueError(f"unknown legend key: {section}.{key}")
        labels[key] = label


def validate_config(config: dict[str, Any], curves: dict[str, Any]) -> None:
    expected_sources = {
        "comparison": {
            "P4": "main",
            "MCST-BP": "historical",
            "BP-R-RGTT": "historical",
            "MTP-GO": "historical",
        },
        "robustness_pd": {
            key: "robustness" for key in config.get("robustness_pd_order", [])
        },
        "robustness_clutter": {
            key: "robustness" for key in config.get("robustness_clutter_order", [])
        },
    }
    for section, source_by_key in expected_sources.items():
        order_key = f"{section}_order"
        order = config.get(order_key)
        labels = config.get(section)
        if not isinstance(order, list) or not order:
            raise ValueError(f"{order_key} must be a non-empty list")
        if not isinstance(labels, dict):
            raise ValueError(f"{section} must be an object mapping curve keys to labels")
        if len(order) != len(set(order)):
            raise ValueError(f"{order_key} contains duplicate keys")
        for key in order:
            if key not in labels or not str(labels[key]).strip():
                raise ValueError(f"missing non-empty legend label for {section}.{key}")
            source = source_by_key.get(key)
            if source is None or key not in curves.get(source, {}):
                raise ValueError(f"curve data not found for {section}.{key}")


def mc_rmse_curve(curve_record: dict[str, Any]) -> np.ndarray:
    values = np.asarray(curve_record["mc_mse_curve"], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("mc_mse_curve must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("mc_mse_curve contains NaN or infinity")
    # Tiny negative values can occur only through roundoff; material negatives
    # indicate that the input is not an MSE curve.
    if float(values.min()) < -1e-12:
        raise ValueError("mc_mse_curve contains negative values")
    return np.sqrt(np.maximum(values, 0.0))


def curve_record(curves: dict[str, Any], section: str, key: str) -> dict[str, Any]:
    if section == "comparison":
        if key == "P4":
            return curves["main"][key]["ospa"]
        return curves["historical"][key]
    return curves["robustness"][key]["ospa"]


def draw_figure(
    curves: dict[str, Any],
    config: dict[str, Any],
    section: str,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for key in config[f"{section}_order"]:
        values = mc_rmse_curve(curve_record(curves, section, key))
        axis.plot(
            np.arange(values.size),
            values,
            label=config[section][key],
            linewidth=1.8 if key == "P4" else 1.5,
        )
    axis.set_xlabel("Frame")
    axis.set_ylabel("Monte Carlo RMSE OSPA / m")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves",
        type=Path,
        default=DEFAULT_CURVES,
        help="aggregate MC-MSE JSON produced by the formal evaluation",
    )
    parser.add_argument(
        "--legend-config",
        type=Path,
        default=DEFAULT_LEGENDS,
        help="JSON file containing legend text and plotting order",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory receiving the three PNG files",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        type=parse_label_override,
        metavar="SECTION.KEY=LABEL",
        help=(
            "override one configured label; repeat as needed, e.g. "
            "--label comparison.P4='Proposed SGGT-Net'"
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--print-labels",
        action="store_true",
        help="print the resolved label configuration before drawing",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    curves = read_json(args.curves.resolve())
    config = read_json(args.legend_config.resolve())
    apply_label_overrides(config, args.label)
    validate_config(config, curves)
    if args.print_labels:
        selected = {
            section: config[section]
            for section in ("comparison", "robustness_pd", "robustness_clutter")
        }
        print(json.dumps(selected, indent=2, ensure_ascii=False))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = {}
    for section, filename in FIGURE_FILENAMES.items():
        target = output / filename
        draw_figure(curves, config, section, target, args.dpi)
        generated[section] = str(target)
    print(json.dumps({"generated": generated}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
