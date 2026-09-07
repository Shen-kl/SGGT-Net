"""Plot Monte Carlo mean target-cardinality estimates for the paper scenario."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name(
    "simulation_scenario_cardinality_sources_v3.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "paper_simulation_scenario_cardinality_20260903"
FIGURE_NAME = "simulation_scenario_cardinality_comparison.png"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_legacy_pickle(
    path: Path, expected_sha256: str | None
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"legacy baseline file does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(f"legacy baseline hash mismatch: {path}")
    # These are trusted local artifacts from the original experiment repository.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    counts = np.asarray(payload["target_set_num_monte"], dtype=float)
    truth = np.asarray(payload["labels_set_num"], dtype=float).reshape(-1)
    if counts.ndim != 2 or counts.shape[1] != truth.size:
        raise ValueError(f"invalid cardinality array shape in {path}")
    delta_t = float(np.asarray(payload.get("delta_T", 2.0)).reshape(-1)[0])
    return counts.mean(axis=0), truth, delta_t, counts.shape[0]


def load_aggregate_json(path: Path) -> tuple[np.ndarray, np.ndarray, float, int]:
    payload = read_json(path)
    arrays = payload["arrays"]
    counts = np.asarray(arrays["estimated_count_mc_mean"], dtype=float)
    truth = np.asarray(arrays["ground_truth_count"], dtype=float)
    summary = payload["summary"]
    return (
        counts,
        truth,
        float(summary["delta_t_seconds"]),
        int(summary["monte_carlo_runs"]),
    )


def load_curves(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    order = config.get("method_order")
    methods = config.get("methods")
    if not isinstance(order, list) or not order:
        raise ValueError("method_order must be a non-empty list")
    if len(order) != len(set(order)):
        raise ValueError("method_order contains duplicate entries")
    if not isinstance(methods, dict):
        raise ValueError("methods must be an object")

    truth_reference: np.ndarray | None = None
    delta_t_reference: float | None = None
    loaded: dict[str, dict[str, Any]] = {}
    for method in order:
        if method not in methods:
            raise ValueError(f"missing configuration for method {method}")
        metadata = methods[method]
        source = resolve_source(metadata["source"])
        kind = metadata["kind"]
        if kind == "legacy_pickle":
            mean, truth, delta_t, runs = load_legacy_pickle(
                source, metadata.get("sha256")
            )
        elif kind == "aggregate_json":
            mean, truth, delta_t, runs = load_aggregate_json(source)
        else:
            raise ValueError(f"unknown source kind for {method}: {kind}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(truth)):
            raise ValueError(f"non-finite cardinality data for {method}")
        if truth_reference is None:
            truth_reference = truth
            delta_t_reference = delta_t
        else:
            if truth.shape != truth_reference.shape or not np.array_equal(
                truth, truth_reference
            ):
                raise ValueError(f"ground-truth cardinality mismatch for {method}")
            if not np.isclose(delta_t, delta_t_reference):
                raise ValueError(f"frame interval mismatch for {method}")
        loaded[method] = {
            "label": metadata["label"],
            "color": metadata["color"],
            "source": str(source),
            "source_sha256": sha256_file(source),
            "monte_carlo_runs": runs,
            "protocol_compatible_with_current": bool(
                metadata.get("protocol_compatible_with_current", False)
            ),
            "estimated_count_mc_mean": mean,
        }

    assert truth_reference is not None and delta_t_reference is not None
    time_s = np.arange(truth_reference.size, dtype=float) * delta_t_reference
    return time_s, {
        "ground_truth": truth_reference,
        "delta_t_seconds": delta_t_reference,
        "methods": loaded,
    }


def write_plot_data(
    output: Path,
    config_path: Path,
    config: dict[str, Any],
    time_s: np.ndarray,
    data: dict[str, Any],
) -> None:
    truth = data["ground_truth"]
    methods = data["methods"]
    json_payload = {
        "schema_version": 1,
        "curve_definition": "per-frame Monte Carlo mean estimated target cardinality",
        "configuration": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "protocol_note": config.get("protocol_note"),
        "delta_t_seconds": data["delta_t_seconds"],
        "frame_index": np.arange(time_s.size).tolist(),
        "time_s": time_s.tolist(),
        "ground_truth_count": truth.tolist(),
        "methods": {
            method: {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in values.items()
            }
            for method, values in methods.items()
        },
    }
    (output / "CARDINALITY_COMPARISON_CURVES.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    with (output / "CARDINALITY_COMPARISON_CURVES.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["frame_index", "time_s", "ground_truth_count", *methods]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(time_s.size):
            row = {
                "frame_index": frame,
                "time_s": time_s[frame],
                "ground_truth_count": truth[frame],
            }
            row.update(
                {
                    method: values["estimated_count_mc_mean"][frame]
                    for method, values in methods.items()
                }
            )
            writer.writerow(row)

    arrays = {
        "frame_index": np.arange(time_s.size, dtype=int),
        "time_s": time_s,
        "ground_truth_count": truth,
    }
    arrays.update(
        {
            f"{method}__estimated_count_mc_mean": values[
                "estimated_count_mc_mean"
            ]
            for method, values in methods.items()
        }
    )
    np.savez_compressed(output / "CARDINALITY_COMPARISON_CURVES.npz", **arrays)


def draw_plot(
    output: Path,
    config: dict[str, Any],
    time_s: np.ndarray,
    data: dict[str, Any],
) -> Path:
    style = config["figure"]
    figure, axis = plt.subplots(
        figsize=(float(style["width_inches"]), float(style["height_inches"])),
        constrained_layout=True,
    )
    axis.plot(
        time_s,
        data["ground_truth"],
        color="black",
        linestyle="-",
        linewidth=2.0,
        label=config["ground_truth_label"],
    )
    for method in config["method_order"]:
        values = data["methods"][method]
        axis.plot(
            time_s,
            values["estimated_count_mc_mean"],
            color=values["color"],
            linestyle="-",
            linewidth=float(style["line_width"]),
            marker=style["marker"],
            markersize=float(style["marker_size"]),
            markevery=int(style["marker_every"]),
            label=values["label"],
        )
    axis.set_xlabel(style["x_label"])
    axis.set_ylabel(style["y_label"])
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=float(style["legend_fontsize"]))
    target = output / FIGURE_NAME
    figure.savefig(target, dpi=int(style["dpi"]))
    plt.close(figure)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    time_s, data = load_curves(config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_plot_data(output, config_path, config, time_s, data)
    figure_path = draw_plot(output, config, time_s, data)
    print(
        json.dumps(
            {
                "figure": str(figure_path),
                "frame_count": int(time_s.size),
                "methods": {
                    method: {
                        "label": values["label"],
                        "monte_carlo_runs": values["monte_carlo_runs"],
                    }
                    for method, values in data["methods"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
