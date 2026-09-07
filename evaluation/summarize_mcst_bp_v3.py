"""Store aggregate-only Monte Carlo MSE curves from an MCST-BP report."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    runs = report["runs"]
    if not runs:
        raise ValueError("The MCST-BP report contains no Monte Carlo runs")
    frame_count = len(runs[0]["per_frame"])
    if any(len(run["per_frame"]) != frame_count for run in runs):
        raise ValueError("Monte Carlo runs do not share the same frame count")

    ospa = np.asarray([
        [frame["ospa"] for frame in run["per_frame"]] for run in runs
    ], dtype=np.float64)
    cardinality_error = np.asarray([
        [frame["estimated_count"] - frame["truth_count"] for frame in run["per_frame"]]
        for run in runs
    ], dtype=np.float64)
    position_sse = np.asarray([
        [frame["position_sse"] for frame in run["per_frame"]] for run in runs
    ], dtype=np.float64)
    velocity_sse = np.asarray([
        [frame["velocity_sse"] for frame in run["per_frame"]] for run in runs
    ], dtype=np.float64)
    matched_count = np.asarray([
        [frame["matched_state_count"] for frame in run["per_frame"]] for run in runs
    ], dtype=np.float64)

    ospa_mse = np.mean(np.square(ospa), axis=0)
    cardinality_mse = np.mean(np.square(cardinality_error), axis=0)
    denominator = np.maximum(matched_count.sum(axis=0), 1.0)
    position_mse = position_sse.sum(axis=0) / denominator
    velocity_mse = velocity_sse.sum(axis=0) / denominator
    arrays = {
        "ospa_mc_mse_curve": ospa_mse,
        "ospa_final_rmse": np.asarray(np.sqrt(np.mean(ospa_mse))),
        "position_mc_mse_curve": position_mse,
        "position_final_rmse": np.asarray(
            np.sqrt(position_sse.sum() / max(matched_count.sum(), 1.0))
        ),
        "velocity_mc_mse_curve": velocity_mse,
        "velocity_final_rmse": np.asarray(
            np.sqrt(velocity_sse.sum() / max(matched_count.sum(), 1.0))
        ),
        "cardinality_mc_mse_curve": cardinality_mse,
        "cardinality_final_rmse": np.asarray(np.sqrt(np.mean(cardinality_mse))),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "mcst_bp_mc_mse_curves.npz", **arrays)
    summary = {
        "source_report": str(Path(args.input).resolve()),
        "monte_carlo_runs": len(runs),
        "frame_count": frame_count,
        **{
            key: float(value)
            for key, value in arrays.items()
            if key.endswith("final_rmse")
        },
    }
    (output_dir / "mcst_bp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plots = (
        ("ospa_mc_mse_curve", "OSPA RMSE", "m"),
        ("position_mc_mse_curve", "Matched position RMSE", "m"),
        ("velocity_mc_mse_curve", "Matched velocity RMSE", "m/s"),
        ("cardinality_mc_mse_curve", "Cardinality RMSE", "targets"),
    )
    for axis, (key, title, unit) in zip(axes.flat, plots):
        axis.plot(np.arange(frame_count), np.sqrt(arrays[key]), linewidth=1.5)
        axis.set_xlabel("Time step")
        axis.set_ylabel(unit)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "mcst_bp_mc_rmse_curves.png", dpi=220)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
