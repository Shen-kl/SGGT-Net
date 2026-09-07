"""Build protocol-audited OSPA and cardinality plots at Pd=0.99, clutter=3.

SGGT-Net, MCST, and MTP-GO use the current simulation-scenario evaluations.
The unchanged BP-R-RGTT curves are loaded from the frozen historical data.
Because older aggregate artifacts did not retain signed cardinality means, the
script reruns only SGGT-Net and MCST with the same 20 Monte Carlo seeds, checks
that their OSPA MSE curves reproduce the frozen aggregates, extracts target
count means, and removes the temporary per-run reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BP_RUNNER = ROOT / "evaluation" / "run_bp_tracker_robustness_v3.py"
TRAJECTORY = Path(
    r"E:\PycharmProjects\Neural Network\GNN_groupTargetTracking_v9_norm_without_struct"
    r"\generate_trajectory\scene02\trajectory01.pt"
)
MCST_REPOSITORY = Path(r"E:\PycharmProjects\Neural Network\MCST_2D_update")
SGGT_CHECKPOINT = (
    ROOT
    / "checkpoint"
    / "formal_association_cgic"
    / "seed_123"
    / "P4_full"
    / "GNN_groupTargetTracking"
    / "best_association_position.pth"
)
MCST_CHECKPOINT = ROOT / "checkpoint" / "mcst_v9_full_seed123" / "best_prior_position.pth"
SGGT_AGGREGATE = (
    ROOT
    / "results"
    / "paper_scene02_mc_mse_20260826"
    / "aggregate"
    / "robustness"
    / "pd_0p99_clutter_3.json"
)
MCST_AGGREGATE_CURVES = (
    ROOT
    / "results"
    / "mcst_v9_seed123"
    / "robustness"
    / "curves"
    / "pd_0p99_clutter_3.npz"
)
MTP_REPORT = (
    ROOT
    / "results"
    / "formal_mtp_go_v1"
    / "seed_123"
    / "simulation_scenario"
    / "native_q_robustness"
    / "pd_0p99_clutter_3.json"
)
MTP_CURVES = MTP_REPORT.parent / "P1_native_q_robustness_ospa_curves.npz"
FORMAL_CURVES = (
    ROOT / "results" / "paper_scene02_mc_mse_20260826" / "curves" / "mc_mse_curves.npz"
)
HISTORICAL_CARDINALITY = (
    ROOT
    / "results"
    / "paper_simulation_scenario_cardinality_20260903"
    / "CARDINALITY_COMPARISON_CURVES.npz"
)
HISTORICAL_BP_SOURCE = Path(
    r"E:\PycharmProjects\Neural Network\SGGT-Net_paper"
    r"\GNN_version3\data\trajectory1_BP_R_RGTT_lambda_3.pkl"
)
DEFAULT_OUTPUT = ROOT / "results" / "paper_pd099_clutter3_method_comparison_20260906"
DEFAULT_TEMP = ROOT / ".tmp" / "pd099_clutter3_method_comparison"

METHODS = ("SGGT-Net", "MCST", "MTP-GO", "BP-R-RGTT")
COLORS = {
    "SGGT-Net": "#D02528",
    "MCST": "#26A76E",
    "MTP-GO": "#E47820",
    "BP-R-RGTT": "#1A758A",
}
LINESTYLES = {
    "SGGT-Net": "-",
    "MCST": "--",
    "MTP-GO": "-.",
    "BP-R-RGTT": ":",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def current_report_is_valid(path: Path, checkpoint: Path, backend: str) -> bool:
    if not path.exists():
        return False
    try:
        report = load_json(path)
        protocol = report["protocol"]
        return all(
            (
                protocol["model_backend"] == backend,
                protocol["checkpoint_sha256"] == sha256(checkpoint),
                float(protocol["actual_pd"]) == 0.99,
                float(protocol["actual_clutter_mean"]) == 3.0,
                float(protocol["nominal_pd"]) == 0.99,
                float(protocol["nominal_clutter_mean"]) == 2.0,
                float(protocol["range_noise_std_m"]) == 3.0,
                float(protocol["azimuth_noise_std_deg"]) == 0.01,
                int(protocol["monte_carlo_runs"]) == 20,
                int(protocol["seed_start"]) == 20260824,
                len(report["runs"]) == 20,
                all(len(run["per_frame"]) == 100 for run in report["runs"]),
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_current_report(
    report_path: Path,
    checkpoint: Path,
    backend: str,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    command = [
        sys.executable,
        str(BP_RUNNER),
        "--model_backend",
        backend,
        "--checkpoint",
        str(checkpoint),
        "--trajectory",
        str(TRAJECTORY),
        "--output",
        str(report_path),
        "--actual_pd",
        "0.99",
        "--actual_clutter_mean",
        "3",
        "--nominal_pd",
        "0.99",
        "--nominal_clutter_mean",
        "2",
        "--monte_carlo",
        "20",
        "--seed",
        "20260824",
        "--device",
        "cuda",
        "--range_noise_std_m",
        "3",
        "--azimuth_noise_std_deg",
        "0.01",
        "--surveillance_padding_m",
        "500",
        "--ospa_cutoff_m",
        "100",
        "--ospa_order",
        "1",
        "--neural_process_noise_scale",
        "6.874306",
        "--neural_position_q_variance_m2",
        "1.316616",
    ]
    if backend == "mcst":
        command.extend(("--mcst_repository", str(MCST_REPOSITORY)))
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )


def ospa_mse_from_report(report: dict) -> np.ndarray:
    matrix = np.asarray(
        [[frame["ospa"] for frame in run["per_frame"]] for run in report["runs"]],
        dtype=np.float64,
    )
    return np.mean(np.square(matrix), axis=0)


def count_mean_from_report(report: dict) -> np.ndarray:
    matrix = np.asarray(
        [
            [frame["estimated_count"] for frame in run["per_frame"]]
            for run in report["runs"]
        ],
        dtype=np.float64,
    )
    return np.mean(matrix, axis=0)


def protocol_check_mtp(report: dict) -> None:
    protocol = report["protocol"]
    checks = (
        float(protocol["actual_pd"]) == 0.99,
        float(protocol["actual_clutter_mean"]) == 3.0,
        int(protocol["monte_carlo_runs"]) == 20,
        int(protocol["seed_start"]) == 20260824,
        float(protocol["neural_process_noise_scale"]) == 1.0,
        float(protocol["neural_position_q_variance_per_axis_m2"]) == 0.0,
        len(report["runs"]) == 20,
        all(len(run["per_frame"]) == 100 for run in report["runs"]),
    )
    if not all(checks):
        raise RuntimeError("MTP-GO source does not match its native-Q c3 protocol")


def save_plot(
    output: Path,
    time_s: np.ndarray,
    ospa_rmse: dict[str, np.ndarray],
    truth_count: np.ndarray,
    count_mean: dict[str, np.ndarray],
) -> None:
    plt.rcParams.update({"font.size": 11})
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method in METHODS:
        axis.plot(
            time_s,
            ospa_rmse[method],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            label=method,
        )
    axis.set_xlabel("Time / s")
    axis.set_ylabel("Monte Carlo OSPA RMSE / m")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "simulation_scenario_ospa_pd099_clutter3.png", dpi=600)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.step(
        time_s,
        truth_count,
        where="post",
        color="#111111",
        linewidth=1.8,
        label="Ground truth",
    )
    for method in METHODS:
        axis.plot(
            time_s,
            count_mean[method],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.7,
            marker="*",
            markersize=4.5,
            markevery=2,
            label=method,
        )
    axis.set_xlabel("Time / s")
    axis.set_ylabel("Estimated number of targets")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.savefig(output / "simulation_scenario_cardinality_pd099_clutter3.png", dpi=600)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    args = parser.parse_args()
    output = args.output.resolve()
    temp = args.temp.resolve()
    allowed_temp = (ROOT / ".tmp").resolve()
    if temp == allowed_temp or allowed_temp not in temp.parents:
        raise ValueError(f"temporary path must be below {allowed_temp}: {temp}")

    required = (
        BP_RUNNER,
        TRAJECTORY,
        SGGT_CHECKPOINT,
        MCST_CHECKPOINT,
        SGGT_AGGREGATE,
        MCST_AGGREGATE_CURVES,
        MTP_REPORT,
        MTP_CURVES,
        FORMAL_CURVES,
        HISTORICAL_CARDINALITY,
        HISTORICAL_BP_SOURCE,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)

    reports = {}
    for method, checkpoint, backend in (
        ("SGGT-Net", SGGT_CHECKPOINT, "sggt"),
        ("MCST", MCST_CHECKPOINT, "mcst"),
    ):
        report_path = temp / f"{method.lower().replace('-', '_')}_pd099_clutter3_mc20.json"
        if not current_report_is_valid(report_path, checkpoint, backend):
            print(f"[{method}] reproduce c3 report for signed cardinality mean", flush=True)
            run_current_report(
                report_path,
                checkpoint,
                backend,
                output / "logs" / f"{method.lower().replace('-', '_')}.stdout.log",
                output / "logs" / f"{method.lower().replace('-', '_')}.stderr.log",
            )
        if not current_report_is_valid(report_path, checkpoint, backend):
            raise RuntimeError(f"invalid reproduced report: {report_path}")
        reports[method] = load_json(report_path)

    sggt_aggregate = load_json(SGGT_AGGREGATE)
    sggt_frozen_mse = np.asarray(
        sggt_aggregate["curves"]["ospa"]["mc_mse_curve"], dtype=np.float64
    )
    with np.load(MCST_AGGREGATE_CURVES) as loaded:
        mcst_frozen_mse = loaded["ospa_mc_mse_curve"].copy()
    reproduced = {
        "SGGT-Net": ospa_mse_from_report(reports["SGGT-Net"]),
        "MCST": ospa_mse_from_report(reports["MCST"]),
    }
    max_differences = {
        "SGGT-Net": float(np.max(np.abs(reproduced["SGGT-Net"] - sggt_frozen_mse))),
        "MCST": float(np.max(np.abs(reproduced["MCST"] - mcst_frozen_mse))),
    }
    reproduction_diagnostics = {
        "SGGT-Net": {
            "max_abs_mse_difference": max_differences["SGGT-Net"],
            "mean_abs_rmse_curve_difference_m": float(
                np.mean(
                    np.abs(
                        np.sqrt(reproduced["SGGT-Net"]) - np.sqrt(sggt_frozen_mse)
                    )
                )
            ),
            "frozen_final_rmse_m": float(np.sqrt(np.mean(sggt_frozen_mse))),
            "rerun_final_rmse_m": float(
                np.sqrt(np.mean(reproduced["SGGT-Net"]))
            ),
        },
        "MCST": {
            "max_abs_mse_difference": max_differences["MCST"],
            "mean_abs_rmse_curve_difference_m": float(
                np.mean(
                    np.abs(np.sqrt(reproduced["MCST"]) - np.sqrt(mcst_frozen_mse))
                )
            ),
            "frozen_final_rmse_m": float(np.sqrt(np.mean(mcst_frozen_mse))),
            "rerun_final_rmse_m": float(np.sqrt(np.mean(reproduced["MCST"]))),
        },
    }
    if max_differences["MCST"] > 1e-6:
        raise RuntimeError(
            f"reproduced MCST OSPA curve differs from its frozen curve: {max_differences}"
        )
    if (
        abs(
            reproduction_diagnostics["SGGT-Net"]["rerun_final_rmse_m"]
            - reproduction_diagnostics["SGGT-Net"]["frozen_final_rmse_m"]
        )
        > 0.05
        or reproduction_diagnostics["SGGT-Net"][
            "mean_abs_rmse_curve_difference_m"
        ]
        > 0.1
    ):
        raise RuntimeError(
            "SGGT-Net rerun exceeds the accepted numerical reproduction tolerance: "
            f"{reproduction_diagnostics['SGGT-Net']}"
        )

    mtp_report = load_json(MTP_REPORT)
    protocol_check_mtp(mtp_report)
    with np.load(MTP_CURVES) as loaded:
        mtp_mse = loaded["pd_0p99_clutter_3__ospa__mc_mse_curve"].copy()
    if np.max(np.abs(ospa_mse_from_report(mtp_report) - mtp_mse)) > 1e-6:
        raise RuntimeError("MTP-GO raw report and aggregate OSPA curve differ")

    with np.load(FORMAL_CURVES) as loaded:
        historical_bp_mse = loaded[
            "historical__BP-R-RGTT__ospa__mc_mse_curve"
        ].copy()
    with np.load(HISTORICAL_CARDINALITY) as loaded:
        frame_index = loaded["frame_index"].copy()
        time_s = loaded["time_s"].copy()
        truth_count = loaded["ground_truth_count"].copy()
        historical_bp_count = loaded[
            "BP-R-RGTT__estimated_count_mc_mean"
        ].copy()
    if not all(array.shape == (100,) for array in (frame_index, time_s, truth_count)):
        raise RuntimeError("historical time/truth arrays do not contain 100 frames")

    ospa_mse = {
        # Use the paired reruns for SGGT-Net and MCST so that each method's
        # OSPA and cardinality curves come from the same Monte Carlo samples.
        "SGGT-Net": reproduced["SGGT-Net"],
        "MCST": reproduced["MCST"],
        "MTP-GO": mtp_mse,
        "BP-R-RGTT": historical_bp_mse,
    }
    ospa_rmse = {method: np.sqrt(values) for method, values in ospa_mse.items()}
    count_mean = {
        "SGGT-Net": count_mean_from_report(reports["SGGT-Net"]),
        "MCST": count_mean_from_report(reports["MCST"]),
        "MTP-GO": count_mean_from_report(mtp_report),
        "BP-R-RGTT": historical_bp_count,
    }
    if any(values.shape != (100,) for values in (*ospa_rmse.values(), *count_mean.values())):
        raise RuntimeError("one or more comparison curves do not contain 100 frames")

    arrays = {
        "frame_index": frame_index,
        "time_s": time_s,
        "ground_truth_count": truth_count,
    }
    for method in METHODS:
        key = method.lower().replace("-", "_")
        arrays[f"{key}__ospa_mc_mse_curve"] = ospa_mse[method]
        arrays[f"{key}__ospa_mc_rmse_curve"] = ospa_rmse[method]
        arrays[f"{key}__estimated_count_mc_mean"] = count_mean[method]
    np.savez_compressed(output / "comparison_curves_pd099_clutter3.npz", **arrays)

    metadata = {
        "condition": {"actual_pd": 0.99, "actual_clutter_mean": 3.0},
        "curve_definition": {
            "ospa": "sqrt(mean over Monte Carlo runs of per-frame OSPA squared)",
            "cardinality": "mean estimated target count over Monte Carlo runs",
            "smoothing": "none",
        },
        "methods": {
            "SGGT-Net": {
                "monte_carlo_runs": 20,
                "source_protocol": reports["SGGT-Net"]["protocol"],
            },
            "MCST": {
                "monte_carlo_runs": 20,
                "source_protocol": reports["MCST"]["protocol"],
            },
            "MTP-GO": {
                "monte_carlo_runs": 20,
                "source_protocol": mtp_report["protocol"],
            },
            "BP-R-RGTT": {
                "monte_carlo_runs": 10,
                "historical_source": str(HISTORICAL_BP_SOURCE.resolve()),
                "historical_source_sha256": sha256(HISTORICAL_BP_SOURCE),
                "protocol_note": "unchanged frozen historical Pd=0.99, clutter=3 result",
            },
        },
        "reproduction_diagnostics": reproduction_diagnostics,
        "final_ospa_rmse_m": {
            method: float(np.sqrt(np.mean(ospa_mse[method]))) for method in METHODS
        },
        "curves": {
            "time_s": time_s.tolist(),
            "ground_truth_count": truth_count.tolist(),
            **{
                method: {
                    "ospa_mc_rmse": ospa_rmse[method].tolist(),
                    "estimated_count_mc_mean": count_mean[method].tolist(),
                }
                for method in METHODS
            },
        },
    }
    (output / "comparison_curves_pd099_clutter3.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with (output / "comparison_curves_pd099_clutter3.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = ["frame_index", "time_s", "ground_truth_count"]
        for method in METHODS:
            fields.extend((f"{method}_ospa_mc_rmse", f"{method}_estimated_count_mc_mean"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(100):
            row = {
                "frame_index": int(frame_index[index]),
                "time_s": float(time_s[index]),
                "ground_truth_count": float(truth_count[index]),
            }
            for method in METHODS:
                row[f"{method}_ospa_mc_rmse"] = float(ospa_rmse[method][index])
                row[f"{method}_estimated_count_mc_mean"] = float(count_mean[method][index])
            writer.writerow(row)

    save_plot(output, time_s, ospa_rmse, truth_count, count_mean)
    for method in ("SGGT-Net", "MCST"):
        report_path = temp / f"{method.lower().replace('-', '_')}_pd099_clutter3_mc20.json"
        report_path.unlink(missing_ok=True)
    print(json.dumps(metadata["final_ospa_rmse_m"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
