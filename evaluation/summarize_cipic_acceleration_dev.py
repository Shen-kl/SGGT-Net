"""Validate and summarize the fixed 200/50 CIPIC development comparison."""

import argparse
import json
from pathlib import Path

import torch


METRICS = (
    "location_rmse_prediction",
    "velocity_rmse_prediction",
    "location_rmse_update",
    "velocity_rmse_update",
    "loss_interaction_base",
    "loss_interaction_expert",
    "loss_interaction_strong_base",
    "loss_interaction_strong_expert",
    "interaction_strong_advantage",
    "causal_innovation_acceleration_rms_mps2",
    "causal_innovation_acceleration_max_mps2",
    "causal_innovation_eligible_fraction",
)


def load_run(root: Path, expected_peer: bool, expected_trajectories: int):
    checkpoint_dir = root / "GNN_groupTargetTracking"
    initial_path = checkpoint_dir / "epoch_minus_one_metrics.json"
    best_path = checkpoint_dir / "best_position.pth"
    if not initial_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(
            f"incomplete run {root}: expected {initial_path} and {best_path}"
        )
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("causal_innovation_mode") != "acceleration_rk4_raw":
        raise ValueError(f"unexpected CIPIC mode in {best_path}")
    if bool(best.get("causal_innovation_use_peer", True)) != expected_peer:
        raise ValueError(f"peer-history switch mismatch in {best_path}")
    if bool(best.get("causal_innovation_dynamic_activation", False)):
        raise ValueError(f"dynamic activation must be disabled in {best_path}")
    if bool(best.get("causal_innovation_association_context", False)):
        raise ValueError(f"association context must be disabled in {best_path}")
    protocol = best.get("evaluation_protocol") or {}
    if int(protocol.get("trajectory_count", -1)) != expected_trajectories:
        raise ValueError(
            f"expected {expected_trajectories} validation trajectories in "
            f"{best_path}"
        )
    metrics = best.get("metrics") or {}
    return {
        "root": str(root.resolve()),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "epoch": int(best.get("epoch", -999)),
        "use_peer": expected_peer,
        "maximum_acceleration_mps2": float(
            best.get("causal_innovation_max_acceleration_mps2", 5.0)
        ),
        "initial": initial["clean_validation"],
        "best": {key: float(metrics.get(key, 0.0)) for key in METRICS},
    }


def rounded_metrics(values):
    return {key: round(float(values[key]), 8) for key in METRICS if key in values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own_only", required=True, type=Path)
    parser.add_argument("--own_peer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected_val_trajectories", type=int, default=50)
    args = parser.parse_args()

    own = load_run(
        args.own_only, False, args.expected_val_trajectories
    )
    peer = load_run(
        args.own_peer, True, args.expected_val_trajectories
    )
    if own["protocol_sha256"] != peer["protocol_sha256"]:
        raise ValueError("own-only and own+peer evaluation protocols differ")

    baseline_keys = (
        "location_rmse_prediction",
        "velocity_rmse_prediction",
        "location_rmse_update",
        "velocity_rmse_update",
    )
    baseline_difference = max(
        abs(float(own["initial"][key]) - float(peer["initial"][key]))
        for key in baseline_keys
    )
    # Repeated CUDA rollouts can differ at the 1e-5 m level because pooled
    # reductions do not promise bitwise identity.  Keep the measured mismatch
    # in the report and reject anything large enough to indicate a real
    # protocol/model discrepancy.
    if baseline_difference > 1e-4:
        raise ValueError(
            "zero-initialized Temporal baselines differ by "
            f"{baseline_difference:.3e}"
        )

    baseline = {
        key: float(own["initial"][key]) for key in baseline_keys
    }
    report = {
        "protocol_sha256": own["protocol_sha256"],
        "baseline_agreement_max_abs": baseline_difference,
        "temporal": baseline,
        "own_only": {
            "epoch": own["epoch"],
            **rounded_metrics(own["best"]),
        },
        "own_peer": {
            "epoch": peer["epoch"],
            **rounded_metrics(peer["best"]),
        },
    }
    for name in ("own_only", "own_peer"):
        result = report[name]
        result["delta_prediction_m_vs_temporal"] = round(
            result["location_rmse_prediction"]
            - baseline["location_rmse_prediction"],
            8,
        )
        result["delta_update_m_vs_temporal"] = round(
            result["location_rmse_update"]
            - baseline["location_rmse_update"],
            8,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
