"""Compare Temporal, acceleration-RK4 CIPIC, and fair direct-state CIPIC."""

import argparse
import json
from pathlib import Path

import torch


TRACKING_KEYS = (
    "location_rmse_prediction",
    "velocity_rmse_prediction",
    "location_rmse_update",
    "velocity_rmse_update",
)
DIRECT_KEYS = (
    "causal_innovation_position_rms_m",
    "causal_innovation_position_max_m",
    "causal_innovation_velocity_rms_mps",
    "causal_innovation_velocity_max_mps",
    "interaction_strong_advantage",
    "loss_interaction_base",
    "loss_interaction_expert",
    "loss_interaction_strong_base",
    "loss_interaction_strong_expert",
    "causal_innovation_eligible_fraction",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceleration_summary", required=True, type=Path)
    parser.add_argument("--direct_run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    acceleration = json.loads(
        args.acceleration_summary.read_text(encoding="utf-8")
    )
    checkpoint_dir = args.direct_run / "GNN_groupTargetTracking"
    direct = torch.load(
        checkpoint_dir / "best_position.pth",
        map_location="cpu",
        weights_only=False,
    )
    initial = json.loads(
        (checkpoint_dir / "epoch_minus_one_metrics.json").read_text(
            encoding="utf-8"
        )
    )["clean_validation"]
    protocol = direct.get("evaluation_protocol") or {}
    if direct.get("causal_innovation_mode") != "state_offset_raw_fair":
        raise ValueError("direct checkpoint is not the fair state-offset mode")
    if not bool(direct.get("causal_innovation_use_peer", False)):
        raise ValueError("direct checkpoint must use peer histories")
    if protocol.get("protocol_sha256") != acceleration["protocol_sha256"]:
        raise ValueError("acceleration and direct-state protocols differ")
    baseline_difference = max(
        abs(
            float(initial[key])
            - float(acceleration["temporal"][key])
        )
        for key in TRACKING_KEYS
    )
    if baseline_difference > 1e-4:
        raise ValueError(
            f"Temporal baselines differ by {baseline_difference:.3e}"
        )

    metrics = direct.get("metrics") or {}
    direct_result = {
        "epoch": int(direct.get("epoch", -999)),
        **{
            key: round(float(metrics.get(key, 0.0)), 8)
            for key in TRACKING_KEYS + DIRECT_KEYS
        },
    }
    for endpoint, metric in (
        ("prediction", "location_rmse_prediction"),
        ("update", "location_rmse_update"),
    ):
        direct_result[f"delta_{endpoint}_m_vs_temporal"] = round(
            direct_result[metric]
            - float(acceleration["temporal"][metric]),
            8,
        )
        direct_result[f"delta_{endpoint}_m_vs_acceleration"] = round(
            direct_result[metric]
            - float(acceleration["own_peer"][metric]),
            8,
        )

    report = {
        "protocol_sha256": protocol.get("protocol_sha256"),
        "baseline_agreement_max_abs": baseline_difference,
        "temporal": acceleration["temporal"],
        "acceleration_rk4_own_peer": acceleration["own_peer"],
        "direct_state_own_peer": direct_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
