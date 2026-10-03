"""Summarize the fair direct-state CIPIC channel attribution experiment."""

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
DIAGNOSTIC_KEYS = (
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


def load_arm(run_root, expected_channels, reference):
    checkpoint_root = run_root / "GNN_groupTargetTracking"
    payload = torch.load(
        checkpoint_root / "best_position.pth",
        map_location="cpu",
        weights_only=False,
    )
    initial = json.loads(
        (checkpoint_root / "epoch_minus_one_metrics.json").read_text(
            encoding="utf-8"
        )
    )["clean_validation"]
    protocol = payload.get("evaluation_protocol") or {}
    if payload.get("causal_innovation_mode") != "state_offset_raw_fair":
        raise ValueError(f"{expected_channels}: wrong output mode")
    if payload.get("causal_innovation_state_channels", "full") != expected_channels:
        raise ValueError(f"{expected_channels}: wrong active channel metadata")
    if not bool(payload.get("causal_innovation_use_peer", False)):
        raise ValueError(f"{expected_channels}: peer innovation is disabled")
    if protocol.get("protocol_sha256") != reference["protocol_sha256"]:
        raise ValueError(f"{expected_channels}: evaluation protocol differs")
    baseline_difference = max(
        abs(float(initial[key]) - float(reference["temporal"][key]))
        for key in TRACKING_KEYS
    )
    if baseline_difference > 1e-4:
        raise ValueError(
            f"{expected_channels}: Temporal baseline differs by "
            f"{baseline_difference:.3e}"
        )
    metrics = payload.get("metrics") or {}
    result = {
        "epoch": int(payload.get("epoch", -999)),
        "baseline_agreement_max_abs": baseline_difference,
        **{
            key: round(float(metrics.get(key, 0.0)), 8)
            for key in TRACKING_KEYS + DIAGNOSTIC_KEYS
        },
    }
    for endpoint, metric in (
        ("prediction", "location_rmse_prediction"),
        ("update", "location_rmse_update"),
    ):
        result[f"delta_{endpoint}_m_vs_temporal"] = round(
            result[metric] - float(reference["temporal"][metric]), 8
        )
        result[f"delta_{endpoint}_m_vs_full"] = round(
            result[metric]
            - float(reference["direct_state_own_peer"][metric]),
            8,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_summary", required=True, type=Path)
    parser.add_argument("--position_run", required=True, type=Path)
    parser.add_argument("--velocity_run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference = json.loads(args.full_summary.read_text(encoding="utf-8"))
    report = {
        "protocol_sha256": reference["protocol_sha256"],
        "temporal": reference["temporal"],
        "acceleration_rk4_own_peer": reference["acceleration_rk4_own_peer"],
        "direct_state_full": reference["direct_state_own_peer"],
        "direct_state_position_only": load_arm(
            args.position_run, "position_only", reference
        ),
        "direct_state_velocity_only": load_arm(
            args.velocity_run, "velocity_only", reference
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
