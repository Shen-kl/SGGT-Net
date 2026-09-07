"""Merge the full V9 Temporal backbone with validated physical-SEAN modules."""

import argparse
import math
from pathlib import Path

import torch


PHYSICAL_PREFIXES = (
    "decoder.GASF.SEAN.",
    "decoder.GASF.context_activation_head.",
    "decoder.GASF.interaction_innovation_projection.",
    "decoder.motion_model.gasf_interaction_context_embedding.",
)
PHYSICAL_KEYS = {
    "decoder.motion_model.interaction_physical_gain_raw",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal_checkpoint", required=True)
    parser.add_argument("--physical_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical_init_gain", type=float, default=0.02)
    parser.add_argument("--physical_max_gain", type=float, default=0.075)
    return parser.parse_args()


def main():
    cli = parse_args()
    temporal = torch.load(
        cli.temporal_checkpoint, map_location="cpu", weights_only=False
    )
    physical = torch.load(
        cli.physical_checkpoint, map_location="cpu", weights_only=False
    )
    temporal_state = temporal["state_dict"]
    merged_state = {
        key: value.detach().clone()
        for key, value in physical["state_dict"].items()
    }
    replaced = []
    retained_physical = []
    shape_mismatch = []
    for key, value in merged_state.items():
        is_physical = key in PHYSICAL_KEYS or key.startswith(PHYSICAL_PREFIXES)
        if is_physical:
            retained_physical.append(key)
            continue
        source = temporal_state.get(key)
        if source is None:
            continue
        if source.shape != value.shape:
            shape_mismatch.append((key, tuple(source.shape), tuple(value.shape)))
            continue
        merged_state[key] = source.detach().clone()
        replaced.append(key)
    unused_temporal = sorted(set(temporal_state) - set(merged_state))
    if shape_mismatch:
        raise RuntimeError(f"shape mismatches: {shape_mismatch[:8]}")
    if not 0.0 < cli.physical_init_gain < cli.physical_max_gain:
        raise ValueError("physical_init_gain must lie inside (0, physical_max_gain)")
    probability = cli.physical_init_gain / cli.physical_max_gain
    merged_state["decoder.motion_model.interaction_physical_gain_raw"] = (
        torch.tensor(math.log(probability / (1.0 - probability)))
    )

    payload = dict(physical)
    payload["state_dict"] = merged_state
    payload.pop("optimizer_state_dict", None)
    payload.pop("lr_schedule", None)
    payload["epoch"] = -1
    payload["metrics"] = temporal.get("metrics", {})
    payload["model_variant"] = "sggt_v2"
    payload["covariance_transition"] = "cv"
    payload["interaction_physical_residual"] = True
    payload["interaction_physical_max_gain"] = float(cli.physical_max_gain)
    payload["interaction_physical_init_gain"] = float(cli.physical_init_gain)
    payload["interaction_activation_floor"] = 0.0
    payload["gasf_fusion_mode"] = "concat"
    payload["gasf_graph_mode"] = "graph"
    payload["decoder_use_SEAN"] = True
    payload["decoder_use_GASF"] = True
    payload["shared_initialization"] = {
        "temporal_checkpoint": str(Path(cli.temporal_checkpoint).resolve()),
        "physical_checkpoint": str(Path(cli.physical_checkpoint).resolve()),
        "temporal_tensors_replaced": len(replaced),
        "physical_tensors_retained": len(retained_physical),
        "physical_init_gain": float(cli.physical_init_gain),
        "physical_max_gain": float(cli.physical_max_gain),
        "unused_temporal_tensors": unused_temporal,
        "policy": (
            "full V9 Temporal weights for every compatible non-physical "
            "tensor; validated SEAN/physical tensors for newly added paths"
        ),
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved={output.resolve()}")
    print(f"temporal_tensors_replaced={len(replaced)}")
    print(f"physical_tensors_retained={len(retained_physical)}")
    print(f"unused_temporal_tensors={len(unused_temporal)}")


if __name__ == "__main__":
    main()
