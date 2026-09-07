"""Diagnose whether causal geometric neighbours share maneuver increments.

This script uses truth velocity increments only for offline analysis.  It never
constructs a structure label and is not imported by the training pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    levels = values.new_tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    result = torch.quantile(values.float(), levels)
    return {
        name: float(value)
        for name, value in zip(("p10", "p25", "p50", "p75", "p90"), result)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_scenes", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir) / "train"
    input_files = sorted((root / "input").glob("batch_data_*.pt"))[
        : args.max_scenes
    ]
    records = []
    for input_path in input_files:
        target_path = root / "target" / input_path.name
        inputs = torch.load(input_path, map_location="cpu", weights_only=False)
        targets = torch.load(target_path, map_location="cpu", weights_only=False)
        state = targets["graph_target"].float()
        valid = targets["real_mask"][..., 0].bool()
        strong = inputs["strong_mask"].bool()
        for frame in range(1, state.size(1)):
            if not bool(strong[frame]):
                continue
            edge_index = inputs["graph_input_edge_index"][frame - 1].long()
            if edge_index.numel() == 0:
                continue
            source, target = edge_index
            usable = (
                valid[source, frame]
                & valid[target, frame]
                & valid[source, frame - 1]
                & valid[target, frame - 1]
                & (source < target)
            )
            source, target = source[usable], target[usable]
            if source.numel() == 0:
                continue
            position = state[:, frame, [0, 2]]
            velocity = state[:, frame - 1, [1, 3]]
            delta_v = state[:, frame, [1, 3]] - velocity
            distance = torch.linalg.vector_norm(
                position[source] - position[target], dim=-1
            )
            velocity_gap = torch.linalg.vector_norm(
                velocity[source] - velocity[target], dim=-1
            )
            maneuver_gap = torch.linalg.vector_norm(
                delta_v[source] - delta_v[target], dim=-1
            )
            maneuver_cosine = torch.nn.functional.cosine_similarity(
                delta_v[source], delta_v[target], dim=-1, eps=1e-8
            )
            records.append(torch.stack(
                (distance, velocity_gap, maneuver_gap, maneuver_cosine), dim=-1
            ))

    if not records:
        raise RuntimeError("No valid strong-maneuver candidate edges found")
    values = torch.cat(records, dim=0)
    distance, velocity_gap, maneuver_gap, maneuver_cosine = values.unbind(dim=-1)
    result = {
        "scene_count": len(input_files),
        "strong_candidate_pair_count": int(values.size(0)),
        "distance_m": _quantiles(distance),
        "velocity_gap_mps": _quantiles(velocity_gap),
        "maneuver_gap_mps": _quantiles(maneuver_gap),
        "maneuver_cosine": _quantiles(maneuver_cosine),
        "bins": [],
    }
    for distance_limit in (50.0, 100.0, 200.0, 400.0):
        for velocity_limit in (2.0, 5.0, 10.0, 20.0):
            selected = (distance <= distance_limit) & (velocity_gap <= velocity_limit)
            if not selected.any():
                continue
            result["bins"].append({
                "distance_limit_m": distance_limit,
                "velocity_limit_mps": velocity_limit,
                "count": int(selected.sum()),
                "fraction": float(selected.float().mean()),
                "maneuver_gap_mean_mps": float(maneuver_gap[selected].mean()),
                "maneuver_cosine_mean": float(maneuver_cosine[selected].mean()),
                "same_direction_fraction": float(
                    (maneuver_cosine[selected] > 0.5).float().mean()
                ),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
