import argparse
import json

import numpy as np
import torch

from data.load_data import GroupTargetTrackingDataset_load_whole_trajectory


def analyze(dataset_path, split, dt, event_radius):
    dataset = GroupTargetTrackingDataset_load_whole_trajectory(dataset_path, split)
    accelerations = []
    event_accelerations = []
    non_event_accelerations = []
    valid_pairs = 0
    event_pairs = 0

    for graph in dataset:
        velocity = graph.y[..., [1, 3]]
        node_valid = graph.real_mask[..., [1, 3]].all(dim=-1)
        pair_valid = node_valid[:, 1:] & node_valid[:, :-1]
        acceleration = torch.linalg.vector_norm(
            velocity[:, 1:] - velocity[:, :-1], dim=-1
        ) / dt

        time = torch.arange(1, graph.y.size(1))
        change_times = graph.param_change_time.reshape(-1)
        change_times = change_times[
            (change_times > 1) & (change_times < graph.y.size(1))
        ]
        if change_times.numel():
            event_time = (
                (time[:, None] - change_times[None, :]).abs() <= event_radius
            ).any(dim=1)
        else:
            event_time = torch.zeros_like(time, dtype=torch.bool)
        event_mask = pair_valid & event_time.unsqueeze(0)
        non_event_mask = pair_valid & ~event_time.unsqueeze(0)

        accelerations.append(acceleration[pair_valid])
        event_accelerations.append(acceleration[event_mask])
        non_event_accelerations.append(acceleration[non_event_mask])
        valid_pairs += int(pair_valid.sum())
        event_pairs += int(event_mask.sum())

    values = torch.cat(accelerations).numpy()
    event_values = torch.cat(event_accelerations).numpy()
    non_event_values = torch.cat(non_event_accelerations).numpy()

    def describe(array):
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "quantiles": {
                str(q): float(np.quantile(array, q))
                for q in (0.5, 0.75, 0.9, 0.95, 0.99)
            },
            "fractions_above_mps2": {
                str(threshold): float((array > threshold).mean())
                for threshold in (0.1, 0.25, 0.5, 1.0, 2.0)
            },
        }

    return {
        "split": split,
        "trajectories": len(dataset),
        "dt_seconds": dt,
        "event_radius_steps": event_radius,
        "valid_node_time_pairs": valid_pairs,
        "event_pair_fraction": event_pairs / max(valid_pairs, 1),
        "all": describe(values),
        "event": describe(event_values),
        "non_event": describe(non_event_values),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--event_radius", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.data_dir, args.split, args.dt, args.event_radius)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
