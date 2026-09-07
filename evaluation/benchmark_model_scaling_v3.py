"""Benchmark the deployed SGGT predictor at controlled target counts.

Accuracy is never computed on the replicated graphs.  A real five-target V9
causal window is copied into disconnected blocks solely to measure how the
deployed eight-frame-Encoder/one-step-Decoder ``model.predict`` interface
scales with node and edge count.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data_v3 import GroupTargetTrackingDatasetV3
from train.training_utils_v3 import GraphDataProcessor
from utils.common.param_config import param_config
from utils.minmax_scaler import minmax_scaler


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replicate_window(window, copies):
    """Replicate node tensors/edges without inventing cross-block edges."""
    if copies < 1:
        raise ValueError("copies must be positive")
    base_nodes = int(window.x.size(0))
    output = Data()
    node_names = (
        "x", "y", "measurement", "measurement_future", "residual",
        "nan_mask", "real_mask", "current_real_mask", "maneuver_delta_v",
        "innovation_raw", "innovation_whitened", "innovation_peer_raw",
        "innovation_peer_whitened", "innovation_covariance", "nis_history",
        "innovation_valid", "innovation_peer_valid",
        "association_detection_mass", "association_entropy",
        "association_spread",
    )
    for name in node_names:
        if hasattr(window, name):
            value = getattr(window, name)
            setattr(output, name, torch.cat([value] * copies, dim=0))
    output.edge_index = []
    output.edge_features = []
    for edge_index, edge_features in zip(
        window.edge_index, window.edge_features
    ):
        output.edge_index.append(torch.cat([
            edge_index + copy * base_nodes for copy in range(copies)
        ], dim=1))
        output.edge_features.append(torch.cat(
            [edge_features] * copies, dim=0
        ))
    # All blocks form one controlled scene.  They remain disconnected in the
    # graph, so replication changes scale without fabricating interactions.
    output.batch = torch.zeros(
        base_nodes * copies, dtype=torch.long, device=window.x.device
    )
    return output


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_trajectory", type=int, default=15)
    parser.add_argument("--target_counts", nargs="+", type=int,
                        default=[5, 10, 20, 40])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--tracker_trajectory",
        default="generate_trajectory/scene01/trajectory01.pt",
        help="used only to construct the deployed checkpoint runtime",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 2:
        raise ValueError("warmup must be >=1 and repetitions must be >=2")
    checkpoint = Path(args.checkpoint).resolve()
    device = torch.device(args.device)
    trajectory_info = torch.load(
        args.tracker_trajectory, map_location="cpu", weights_only=False
    )
    radar = param_config(
        trajectory_info,
        np.asarray([3.0, 0.01], dtype=np.float64),
        nominal_pd=0.85,
        nominal_clutter_mean=20.0,
        checkpoint=str(checkpoint),
        device=device,
        adaptive_detection_probability=False,
        verbose=False,
    )
    model = radar.neural_network["model"]
    config = radar.neural_network["config"]
    model.eval()
    runtime_args = radar.neural_network.get("args")
    if runtime_args is None:
        # param_config keeps the same deployment defaults; these two fields are
        # all GraphDataProcessor needs to form the first causal window.
        from config import Args
        runtime_args = Args().initialize(Args().parse()).parse_args([])
        runtime_args.device = device

    dataset = GroupTargetTrackingDatasetV3(
        args.data_dir, "val", load_structure=False,
        load_analysis_metadata=False,
    )
    sample = dataset[int(args.base_trajectory)].to(device)
    sample.batch = torch.zeros(sample.x.size(0), dtype=torch.long, device=device)
    base_nodes = int(sample.x.size(0))
    if base_nodes != 5:
        raise ValueError(
            f"base trajectory must contain exactly 5 targets, got {base_nodes}"
        )
    base_window = GraphDataProcessor.init_input_window(
        sample, config, runtime_args, start_bias=0
    )
    rows = []
    with torch.inference_mode():
        for target_count in args.target_counts:
            if target_count % base_nodes:
                raise ValueError(
                    f"target count {target_count} is not divisible by {base_nodes}"
                )
            raw = replicate_window(base_window, target_count // base_nodes)
            normalized, _ = minmax_scaler(
                raw, raw.clone(), config, "sggt_net"
            )
            normalized.P_scaler = torch.eye(
                4, dtype=normalized.x.dtype, device=device
            ).view(1, 1, 4, 4).expand(target_count, 1, -1, -1).clone() * 1e-2
            for _ in range(args.warmup):
                model.predict(normalized)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            samples_ms = []
            for _ in range(args.repetitions):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model.predict(normalized)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples_ms.append(1000.0 * (time.perf_counter() - start))
            rows.append({
                "target_count": int(target_count),
                "edge_count_last_frame": int(
                    normalized.edge_index[-1].size(1)
                ),
                "mean_ms": float(statistics.mean(samples_ms)),
                "std_ms": float(statistics.stdev(samples_ms)),
                "median_ms": float(statistics.median(samples_ms)),
                "p95_ms": percentile(samples_ms, 95),
                "mean_us_per_target": float(
                    1000.0 * statistics.mean(samples_ms) / target_count
                ),
                "peak_cuda_memory_mib": (
                    float(torch.cuda.max_memory_allocated(device) / 2 ** 20)
                    if device.type == "cuda" else None
                ),
            })
    report = {
        "protocol": {
            "version": 1,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_root": str(Path(args.data_dir).resolve()),
            "base_split": "val",
            "base_trajectory": int(args.base_trajectory),
            "base_target_count": base_nodes,
            "controlled_replication": (
                "real five-target causal window copied into disconnected blocks"
            ),
            "accuracy_interpretation": "timing-only; no replicated-graph accuracy",
            "timed_interface": (
                "model.predict: eight-frame encoder + one decoder + covariance time update"
            ),
            "excluded": "model load, dataset I/O, BP association and measurement update",
            "warmup_calls": int(args.warmup),
            "timed_calls": int(args.repetitions),
            "device": str(device),
        },
        "model": {
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameter_count": int(sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )),
        },
        "scaling": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
