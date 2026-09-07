"""Full online counterfactual evaluation of learned process noise Q."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from evaluation.diagnose_v3_dynamics import build_model
from train.trainer_v3 import Trainer
from train.training_utils_v3 import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


def event_name(graph_data, start):
    for name, attribute in (("strong", "strong_window_mask"),
                            ("switch", "switch_window_mask"),
                            ("steady", "steady_window_mask")):
        if hasattr(graph_data, attribute):
            mask = getattr(graph_data, attribute).reshape(-1)
            if 0 <= start < mask.numel() and bool(mask[start]):
                return name
    return "unlabelled"


def safe_nll(error, covariance):
    eye = torch.eye(4, device=error.device, dtype=error.dtype)
    covariance = covariance + 1e-5 * eye
    solution = torch.linalg.solve(covariance, error.unsqueeze(-1))
    mahalanobis = (error.unsqueeze(-2) @ solution).flatten()
    logdet = torch.linalg.slogdet(covariance).logabsdet
    return 0.5 * (mahalanobis + logdet)


class Stats:
    def __init__(self):
        self.values = defaultdict(list)
        self.squared = defaultdict(float)
        self.count = defaultdict(int)
        self.q_sum = None
        self.q_count = 0

    def add_value(self, key, tensor):
        values = torch.as_tensor(tensor).detach().float().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel():
            self.values[key].extend(values.cpu().tolist())

    def add_error(self, key, error):
        error = error.detach().float()
        self.squared[key] += float(error.square().sum())
        self.count[key] += int(error.shape[0])

    def add_q(self, q, valid):
        selected = q[valid, 0].detach().double()
        if selected.numel():
            total = selected.sum(dim=0).cpu()
            self.q_sum = total if self.q_sum is None else self.q_sum + total
            self.q_count += selected.size(0)

    def result(self):
        result = {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
                "count": len(values),
            }
            for key, values in self.values.items() if values
        }
        result["rmse"] = {
            key: math.sqrt(value / max(self.count[key], 1))
            for key, value in self.squared.items()
        }
        return result


@torch.no_grad()
def run_variant(model, trainer, loader, config, mode, scale=1.0, fixed_q=None,
                max_trajectories=0):
    model.process_noise_mode = mode
    model.process_noise_scale = scale
    model.fixed_process_noise = fixed_q
    stats = Stats()
    event_stats = defaultdict(Stats)
    input_len = config["input_graph_window_len"]
    trajectories = 0
    for graph_data in loader:
        if max_trajectories and trajectories >= max_trajectories:
            break
        trajectories += 1
        graph_data = graph_data.to(trainer.device)
        raw_r, _ = trainer.compute_measurement_noise(graph_data, 0)
        covariance = trainer.init_state_covariance(raw_r)
        window = GraphDataProcessor.init_input_window(graph_data, config, trainer.args)
        normalized = window.clone()
        last_step = graph_data.y.size(1) - 1
        for time_step in range(input_len, last_step + 1):
            normalized, centroids = minmax_scaler(
                window, normalized, config, trainer.args.neural_net
            )
            _, normalized_r = trainer.compute_measurement_noise(graph_data, time_step)
            outputs = model(normalized, 0.0, covariance, None, normalized_r)
            valid = outputs.real_mask[:, 0].bool()
            category = event_name(graph_data, time_step - input_len)
            targets = (stats, event_stats[category])
            prediction = outputs.states[:, 0, 0]
            update = outputs.updated_state[:, 0]
            truth = outputs.target[:, 0]
            scale_vector = prediction.new_tensor([
                config["S_POS"], config["S_POS"],
                config["S_VEL"], config["S_VEL"],
            ])
            pred_error = (prediction - truth) * scale_vector
            update_error = (update - truth) * scale_vector
            P_pred = outputs.covariances[:, 0, 0]
            P_update = outputs.updated_covariance
            H = P_pred.new_zeros(2, 4)
            H[0, 0] = H[1, 1] = 1.0
            D = torch.diag(scale_vector)
            H_bar = H @ D
            R_phys = normalized_r * config["S_POS"] ** 2
            innovation = config["S_POS"] * (
                normalized.measurement_future[:, 0] - prediction[:, :2]
            )
            S = H_bar.unsqueeze(0) @ P_pred @ H_bar.T.unsqueeze(0) + R_phys
            PHt = P_pred @ H_bar.T.unsqueeze(0)
            K = torch.linalg.solve(S, PHt.transpose(1, 2)).transpose(1, 2)
            K_phys = scale_vector.view(1, 4, 1) * K
            nis = (innovation.unsqueeze(1) @ torch.linalg.solve(
                S, innovation.unsqueeze(-1)
            )).flatten()
            prior_nll = safe_nll(prediction - truth, P_pred)
            posterior_nll = safe_nll(update - truth, P_update)
            q_raw = model.last_process_noise_raw
            q_used = model.last_process_noise_used
            q_state = model.last_state_process_noise
            q_phys = (scale_vector.view(1, 1, 4, 1) * q_state
                      * scale_vector.view(1, 1, 1, 4))
            for target_stats in targets:
                target_stats.add_error("position_prediction", pred_error[valid, :2])
                target_stats.add_error("velocity_prediction", pred_error[valid, 2:])
                target_stats.add_error("position_update", update_error[valid, :2])
                target_stats.add_error("velocity_update", update_error[valid, 2:])
                target_stats.add_value("innovation_norm", innovation[valid].norm(dim=-1))
                target_stats.add_value("nis", nis[valid])
                target_stats.add_value("nis_below_95pct", (nis[valid] <= 5.991).float())
                target_stats.add_value("prior_nll", prior_nll[valid])
                target_stats.add_value("posterior_nll", posterior_nll[valid])
                target_stats.add_value("position_gain", torch.linalg.matrix_norm(
                    K_phys[valid, :2], dim=(-2, -1)))
                target_stats.add_value("velocity_gain", torch.linalg.matrix_norm(
                    K_phys[valid, 2:], dim=(-2, -1)))
                target_stats.add_value("raw_q_trace", q_used[valid, 0].diagonal(
                    dim1=-2, dim2=-1).sum(-1))
                target_stats.add_value("state_q_position_trace", q_phys[valid, 0, :2, :2].diagonal(
                    dim1=-2, dim2=-1).sum(-1))
                target_stats.add_value("state_q_velocity_trace", q_phys[valid, 0, 2:, 2:].diagonal(
                    dim1=-2, dim2=-1).sum(-1))
                target_stats.add_value(
                    "q_acceleration_multiplier",
                    outputs.q_acceleration_multiplier[valid, 0, 0],
                )
                target_stats.add_value(
                    "q_causal_surprise", outputs.q_innovation_score[valid, 0, 0]
                )
                extra_q = outputs.q_position_noise[valid, 0, 0]
                extra_position_phys = (
                    extra_q[:, :2, :2].diagonal(dim1=-2, dim2=-1).sum(-1)
                    * config["S_POS"] ** 2
                )
                target_stats.add_value("q_extra_position_trace", extra_position_phys)
            if mode == "learned":
                stats.add_q(q_raw, valid)
            _, posterior_raw = trainer.compute_evaluation_metrics(
                window, outputs, centroids
            )
            covariance = outputs.updated_covariance.detach()
            if time_step < last_step:
                prior_raw = trainer._denormalize_prior(
                    outputs, centroids, graph_data.batch
                )
                window = GraphDataProcessor.update_input_window(
                    window, graph_data, config, posterior_raw, prior_raw,
                    time_step - input_len + 1, trainer.args,
                )
    return {
        "mode": mode, "scale": scale, "trajectories": trajectories,
        "global": stats.result(),
        "events": {key: value.result() for key, value in event_stats.items()},
    }, (stats.q_sum / stats.q_count if stats.q_count else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--adaptive-q", action="store_true")
    parser.add_argument("--learned-only", action="store_true")
    parser.add_argument("--tail-boost", type=float, default=0.0)
    parser.add_argument("--tail-threshold", type=float, default=0.8)
    parser.add_argument("--tail-position-boost", type=float, default=0.0,
                        help='per-axis physical position variance per surprise unit')
    args_cli = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(
        T=2.0, device=device, use_sean=True, use_gasf=True,
        decoder_residual_length=[8, 16], neural_net="sggt_net",
        train_epochs=1, initial_covariance_scale=1.0,
        output_dir="./tmp/q_evaluation",
    )
    config = dict(CONFIG)
    config.update({"S_VEL": 40.0, "S_EDGE_V_DIST": 40.0,
                   "output_graph_window_len": 1,
                   "window_sampling_ratio": [0.4, 0.3, 0.3]})
    CONFIG.update(config)
    model = build_model(args).to(device)
    if args_cli.adaptive_q:
        # Rebuild with the Q-v2 constructor option; all other architecture
        # settings remain identical to ``build_model``.
        model.decoder.adaptive_q_mode = "innovation"
        if model.decoder.Q_innovation_encoder is None:
            hidden = model.decoder.hidden_size
            model.decoder.Q_innovation_encoder = torch.nn.Sequential(
                torch.nn.Linear(hidden * 2 + 4, hidden),
                torch.nn.LayerNorm(hidden), torch.nn.SiLU(),
                torch.nn.Linear(hidden, 3),
            ).to(device)
    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.q_tail_boost = args_cli.tail_boost
    model.decoder.q_tail_threshold = args_cli.tail_threshold
    model.decoder.q_tail_position_boost = args_cli.tail_position_boost
    model.eval()
    trainer = Trainer(model, None, None, None, config, args)
    dataset = GroupTargetTrackingDatasetV3(
        args_cli.data_dir, "val", load_structure=True,
        load_analysis_metadata=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    learned, mean_q = run_variant(
        model, trainer, loader, config, "learned",
        max_trajectories=args_cli.max_trajectories,
    )
    variants = [learned]
    specifications = [] if args_cli.learned_only else [
        ("zero", 0.0, None),
        ("fixed", 1.0, mean_q),
        ("scaled", 0.5, None),
        ("scaled", 1.5, None),
        ("scaled", 2.0, None),
    ]
    for mode, scale, fixed in specifications:
        result, _ = run_variant(
            model, trainer, loader, config, mode, scale, fixed,
            args_cli.max_trajectories,
        )
        variants.append(result)
    output = {
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "tail_boost": args_cli.tail_boost,
        "tail_threshold": args_cli.tail_threshold,
        "tail_position_boost": args_cli.tail_position_boost,
        "dataset_size": len(dataset),
        "mean_learned_acceleration_q": mean_q.tolist(),
        "variants": variants,
    }
    path = Path(args_cli.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
