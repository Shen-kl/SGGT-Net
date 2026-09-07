"""Diagnose loss gradients, EKF behavior, and kinematics on v3 rollouts."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from model.sggt_net import SGGT_Net
from train.trainer_v3 import Trainer
from train.training_utils_v3 import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


def build_model(args):
    motion = SecondOrderNeuralODE_groupTrack(
        solver="rk4", dt=args.T, mixtures=1, static_f_dim=0,
        n_hidden=64, n_layers=3, use_SEAN=args.use_sean,
        use_GASF=args.use_gasf, gasf_fusion_mode="concat",
        gasf_aux_source="gasf",
    )
    return SGGT_Net(
        4, 64, 3, 2, 1, 0.1, "graphconv", True, motion,
        9, 64, 3, 2, 0.2, 0.1, [8, 16], 2, "graphconv",
        args.use_gasf, args.use_sean, args.T,
        model_variant="sggt_v2", covariance_transition="cv",
        gasf_filter_mode="abc", gasf_graph_mode="graph",
    )


def grad_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    return float(sum(
        gradient.detach().square().sum()
        for gradient in gradients if gradient is not None
    ).sqrt())


def append(store, **values):
    for key, value in values.items():
        store[key].append(float(value))


def summarize(store):
    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p95": float(np.percentile(values, 95)),
            "count": len(values),
        }
        for key, values in store.items() if values
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-sean", action="store_true")
    parser.add_argument("--use-gasf", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=8)
    parser.add_argument("--gradient-steps", type=int, default=6)
    cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(
        T=2.0, device=device, use_sean=cli.use_sean, use_gasf=cli.use_gasf,
        decoder_residual_length=[8, 16], neural_net="sggt_net",
        train_epochs=1, initial_covariance_scale=1.0,
        lambda_struct=0.5, lambda_temporal=0.05, lambda_aux=0.1,
        lambda_nll=1e-4, lambda_cv_guard=1.0, cv_guard_margin=1e-6,
        cv_guard_position_weight=1.0, cv_guard_velocity_weight=1.0,
        output_dir="./tmp/diagnostic",
    )
    config = dict(CONFIG)
    config.update({"S_VEL": 40.0, "S_EDGE_V_DIST": 40.0,
                   "output_graph_window_len": 1,
                   "window_sampling_ratio": [0.4, 0.3, 0.3]})
    CONFIG.update(config)
    model = build_model(args).to(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    trainer = Trainer(model, None, None, None, config, args)
    dataset = GroupTargetTrackingDatasetV3(
        cli.data_dir, "val", load_structure=cli.use_sean,
        load_analysis_metadata=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    gradients, kalman, kinematics = defaultdict(list), defaultdict(list), defaultdict(list)

    for trajectory_index, graph_data in enumerate(loader):
        if trajectory_index >= cli.max_trajectories:
            break
        graph_data = graph_data.to(device)
        raw_r, _ = trainer.compute_measurement_noise(graph_data, 0)
        covariance = trainer.init_state_covariance(raw_r)
        window = GraphDataProcessor.init_input_window(graph_data, config, args)
        normalized = window.clone()
        input_len = config["input_graph_window_len"]
        last_step = graph_data.y.size(1) - 1
        scored_steps = set(np.linspace(
            input_len, last_step, cli.gradient_steps, dtype=int
        ).tolist())
        for time_step in range(input_len, last_step + 1):
            normalized, centroids = minmax_scaler(
                window, normalized, config, args.neural_net
            )
            _, normalized_r = trainer.compute_measurement_noise(graph_data, time_step)
            scored = time_step in scored_steps
            with torch.set_grad_enabled(scored):
                outputs = model(normalized, 0.0, covariance, None, normalized_r)
                target = outputs.target[:, 0]
                prediction = outputs.states[:, 0, 0]
                update = outputs.updated_state[:, 0]
                mask = outputs.real_mask[:, 0].float().unsqueeze(-1)
                denominator = mask.sum().clamp_min(1.0)
                pred_pos = (F.huber_loss(prediction[:, :2], target[:, :2], reduction="none") * mask).sum() / denominator
                pred_vel = (F.huber_loss(prediction[:, 2:], target[:, 2:], reduction="none") * mask).sum() / denominator
                update_pos = (F.huber_loss(update[:, :2], target[:, :2], reduction="none") * mask).sum() / denominator
                update_vel = (F.huber_loss(update[:, 2:], target[:, 2:], reduction="none") * mask).sum() / denominator
            if scored:
                append(
                    gradients,
                    prediction_position_loss=pred_pos.detach(),
                    prediction_velocity_loss=pred_vel.detach(),
                    update_position_loss=update_pos.detach(),
                    update_velocity_loss=update_vel.detach(),
                    prediction_position_grad=grad_norm(pred_pos, parameters),
                    prediction_velocity_grad=grad_norm(pred_vel, parameters),
                    update_position_grad=grad_norm(update_pos, parameters),
                    update_velocity_grad=grad_norm(update_vel, parameters),
                )

            with torch.no_grad():
                Pn = outputs.covariances[:, 0, 0]
                H = Pn.new_zeros(2, 4)
                H[0, 0] = H[1, 1] = 1.0
                S = H.unsqueeze(0) @ Pn @ H.T.unsqueeze(0) + normalized_r
                K = torch.linalg.solve(
                    S, (Pn @ H.T.unsqueeze(0)).transpose(1, 2)
                ).transpose(1, 2)
                scale = Pn.new_tensor([config["S_POS"], config["S_POS"],
                                       config["S_VEL"], config["S_VEL"]])
                K_phys = scale.view(1, 4, 1) * K / config["S_POS"]
                innovation = config["S_POS"] * (
                    normalized.measurement_future[:, 0] - prediction[:, :2]
                )
                P_phys = scale.view(1, 4, 1) * Pn * scale.view(1, 1, 4)
                append(
                    kalman,
                    position_gain=torch.linalg.matrix_norm(K_phys[:, :2], dim=(-2, -1)).mean(),
                    velocity_gain=torch.linalg.matrix_norm(K_phys[:, 2:], dim=(-2, -1)).mean(),
                    innovation_norm=torch.linalg.vector_norm(innovation, dim=-1).mean(),
                    position_variance=P_phys[:, [0, 1], [0, 1]].mean(),
                    velocity_variance=P_phys[:, [2, 3], [2, 3]].mean(),
                    position_velocity_cross_cov=torch.linalg.matrix_norm(P_phys[:, :2, 2:], dim=(-2, -1)).mean(),
                )
                prior_raw = trainer._denormalize_prior(outputs, centroids, graph_data.batch)
                last = window.x[:, -1]
                predicted_raw = prior_raw[:, 0]
                displacement = predicted_raw[:, [0, 2]] - last[:, [0, 2]]
                trapezoid = 0.5 * args.T * (
                    last[:, [1, 3]] + predicted_raw[:, [1, 3]]
                )
                consistency = torch.linalg.vector_norm(displacement - trapezoid, dim=-1)
                true_next = window.y[:, 0]
                true_displacement = true_next[:, [0, 2]] - last[:, [0, 2]]
                valid = window.real_mask[:, 0, 0].bool()
                if not valid.any():
                    valid = torch.ones_like(valid)
                append(
                    kinematics,
                    consistency_error=consistency[valid].mean(),
                    displacement_error=torch.linalg.vector_norm(
                        displacement - true_displacement, dim=-1
                    )[valid].mean(),
                    terminal_velocity_error=torch.linalg.vector_norm(
                        predicted_raw[:, [1, 3]] - true_next[:, [1, 3]], dim=-1
                    )[valid].mean(),
                )
                _, posterior_raw = trainer.compute_evaluation_metrics(
                    window, outputs, centroids
                )
                covariance = outputs.updated_covariance.detach()
                if time_step < last_step:
                    bias = time_step - input_len + 1
                    window = GraphDataProcessor.update_input_window(
                        window, graph_data, config, posterior_raw, prior_raw,
                        bias, args,
                    )

    result = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "trajectories": min(cli.max_trajectories, len(dataset)),
        "gradient": summarize(gradients),
        "kalman": summarize(kalman),
        "kinematics": summarize(kinematics),
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
