"""Locate the first tensor that differs between single-scene and batched rollout."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from model.sggt_net import SGGT_Net
from train.trainer_v3 import Trainer
from train.training_utils_v3 import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


def build_model(checkpoint):
    motion = SecondOrderNeuralODE_groupTrack(
        solver="rk4", dt=2.0, mixtures=1, static_f_dim=0,
        n_hidden=64, n_layers=3, use_SEAN=False, use_GASF=False,
        gasf_fusion_mode="concat", gasf_aux_source="gasf",
    )
    model = SGGT_Net(
        4, 64, 3, 2, 1, 0.1, "graphconv", True,
        motion, 9, 64, 3, 2, 0.2, 0.1, [8, 16], 2,
        "graphconv", False, False, 2.0,
        model_variant="sggt_v2", covariance_transition="cv",
        adaptive_q_mode="legacy", adaptive_q_tail_boost=0.0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def rollout(model, full, config, args, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    trainer = Trainer(model, optimizer, schedule, logger, config, args)
    raw_r, _ = trainer.compute_measurement_noise(full, 0)
    covariance = trainer.init_state_covariance(raw_r)
    window = GraphDataProcessor.init_input_window(full, config, args)
    selector = full.batch == 0
    traces = []
    for offset in range(steps):
        normalized, centroids = minmax_scaler(
            window, window.clone(), config, args.neural_net
        )
        _, normalized_r = trainer.compute_measurement_noise(
            full, config["input_graph_window_len"] + offset
        )
        with torch.no_grad():
            output = model(normalized, 0.0, covariance, None, normalized_r)
        traces.append({
            "normalized_x": normalized.x[selector].clone(),
            "normalized_residual": normalized.residual[selector].clone(),
            "R": normalized_r[selector].clone(),
            "encoder": output.encoder_output_cache[selector].clone(),
            "decoder_hidden": output.decoder_hidden_cache[selector].clone(),
            "states": output.states[selector].clone(),
            "updated_state": output.updated_state[selector].clone(),
            "covariances": output.covariances[selector].clone(),
            "updated_covariance": output.updated_covariance[selector].clone(),
        })
        prior = trainer._denormalize_prior(output, centroids, full.batch)
        posterior = trainer._denormalize_update(output, centroids, full.batch)
        covariance = output.updated_covariance.detach()
        if offset + 1 < steps:
            window = GraphDataProcessor.update_input_window(
                window, full, config, posterior, prior, offset + 1, args
            )
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=12)
    options = parser.parse_args()
    torch.manual_seed(123)
    config = dict(CONFIG)
    config.update({"S_VEL": 40.0, "S_EDGE_V_DIST": 40.0,
                   "output_graph_window_len": 1})
    CONFIG.update(config)
    args = SimpleNamespace(
        device="cpu", train_epochs=1, T=2.0,
        decoder_residual_length=[8, 16], neural_net="sggt_net",
        initial_covariance_scale=1.0,
    )
    dataset = GroupTargetTrackingDatasetV3(
        options.dataset_root, "val", load_structure=False
    )
    single = next(iter(DataLoader([dataset[0]], batch_size=1, shuffle=False)))
    batched = next(iter(DataLoader([dataset[i] for i in range(4)], batch_size=4, shuffle=False)))
    model = build_model(options.checkpoint)
    single_trace = rollout(model, single, config, args, options.steps)
    batch_trace = rollout(model, batched, config, args, options.steps)
    first_failure = None
    for step, (left, right) in enumerate(zip(single_trace, batch_trace)):
        for name in left:
            if left[name].shape != right[name].shape:
                print(f"step={step:02d} tensor={name:22s} shape_single={tuple(left[name].shape)} shape_batch={tuple(right[name].shape)}")
                if first_failure is None:
                    first_failure = (step, name, "shape")
                continue
            difference = float((left[name] - right[name]).abs().max())
            print(f"step={step:02d} tensor={name:22s} max_abs_diff={difference:.9g}")
            if first_failure is None and difference > 1e-5:
                first_failure = (step, name, difference)
    print({"first_failure": first_failure})


if __name__ == "__main__":
    main()
