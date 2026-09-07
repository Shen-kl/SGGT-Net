"""Report per-loss gradient norms on one real causal v3 window."""

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


def group_name(name):
    if name.startswith("encoder."):
        return "encoder"
    if ".motion_model." in name:
        return "motion_model"
    if ".SEAN." in name:
        return "sean"
    if ".GASF." in name:
        return "gasf"
    return "decoder_other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--gasf", action="store_true")
    parser.add_argument("--sean", action="store_true")
    parser.add_argument("--lambda-cv-guard", type=float, default=1.0)
    options = parser.parse_args()

    torch.manual_seed(123)
    args = SimpleNamespace(
        device="cpu", train_epochs=1, T=2.0,
        decoder_residual_length=[8, 16], neural_net="sggt_net",
        lambda_struct=0.0, lambda_temporal=0.0,
        lambda_aux=0.1, lambda_nll=1e-4,
        lambda_cv_guard=options.lambda_cv_guard, cv_guard_margin=0.0,
        cv_guard_position_weight=1.0, cv_guard_velocity_weight=1.0,
        output_dir="./tmp/gradient_diagnostic",
    )
    config = dict(CONFIG)
    config.update({"output_graph_window_len": 1, "S_VEL": 40.0,
                   "S_EDGE_V_DIST": 40.0,
                   "window_sampling_ratio": [0.4, 0.3, 0.3]})
    CONFIG.update(config)
    dataset = GroupTargetTrackingDatasetV3(
        options.dataset_root, "train", load_structure=False,
        load_analysis_metadata=False,
    )
    full = next(iter(DataLoader([dataset[options.index]], batch_size=1)))
    motion = SecondOrderNeuralODE_groupTrack(
        solver="rk4", dt=args.T, mixtures=1, static_f_dim=0,
        n_hidden=64, n_layers=3, use_SEAN=options.sean,
        use_GASF=options.gasf, gasf_fusion_mode="concat",
    )
    model = SGGT_Net(
        4, 64, 3, 2, motion.mixtures, 0.1, "graphconv", True,
        motion, 9, 64, 3, 2, 0.2, 0.1, args.decoder_residual_length,
        2, "graphconv", options.gasf, options.sean, args.T,
        model_variant="sggt_v2", covariance_transition="cv",
        gasf_filter_mode="abc", gasf_graph_mode="graph",
        counterfactual_training=options.sean,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, None, logger, config, args)
    window = GraphDataProcessor.init_input_window(full, config, args)
    normalized, _ = minmax_scaler(window, window.clone(), config, args.neural_net)
    raw_r, normalized_r = trainer.compute_measurement_noise(full, config["input_graph_window_len"])
    covariance = trainer.init_state_covariance(raw_r)
    outputs = model(normalized, 0.0, covariance, None, normalized_r)
    weights = torch.ones(full.x.size(0))
    total = trainer.compute_loss(outputs, window, weights)

    coefficients = {
        "prediction": 1.0, "update": 1.0,
        "structure": args.lambda_struct, "temporal": args.lambda_temporal,
        "auxiliary": args.lambda_aux, "nll": args.lambda_nll,
        "cv_guard": args.lambda_cv_guard,
        "interaction_base": 0.05,
        "interaction_expert": 0.25,
        "interaction_gate": 5e-3,
        "interaction_regret": 0.1,
    }
    parameters = [(name, value) for name, value in model.named_parameters()
                  if value.requires_grad]
    print("component,value,weighted_value,total_grad,encoder,motion_model,gasf,sean,decoder_other")
    for component, value in trainer.last_loss_tensors.items():
        weighted = value * coefficients.get(component, 0.0)
        if not weighted.requires_grad:
            norms = {key: 0.0 for key in
                     ("encoder", "motion_model", "gasf", "sean", "decoder_other")}
            total_norm = 0.0
        else:
            gradients = torch.autograd.grad(
                weighted, [parameter for _, parameter in parameters],
                retain_graph=True, allow_unused=True,
            )
            sums = {key: 0.0 for key in
                    ("encoder", "motion_model", "gasf", "sean", "decoder_other")}
            for (name, _), gradient in zip(parameters, gradients):
                if gradient is not None:
                    sums[group_name(name)] += float(gradient.square().sum())
            norms = {key: value ** 0.5 for key, value in sums.items()}
            total_norm = sum(sums.values()) ** 0.5
        fields = [component, float(value.detach()), float(weighted.detach()), total_norm]
        fields.extend(norms[key] for key in
                      ("encoder", "motion_model", "gasf", "sean", "decoder_other"))
        print(",".join([fields[0]] + [f"{number:.8g}" for number in fields[1:]]))
    print(f"total_loss,{float(total.detach()):.8g}")


if __name__ == "__main__":
    main()
