"""Offline cross-trajectory predictability of physical interaction demand."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from evaluation.compare_sean_temporal_v3 import (
    build_model,
    load_compatible_state_dict,
)
from train.trainer_v3 import Trainer


def build_runtime(checkpoint, data_dir, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = device
    args.decoder_use_GASF = True
    args.decoder_use_SEAN = True
    args.model_variant = payload.get("model_variant", "sggt_v2")
    args.covariance_transition = payload.get(
        "covariance_transition", "ode_jacobian"
    )
    args.gasf_filter_mode = "abc"
    args.gasf_graph_mode = "graph"
    args.gasf_fusion_mode = payload.get("gasf_fusion_mode", "concat")
    args.gasf_aux_source = "gasf"
    args.counterfactual_training = True
    args.interaction_physical_residual = True
    args.interaction_physical_max_gain = float(
        payload.get("interaction_physical_max_gain", 0.075)
    )
    args.interaction_physical_init_gain = float(
        payload.get("interaction_physical_init_gain", 0.05)
    )
    args.interaction_activation_floor = float(
        payload.get("interaction_activation_floor", 0.0)
    )
    args.adaptive_q_mode = "legacy"
    args.encoder_execution_mode = "stream"
    args.full_time_supervision = True
    args.eval_batch_size = 1
    args.report_trajectory_quantiles = False
    args.training_stage = "gate"
    args.interaction_gate_position_margin_m = 0.05

    config = dict(payload.get("config", CONFIG))
    config.update({
        "S_VEL": 40.0,
        "S_EDGE_V_DIST": 40.0,
        "output_graph_window_len": 1,
    })
    CONFIG.update(config)
    model = build_model(args).to(device)
    load_compatible_state_dict(model, payload["state_dict"])
    model.eval()
    model.counterfactual_evaluation = True
    model.decoder.GASF.interaction_activation_override = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    trainer = Trainer(model, optimizer, scheduler, None, config, args)
    return payload, args, config, model, trainer


def collect_split(model, trainer, data_dir, mode, start, count, margin):
    dataset = GroupTargetTrackingDatasetV3(
        data_dir, mode, load_structure=False
    )
    stop = min(int(start) + int(count), len(dataset))
    subset = Subset(dataset, range(int(start), stop))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    features, labels, advantages, events = [], [], [], []

    def observer(**values):
        outputs = values["outputs"]
        valid = outputs.real_mask[:, 0].bool()
        causal = model.decoder.GASF.last_context_activation_features
        if causal is None or not valid.any():
            return
        target = outputs.target[:, 0]
        base = outputs.counterfactual_base[:, 0, 0]
        expert = outputs.counterfactual_expert[:, 0, 0]
        scale = float(trainer.config["S_POS"])
        base_error = torch.linalg.vector_norm(
            (base[:, :2] - target[:, :2]) * scale, dim=-1
        )
        expert_error = torch.linalg.vector_norm(
            (expert[:, :2] - target[:, :2]) * scale, dim=-1
        )
        advantage = base_error - expert_error
        informative = valid & (advantage.abs() > float(margin))
        if not informative.any():
            return
        features.append(causal[informative].cpu())
        advantages.append(advantage[informative].detach().cpu())
        labels.append((advantage[informative] > float(margin)).float().cpu())
        events.append(values["event_class"][informative].detach().cpu())

    trainer.step_observer = observer
    try:
        trainer.validate(loader, f"collect activation {mode}[{start}:{stop}]")
    finally:
        trainer.step_observer = None
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "advantages": torch.cat(advantages),
        "events": torch.cat(events),
        "trajectory_count": len(subset),
    }


class TinySelector(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.network = (
            nn.Linear(8, 1)
            if hidden <= 0
            else nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def balanced_loss(logits, labels):
    positive = labels > 0.5
    negative = ~positive
    return 0.5 * (
        F.binary_cross_entropy_with_logits(logits[positive], labels[positive])
        + F.binary_cross_entropy_with_logits(logits[negative], labels[negative])
    )


def classification_metrics(logits, labels, events):
    scores = torch.sigmoid(logits)
    truth = labels > 0.5
    prediction = scores >= 0.5
    positive_recall = (prediction[truth]).float().mean()
    negative_recall = (~prediction[~truth]).float().mean()
    result = {
        "count": int(labels.numel()),
        "positive_fraction": float(labels.mean()),
        "balanced_accuracy": float(0.5 * (positive_recall + negative_recall)),
        "positive_score_mean": float(scores[truth].mean()),
        "negative_score_mean": float(scores[~truth].mean()),
        "score_separation": float(scores[truth].mean() - scores[~truth].mean()),
        "score_std": float(scores.std(unbiased=False)),
    }
    for value, name in ((0, "steady"), (1, "switch"), (2, "strong")):
        mask = events == value
        if mask.any():
            result[f"{name}_count"] = int(mask.sum())
            result[f"{name}_positive_fraction"] = float(labels[mask].mean())
    return result


def fit_selector(train, val, hidden, steps, lr, seed):
    torch.manual_seed(seed)
    mean = train["features"].mean(dim=0)
    std = train["features"].std(dim=0, unbiased=False).clamp_min(1e-6)
    train_features = (train["features"] - mean) / std
    val_features = (val["features"] - mean) / std
    selector = TinySelector(hidden)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_loss(selector(train_features), train["labels"])
        loss.backward()
        optimizer.step()
    selector.eval()
    with torch.no_grad():
        train_logits = selector(train_features)
        val_logits = selector(val_features)
    return {
        "hidden": int(hidden),
        "train": classification_metrics(
            train_logits, train["labels"], train["events"]
        ),
        "validation": classification_metrics(
            val_logits, val["labels"], val["events"]
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature_output", required=True)
    parser.add_argument("--train_count", type=int, default=32)
    parser.add_argument("--val_start", type=int, default=48)
    parser.add_argument("--val_count", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    device = torch.device(cli.device)
    payload, _args, _config, model, trainer = build_runtime(
        cli.checkpoint, cli.data_dir, device
    )
    train = collect_split(
        model, trainer, cli.data_dir, "train", 0, cli.train_count, cli.margin
    )
    val = collect_split(
        model, trainer, cli.data_dir, "val", cli.val_start, cli.val_count, cli.margin
    )
    feature_path = Path(cli.feature_output)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train, "validation": val}, feature_path)
    results = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -999)),
        "margin_m": float(cli.margin),
        "train_trajectories": train["trajectory_count"],
        "validation_start": int(cli.val_start),
        "validation_trajectories": val["trajectory_count"],
        "models": {
            "linear": fit_selector(
                train, val, hidden=0, steps=cli.steps, lr=cli.lr, seed=cli.seed
            ),
            "mlp16": fit_selector(
                train, val, hidden=16, steps=cli.steps, lr=cli.lr, seed=cli.seed
            ),
        },
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
