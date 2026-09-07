"""Pretrain the formal physical-interaction selector on cached causal features."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextSelector(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
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


def metrics(logits, labels):
    scores = torch.sigmoid(logits)
    truth = labels > 0.5
    prediction = scores >= 0.5
    positive_recall = prediction[truth].float().mean()
    negative_recall = (~prediction[~truth]).float().mean()
    return {
        "count": int(labels.numel()),
        "positive_fraction": float(labels.mean()),
        "balanced_accuracy": float(0.5 * (positive_recall + negative_recall)),
        "positive_score_mean": float(scores[truth].mean()),
        "negative_score_mean": float(scores[~truth].mean()),
        "score_separation": float(scores[truth].mean() - scores[~truth].mean()),
        "score_std": float(scores.std(unbiased=False)),
    }


def fold_standardization(selector, mean, std):
    """Fold x'=(x-mean)/std into the first affine layer."""
    first = selector.network[0]
    with torch.no_grad():
        standardized_weight = first.weight.clone()
        first.weight.copy_(standardized_weight / std.unsqueeze(0))
        first.bias.sub_((standardized_weight * (mean / std).unsqueeze(0)).sum(dim=1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--activation_floor", type=float, default=0.75)
    parser.add_argument(
        "--gasf_fusion_mode", choices=["add", "concat"], default="concat"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    device = torch.device(cli.device)
    cached = torch.load(cli.features, map_location="cpu", weights_only=False)
    train = cached["train"]
    validation = cached["validation"]
    train_features = train["features"].to(device)
    train_labels = train["labels"].to(device)
    val_features = validation["features"].to(device)
    val_labels = validation["labels"].to(device)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_train = (train_features - mean) / std
    normalized_val = (val_features - mean) / std

    selector = ContextSelector(cli.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=cli.lr, weight_decay=cli.weight_decay
    )
    best_state = None
    best_validation_loss = float("inf")
    best_step = -1
    for step in range(int(cli.steps)):
        selector.train()
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_loss(selector(normalized_train), train_labels)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == int(cli.steps) - 1:
            selector.eval()
            with torch.no_grad():
                val_loss = float(balanced_loss(
                    selector(normalized_val), val_labels
                ))
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in selector.state_dict().items()
                }

    selector.load_state_dict(best_state)
    selector.eval()
    with torch.no_grad():
        train_result = metrics(selector(normalized_train), train_labels)
        val_result = metrics(selector(normalized_val), val_labels)
    fold_standardization(selector, mean, std)
    selector = selector.cpu()

    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    state = dict(payload["state_dict"])
    prefix = "decoder.GASF.context_activation_head."
    state[prefix + "0.weight"] = selector.network[0].weight.detach().clone()
    state[prefix + "0.bias"] = selector.network[0].bias.detach().clone()
    state[prefix + "2.weight"] = selector.network[2].weight.detach().clone()
    state[prefix + "2.bias"] = selector.network[2].bias.detach().clone()
    payload["state_dict"] = state
    payload["interaction_physical_residual"] = True
    payload["interaction_physical_max_gain"] = float(
        payload.get("interaction_physical_max_gain", 0.075)
    )
    payload["interaction_physical_init_gain"] = float(
        payload.get("interaction_physical_init_gain", 0.05)
    )
    payload["interaction_activation_floor"] = float(cli.activation_floor)
    payload["gasf_fusion_mode"] = cli.gasf_fusion_mode
    payload["activation_pretraining"] = {
        "features": str(Path(cli.features).resolve()),
        "steps": int(cli.steps),
        "best_step": int(best_step),
        "best_validation_loss": best_validation_loss,
        "hidden_dim": int(cli.hidden_dim),
        "activation_floor": float(cli.activation_floor),
        "gasf_fusion_mode": cli.gasf_fusion_mode,
        "standardization_folded": True,
    }
    output_checkpoint = Path(cli.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_checkpoint)

    report = {
        "source_checkpoint": str(Path(cli.checkpoint).resolve()),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "best_step": int(best_step),
        "best_validation_loss": best_validation_loss,
        "activation_floor": float(cli.activation_floor),
        "gasf_fusion_mode": cli.gasf_fusion_mode,
        "train": train_result,
        "validation": val_result,
    }
    output_report = Path(cli.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
