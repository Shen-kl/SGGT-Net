"""Stress-test graph-interaction gain beyond the activation range."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.compare_sean_temporal_v3 import EventAccumulator
from evaluation.sweep_activation_oracle_v3 import build_runtime


def evaluate_gain(model, trainer, loader, gain):
    model.decoder.GASF.interaction_activation_override = 1.0
    model.decoder.GASF.interaction_gain_override = float(gain)
    accumulator = EventAccumulator(trainer.config)
    latest = {}

    def gasf_hook(_module, _inputs, output):
        latest["output"] = output

    def observer(**values):
        accumulator.add(
            values["outputs"], values["event_class"], latest["output"]
        )

    hook = model.decoder.GASF.register_forward_hook(gasf_hook)
    trainer.step_observer = observer
    try:
        metrics = trainer.validate(loader, f"interaction gain={gain:.3f}")
    finally:
        hook.remove()
        trainer.step_observer = None
        model.decoder.GASF.interaction_activation_override = None
        model.decoder.GASF.interaction_gain_override = None
        model.eval()
    return {"global": metrics, "events": accumulator.result()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_trajectories", type=int, default=8)
    parser.add_argument("--start_trajectory", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--gains", nargs="+", type=float, default=[0, 1, 2, 4, 8]
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    if any(not 0.0 <= gain < float("inf") for gain in cli.gains):
        raise ValueError("all gains must be finite and non-negative")
    device = torch.device(cli.device)
    payload, _args, _config, model, trainer, dataset, loader = build_runtime(
        cli.checkpoint,
        cli.data_dir,
        cli.max_trajectories,
        device,
        start_trajectory=cli.start_trajectory,
    )
    result = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -999)),
        "trajectory_count": len(dataset),
        "protocol": {
            "online_rollout": True,
            "encoder_mode": "stream",
            "evaluation_batch_size": 1,
            "gasf_fusion_mode": "concat",
            "activation_override": 1.0,
            "interaction_gain_values": cli.gains,
            "training_parameters_changed": False,
            "validation_start_index": cli.start_trajectory,
        },
        "gains": {},
    }
    for gain in cli.gains:
        result["gains"][f"{gain:.6f}"] = evaluate_gain(
            model, trainer, loader, gain
        )
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
