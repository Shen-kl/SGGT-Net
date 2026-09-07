"""Closed-loop calibration sweep for a learned physical-interaction selector."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.compare_sean_temporal_v3 import EventAccumulator
from evaluation.sweep_activation_oracle_v3 import build_runtime


def evaluate_floor(model, trainer, loader, floor):
    gasf = model.decoder.GASF
    gasf.interaction_activation_override = None
    gasf.interaction_activation_floor_override = float(floor)
    accumulator = EventAccumulator(trainer.config)
    latest = {}

    def gasf_hook(_module, _inputs, output):
        latest["output"] = output

    def observer(**values):
        accumulator.add(
            values["outputs"], values["event_class"], latest["output"],
            values.get("window"),
        )

    hook = gasf.register_forward_hook(gasf_hook)
    trainer.step_observer = observer
    try:
        metrics = trainer.validate(loader, f"activation floor={floor:.3f}")
    finally:
        hook.remove()
        trainer.step_observer = None
        gasf.interaction_activation_floor_override = None
        model.eval()
    return {"global": metrics, "events": accumulator.result()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_trajectories", type=int, default=16)
    parser.add_argument("--start_trajectory", type=int, default=0)
    parser.add_argument("--floors", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    if any(not 0.0 <= floor <= 1.0 for floor in cli.floors):
        raise ValueError("all floors must lie in [0, 1]")
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
            "validation_start_index": cli.start_trajectory,
            "gasf_fusion_mode": _args.gasf_fusion_mode,
            "calibration": "alpha'=floor+(1-floor)*alpha",
            "training_parameters_changed": False,
        },
        "floors": {},
    }
    for floor in cli.floors:
        result["floors"][f"{floor:.6f}"] = evaluate_floor(
            model, trainer, loader, floor
        )
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
