"""Validate a P1 (MTP-GO backbone) training run before promotion.

The gate is intentionally independent of the trainer's console output.  It
uses checkpoint metadata and tensors so a detached Windows launch cannot hide
NaN/Inf values or silently enable a removed correction branch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def finite_scalars(mapping: dict, prefix: str) -> list[str]:
    errors: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            errors.append(f"{prefix}.{key} is not finite: {value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-validation-trajectories", type=int, required=True)
    parser.add_argument("--minimum-prediction-improvement-m", type=float, default=0.05)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_dir = args.run_dir / "GNN_groupTargetTracking"
    initial_path = checkpoint_dir / "initial.pth"
    best_path = checkpoint_dir / "best_prediction.pth"
    last_path = checkpoint_dir / "last.pth"
    errors: list[str] = []
    for path in (initial_path, best_path, last_path):
        if not path.is_file():
            errors.append(f"missing checkpoint: {path}")

    report = {
        "status": "fail",
        "run_dir": str(args.run_dir.resolve()),
        "expected_validation_trajectories": args.expected_validation_trajectories,
        "minimum_prediction_improvement_m": args.minimum_prediction_improvement_m,
        "errors": errors,
    }
    if not errors:
        initial = load_checkpoint(initial_path)
        best = load_checkpoint(best_path)
        last = load_checkpoint(last_path)

        expected_metadata = {
            "model_variant": "sggt_v2",
            "covariance_transition": "cv",
            "decoder_use_GASF": False,
            "decoder_use_SEAN": False,
            "causal_innovation_correction": False,
            "innovation_decomposition": False,
            "adaptive_q_mode": "legacy",
        }
        for key, expected in expected_metadata.items():
            actual = best.get(key)
            if actual != expected:
                errors.append(f"{key}={actual!r}, expected {expected!r}")

        protocol = best.get("evaluation_protocol") or {}
        trajectory_count = protocol.get("trajectory_count")
        if trajectory_count != args.expected_validation_trajectories:
            errors.append(
                "validation trajectory count "
                f"{trajectory_count!r}, expected {args.expected_validation_trajectories}"
            )
        if protocol.get("training_supervision") != "all_legal_causal_steps":
            errors.append("training supervision is not all_legal_causal_steps")
        if protocol.get("encoder_execution_mode") != "stream":
            errors.append("encoder execution mode is not stream")

        initial_metrics = initial.get("metrics") or {}
        best_metrics = best.get("metrics") or {}
        last_metrics = last.get("metrics") or {}
        errors.extend(finite_scalars(initial_metrics, "initial.metrics"))
        errors.extend(finite_scalars(best_metrics, "best.metrics"))
        errors.extend(finite_scalars(last_metrics, "last.metrics"))

        for checkpoint_name, checkpoint in (
            ("initial", initial),
            ("best", best),
            ("last", last),
        ):
            for name, tensor in (checkpoint.get("state_dict") or {}).items():
                if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                    errors.append(f"{checkpoint_name}.state_dict.{name} contains NaN/Inf")
                    break

        state_keys = tuple((best.get("state_dict") or {}).keys())
        forbidden_prefixes = (
            "decoder.GASF.",
            "causal_innovation_corrector.",
        )
        for prefix in forbidden_prefixes:
            offending = next((key for key in state_keys if key.startswith(prefix)), None)
            if offending is not None:
                errors.append(f"forbidden P1 parameter present: {offending}")

        initial_prediction = float(
            initial_metrics.get("location_rmse_prediction", float("nan"))
        )
        best_prediction = float(
            best_metrics.get("location_rmse_prediction", float("nan"))
        )
        improvement = initial_prediction - best_prediction
        best_epoch = int(best.get("epoch", -1))
        if best_epoch < 0:
            errors.append("best prediction checkpoint is still the untrained epoch -1")
        if not math.isfinite(improvement):
            errors.append("initial-to-best prediction improvement is not finite")
        elif improvement < args.minimum_prediction_improvement_m:
            errors.append(
                f"prediction improvement {improvement:.6f} m is below "
                f"{args.minimum_prediction_improvement_m:.6f} m"
            )

        report.update(
            {
                "protocol_sha256": protocol.get("protocol_sha256"),
                "validation_trajectories": trajectory_count,
                "initial_epoch": int(initial.get("epoch", -1)),
                "best_epoch": best_epoch,
                "last_epoch": int(last.get("epoch", -1)),
                "initial_metrics": initial_metrics,
                "best_metrics": best_metrics,
                "last_metrics": last_metrics,
                "prediction_improvement_m": improvement,
                "trainable_state_tensor_count": len(state_keys),
            }
        )

    report["errors"] = errors
    report["status"] = "pass" if not errors else "fail"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
