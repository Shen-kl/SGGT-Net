"""Verify that all seed Temporal checkpoints share the frozen architecture."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_value(payload, key, default, expected, errors, seed):
    value = payload.get(key, default)
    if value != expected:
        errors.append(
            f"seed {seed}: {key}={value!r}, expected {expected!r}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", default="checkpoint/formal_innovation_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records, errors = [], []
    reference_shapes = None
    reference_protocol = None
    for seed in args.seeds:
        path = (
            Path(args.checkpoint_root) / f"seed_{seed}" / "A_temporal"
            / "GNN_groupTargetTracking" / "best_prediction.pth"
        ).resolve()
        if not path.exists():
            errors.append(f"seed {seed}: missing {path}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_value(payload, "model_variant", "sggt_v1", "sggt_v2", errors, seed)
        expected_value(payload, "decoder_use_GASF", False, True, errors, seed)
        expected_value(payload, "decoder_use_SEAN", True, False, errors, seed)
        expected_value(payload, "gasf_filter_mode", "learned", "abc", errors, seed)
        expected_value(payload, "gasf_graph_mode", "graph", "temporal_only", errors, seed)
        expected_value(payload, "gasf_fusion_mode", "film", "concat", errors, seed)
        expected_value(payload, "gasf_aux_source", "gasf", "fusion", errors, seed)
        expected_value(payload, "innovation_decomposition", False, False, errors, seed)
        expected_value(payload, "causal_innovation_correction", False, False, errors, seed)
        expected_value(payload, "covariance_transition", "ode_jacobian", "cv", errors, seed)
        protocol = payload.get("evaluation_protocol") or {}
        protocol_identity = (
            protocol.get("protocol_sha256"),
            int(protocol.get("version", -1)),
            protocol.get("split"),
            int(protocol.get("trajectory_count", -1)),
            protocol.get("validation_selection"),
            protocol.get("metric_reduction"),
            protocol.get("training_supervision"),
            protocol.get("encoder_execution_mode"),
            protocol.get("covariance_transition"),
        )
        if protocol_identity[1:] != (
            2, "val", 200, "all_legal_causal_steps",
            "global_valid_point_sse_then_sqrt", "all_legal_causal_steps",
            "stream", "cv",
        ):
            errors.append(
                f"seed {seed}: incompatible evaluation protocol {protocol_identity}"
            )
        if reference_protocol is None:
            reference_protocol = protocol_identity
        elif protocol_identity != reference_protocol:
            errors.append(
                f"seed {seed}: protocol differs from first seed: "
                f"{protocol_identity} != {reference_protocol}"
            )
        config = payload.get("config") or {}
        for key, expected in (
            ("input_graph_window_len", 8),
            ("output_graph_window_len", 1),
            ("S_VEL", 40.0),
            ("S_EDGE_V_DIST", 40.0),
        ):
            if float(config.get(key, float("nan"))) != float(expected):
                errors.append(
                    f"seed {seed}: config[{key}]={config.get(key)!r}, "
                    f"expected {expected!r}"
                )
        shapes = {
            key: list(value.shape)
            for key, value in payload["state_dict"].items()
        }
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            errors.append(f"seed {seed}: state_dict keys/shapes differ")
        metrics = payload.get("metrics") or {}
        for metric in (
            "location_rmse_prediction", "location_rmse_update",
            "velocity_rmse_prediction", "velocity_rmse_update",
        ):
            value = metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                errors.append(f"seed {seed}: invalid {metric}={value!r}")
        records.append({
            "seed": seed,
            "path": str(path),
            "sha256": sha256_file(path),
            "epoch": int(payload.get("epoch", -999)),
            "protocol_sha256": protocol.get("protocol_sha256"),
            "state_tensor_count": len(shapes),
            "metrics": {
                key: float(metrics[key]) for key in (
                    "location_rmse_prediction", "location_rmse_update",
                    "velocity_rmse_prediction", "velocity_rmse_update",
                ) if key in metrics
            },
        })
    report = {
        "compatible": not errors and len(records) == len(args.seeds),
        "seeds": list(args.seeds),
        "checkpoints": records,
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
