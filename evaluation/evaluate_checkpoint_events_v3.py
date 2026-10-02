"""Evaluate any formal V3 checkpoint with identical online event metrics."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from evaluation.compare_sean_temporal_v3 import (
    build_model,
    evaluate_mode,
    load_compatible_state_dict,
)
from train.trainer_v3 import Trainer


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=["val", "iid_test", "ood_test"])
    parser.add_argument("--start_trajectory", type=int, default=0)
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--trajectory_sse", action="store_true",
        help="save per-trajectory/event SSE and counts for paired bootstrap",
    )
    parser.add_argument(
        "--causal_alpha", type=float, default=1.0,
        help=(
            "legacy direct-state CIPIC option; acceleration-RK4 checkpoints "
            "require the default value 1"
        ),
    )
    parser.add_argument(
        "--causal_velocity_alpha", type=float, default=1.0,
        help=(
            "legacy direct-state CIPIC option; acceleration-RK4 checkpoints "
            "require the default value 1"
        ),
    )
    parser.add_argument(
        "--association_augmentation", action="store_true",
        help="run the deterministic missed-detection/local-clutter rollout",
    )
    parser.add_argument(
        "--association_seed", type=int, default=20260813,
        help="fixed RNG seed for --association_augmentation",
    )
    parser.add_argument(
        "--process_noise_scale", type=float, default=1.0,
        help="evaluation-only multiplier for learned acceleration Q",
    )
    parser.add_argument(
        "--position_q_variance", type=float,
        help="optional fixed direct position-Q variance per axis in m^2",
    )
    parser.add_argument(
        "--q_response_scale", type=float,
        help=(
            "evaluation-only scale around fixed Q means: 0=fixed mean, "
            "1=learned response, negative=inverted response"
        ),
    )
    parser.add_argument(
        "--q_response_acceleration_center", type=float, default=8.0,
    )
    parser.add_argument(
        "--q_response_position_center", type=float, default=1.0,
    )
    parser.add_argument(
        "--gasf_recurrent_alpha", type=float,
        help=(
            "enable previous-ABC recurrence with a fixed evaluation-only "
            "update coefficient in [0,1]"
        ),
    )
    cli = parser.parse_args()
    device = torch.device(cli.device)
    checkpoint_path = Path(cli.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    causal_mode = payload.get("causal_innovation_mode")
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = device
    args.decoder_use_GASF = bool(payload.get("decoder_use_GASF", True))
    args.decoder_use_SEAN = bool(payload.get("decoder_use_SEAN", False))
    args.model_variant = payload.get("model_variant", "sggt_v2")
    args.covariance_transition = payload.get("covariance_transition", "cv")
    args.gasf_filter_mode = payload.get("gasf_filter_mode", "abc")
    args.gasf_graph_mode = payload.get("gasf_graph_mode", "temporal_only")
    if (
        cli.gasf_recurrent_alpha is not None
        and not 0.0 <= cli.gasf_recurrent_alpha <= 1.0
    ):
        raise ValueError("gasf_recurrent_alpha must be in [0, 1]")
    args.gasf_recurrent_filter = bool(
        payload.get("gasf_recurrent_filter", False)
        or cli.gasf_recurrent_alpha is not None
    )
    args.gasf_recurrent_gate_initial_update = float(
        payload.get("gasf_recurrent_gate_initial_update", 0.1)
    )
    args.gasf_fusion_mode = payload.get("gasf_fusion_mode", "concat")
    args.gasf_aux_source = payload.get("gasf_aux_source", "fusion")
    args.innovation_decomposition = bool(
        payload.get("innovation_decomposition", False)
    )
    args.causal_innovation_correction = bool(
        payload.get("causal_innovation_correction", False)
    )
    args.causal_innovation_history = int(
        payload.get("causal_innovation_history", 8)
    )
    args.causal_innovation_hidden_size = int(
        payload.get("causal_innovation_hidden_size", 32)
    )
    args.causal_innovation_max_acceleration_mps2 = float(
        payload.get("causal_innovation_max_acceleration_mps2", 5.0)
    )
    args.causal_innovation_max_position_m = float(
        payload.get("causal_innovation_max_position_m", 50.0)
    )
    args.causal_innovation_max_velocity_mps = float(
        payload.get("causal_innovation_max_velocity_mps", 5.0)
    )
    args.causal_innovation_dynamic_activation = bool(
        payload.get("causal_innovation_dynamic_activation", False)
    )
    args.causal_innovation_activation_floor = float(
        payload.get("causal_innovation_activation_floor", 0.5)
    )
    args.causal_innovation_activation_nis_low = float(
        payload.get("causal_innovation_activation_nis_low", 1.38629436112)
    )
    args.causal_innovation_activation_nis_high = float(
        payload.get("causal_innovation_activation_nis_high", 5.99146454711)
    )
    args.causal_innovation_minimum_valid_history = int(
        payload.get("causal_innovation_minimum_valid_history", 0)
    )
    args.causal_innovation_recent_valid_window = int(
        payload.get("causal_innovation_recent_valid_window", 3)
    )
    args.causal_innovation_association_context = bool(
        payload.get("causal_innovation_association_context", False)
    )
    args.causal_innovation_use_peer = bool(
        payload.get("causal_innovation_use_peer", True)
    )
    # Association augmentation has no trainable parameters.  Enable it at
    # evaluation time for older Temporal checkpoints so all methods experience
    # the same fixed missed-detection/local-clutter realization.
    args.association_augmentation = bool(
        payload.get("association_augmentation", False)
        or cli.association_augmentation
    )
    args.association_pd_min = float(payload.get("association_pd_min", 0.75))
    args.association_pd_max = float(payload.get("association_pd_max", 0.98))
    args.association_clutter_mu_min = float(
        payload.get("association_clutter_mu_min", 0.0)
    )
    args.association_clutter_mu_max = float(
        payload.get("association_clutter_mu_max", 0.0)
    )
    args.association_surveillance_side_m = float(
        payload.get("association_surveillance_side_m", 3000.0)
    )
    args.association_max_clutter = int(payload.get("association_max_clutter", 32))
    args.association_local_clutter_rate = float(
        payload.get("association_local_clutter_rate", 0.25)
    )
    args.association_max_local_clutter = int(
        payload.get("association_max_local_clutter", 2)
    )
    args.association_local_clutter_scale = float(
        payload.get("association_local_clutter_scale", 2.5)
    )
    args.association_gate_threshold = float(
        payload.get("association_gate_threshold", 9.21034037198)
    )
    args.association_minimum_detection_mass = float(
        payload.get("association_minimum_detection_mass", 0.05)
    )
    args.association_include_target_competitors = bool(
        payload.get("association_include_target_competitors", False)
    )
    args.association_clutter_requires_detection = bool(
        payload.get("association_clutter_requires_detection", True)
    )
    args.innovation_whiten_clip = float(
        payload.get("innovation_whiten_clip", 20.0)
    )
    args.counterfactual_training = False
    args.interaction_physical_residual = bool(
        payload.get("interaction_physical_residual", False)
    )
    args.interaction_physical_max_gain = float(
        payload.get("interaction_physical_max_gain", 0.075)
    )
    args.interaction_physical_init_gain = float(
        payload.get("interaction_physical_init_gain", 0.05)
    )
    args.interaction_activation_floor = float(
        payload.get("interaction_activation_floor", 0.0)
    )
    args.adaptive_q_mode = payload.get("adaptive_q_mode", "legacy")
    args.adaptive_q_max_multiplier = float(payload.get(
        "adaptive_q_max_multiplier", 5.0
    ))
    args.adaptive_q_max_position_variance = float(payload.get(
        "adaptive_q_max_position_variance", 400.0
    ))
    args.adaptive_q_initial_multiplier = float(payload.get(
        "adaptive_q_initial_multiplier", 1.0
    ))
    args.adaptive_q_initial_position_variance = float(payload.get(
        "adaptive_q_initial_position_variance", 0.01
    ))
    args.adaptive_q_acceleration = bool(payload.get(
        "adaptive_q_acceleration", True
    ))
    args.encoder_execution_mode = "stream"
    args.full_time_supervision = True
    if cli.batch_size < 1:
        raise ValueError("batch_size must be positive")
    args.eval_batch_size = int(cli.batch_size)
    args.report_trajectory_quantiles = False
    args.association_validation_seed = int(cli.association_seed)

    config = dict(payload.get("config", CONFIG))
    config.update({
        "S_VEL": 40.0,
        "S_EDGE_V_DIST": 40.0,
        "output_graph_window_len": 1,
    })
    CONFIG.update(config)
    model = build_model(args).to(device)
    load_compatible_state_dict(model, payload["state_dict"])
    gasf = getattr(model.decoder, "GASF", None)
    if gasf is None and cli.gasf_recurrent_alpha is not None:
        raise ValueError("gasf_recurrent_alpha requires a GASF checkpoint")
    if gasf is not None:
        gasf.parameter_update_gate_override = cli.gasf_recurrent_alpha
    if cli.process_noise_scale < 0.0:
        raise ValueError("process_noise_scale must be non-negative")
    if cli.position_q_variance is not None and cli.position_q_variance < 0.0:
        raise ValueError("position_q_variance must be non-negative")
    model.process_noise_mode = (
        "zero" if cli.process_noise_scale == 0.0 else "scaled"
    )
    model.process_noise_scale = float(cli.process_noise_scale)
    model.decoder.q_position_variance_override_physical = (
        None if cli.position_q_variance is None
        else float(cli.position_q_variance)
    )
    model.decoder.q_response_scale_override = cli.q_response_scale
    model.decoder.q_response_acceleration_center = float(
        cli.q_response_acceleration_center
    )
    model.decoder.q_response_position_center = float(
        cli.q_response_position_center
    )
    if (
        args.causal_innovation_correction
        and causal_mode == "acceleration_rk4_raw"
        and (
            float(cli.causal_alpha) != 1.0
            or float(cli.causal_velocity_alpha) != 1.0
        )
    ):
        raise ValueError(
            "causal_alpha/causal_velocity_alpha do not apply to the "
            "acceleration-RK4 CIPIC architecture"
        )
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    trainer = Trainer(model, optimizer, scheduler, None, config, args)

    dataset = GroupTargetTrackingDatasetV3(
        cli.data_dir, cli.split, load_structure=False,
        load_analysis_metadata=False,
    )
    start = max(0, int(cli.start_trajectory))
    stop = len(dataset)
    if cli.max_trajectories > 0:
        stop = min(stop, start + int(cli.max_trajectories))
    if start >= stop:
        raise ValueError(f"empty trajectory range [{start}, {stop})")
    loader = DataLoader(
        Subset(dataset, range(start, stop)), batch_size=cli.batch_size, shuffle=False,
        num_workers=0,
    )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    result = evaluate_mode(
        model, trainer, loader, "expert",
        collect_trajectory_sse=cli.trajectory_sse,
        association_augmentation=cli.association_augmentation,
        association_seed=cli.association_seed,
    )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    valid_points = float(
        result["global"].get("evaluated_prediction_points", 0.0)
    )
    report = {
        "protocol": {
            "version": 1,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch", -999)),
            "dataset_root": str(Path(cli.data_dir).resolve()),
            "split": cli.split,
            "trajectory_start": start,
            "trajectory_stop_exclusive": stop,
            "trajectory_count": stop - start,
            "batch_size": int(cli.batch_size),
            "process_noise_scale": float(cli.process_noise_scale),
            "position_q_variance_per_axis_m2": cli.position_q_variance,
            "q_response_scale": cli.q_response_scale,
            "q_response_acceleration_center": float(
                cli.q_response_acceleration_center
            ),
            "q_response_position_center": float(
                cli.q_response_position_center
            ),
            "gasf_recurrent_alpha_override": cli.gasf_recurrent_alpha,
            "online_rollout": True,
            "association_augmentation": bool(cli.association_augmentation),
            "association_seed": (
                int(cli.association_seed) if cli.association_augmentation else None
            ),
            "operational_interaction": (
                "eligible_raw_innovation_acceleration_increment"
                if (
                    args.causal_innovation_correction
                    and causal_mode == "acceleration_rk4_raw"
                ) else (
                    "legacy_causal_innovation_correction"
                    if args.causal_innovation_correction else (
                    "alpha=1" if args.innovation_decomposition
                    else "configured graph mode"
                    )
                )
            ),
        },
        "model": {
            "decoder_use_GASF": args.decoder_use_GASF,
            "decoder_use_SEAN": args.decoder_use_SEAN,
            "gasf_graph_mode": args.gasf_graph_mode,
            "gasf_recurrent_filter": args.gasf_recurrent_filter,
            "gasf_recurrent_gate_initial_update": (
                args.gasf_recurrent_gate_initial_update
            ),
            "innovation_decomposition": args.innovation_decomposition,
            "causal_innovation_correction": args.causal_innovation_correction,
            "causal_innovation_mode": causal_mode,
            "causal_innovation_max_acceleration_mps2": float(
                args.causal_innovation_max_acceleration_mps2
            ),
            "causal_innovation_dynamic_activation": (
                args.causal_innovation_dynamic_activation
            ),
            "causal_innovation_alpha": float(cli.causal_alpha),
            "causal_innovation_velocity_alpha": float(
                cli.causal_velocity_alpha
            ),
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
        "timing": {
            "total_seconds": elapsed,
            "milliseconds_per_valid_node_step": (
                1000.0 * elapsed / valid_points if valid_points else None
            ),
        },
        "global": {
            key: float(value) for key, value in result["global"].items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        "events": result["events"],
    }
    if cli.trajectory_sse:
        report["trajectories"] = result["trajectories"]
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
