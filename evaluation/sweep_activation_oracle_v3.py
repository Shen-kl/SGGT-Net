"""Fixed-alpha sweep and non-causal closed-loop activation oracle for V9."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from evaluation.compare_sean_temporal_v3 import (
    EventAccumulator,
    build_model,
    load_compatible_state_dict,
)
from train.trainer_v3 import Trainer
from train.training_utils_v3 import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


def build_runtime(
    checkpoint,
    data_dir,
    max_trajectories,
    device,
    start_trajectory=0,
):
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
    args.adaptive_q_mode = "legacy"
    args.encoder_execution_mode = "stream"
    args.full_time_supervision = True
    args.eval_batch_size = 1
    args.report_trajectory_quantiles = False

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    trainer = Trainer(model, optimizer, scheduler, None, config, args)

    dataset = GroupTargetTrackingDatasetV3(
        data_dir, "val", load_structure=False
    )
    start_trajectory = max(0, int(start_trajectory))
    if start_trajectory >= len(dataset):
        raise ValueError(
            f"start_trajectory={start_trajectory} exceeds dataset size {len(dataset)}"
        )
    stop = len(dataset)
    if max_trajectories > 0:
        stop = min(start_trajectory + max_trajectories, len(dataset))
    if start_trajectory > 0 or stop < len(dataset):
        dataset = Subset(dataset, range(start_trajectory, stop))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return payload, args, config, model, trainer, dataset, loader


def evaluate_fixed_alpha(model, trainer, loader, alpha):
    model.decoder.GASF.interaction_gain_override = None
    model.decoder.GASF.interaction_activation_override = float(alpha)
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
        global_metrics = trainer.validate(loader, f"fixed alpha={alpha:.3f}")
    finally:
        hook.remove()
        trainer.step_observer = None
        model.decoder.GASF.interaction_activation_override = None
        model.decoder.GASF.interaction_gain_override = None
        model.eval()
    return {"global": global_metrics, "events": accumulator.result()}


def _select_nodes(base, expert, use_expert):
    if base.shape != expert.shape or base.size(0) != use_expert.numel():
        raise ValueError("oracle branch tensors are not node-aligned")
    view = (use_expert.numel(),) + (1,) * (base.ndim - 1)
    return torch.where(use_expert.view(view), expert, base)


def _oracle_trajectory(
    model,
    trainer,
    graph_data,
    accumulator,
    latest,
    rollout_selected,
):
    graph_data = graph_data.to(trainer.device, non_blocking=True)
    measurement_covariance, _ = trainer.compute_measurement_noise(graph_data, 0)
    covariance = trainer.init_state_covariance(measurement_covariance)
    input_len = trainer.config["input_graph_window_len"]
    output_len = trainer.config["output_graph_window_len"]
    last_time_step = graph_data.y.size(1) - output_len
    window = GraphDataProcessor.init_input_window(
        graph_data, trainer.config, trainer.args, start_bias=0
    )
    normalized_window = window.clone()
    encoder_output_cache = None
    encoder_hidden_cache = None

    for time_step in range(input_len, last_time_step + 1):
        normalized_window, centroids = minmax_scaler(
            window,
            normalized_window,
            trainer.config,
            trainer.args.neural_net,
        )
        measurement_r, normalized_r = trainer.compute_measurement_noise(
            graph_data, time_step
        )
        if encoder_output_cache is None:
            rolling_encoder = None
            step_mixture = None
        else:
            encoder_step = normalized_window.clone()
            encoder_step.x = normalized_window.x[:, -1:, :]
            encoder_step.edge_index = normalized_window.edge_index[-1:]
            encoder_step.edge_features = normalized_window.edge_features[-1:]
            step_output, step_mixture = model.encoder(
                encoder_step, encoder_hidden_cache
            )
            new_hidden = step_output[:, -1]
            rolling_encoder = torch.cat(
                (encoder_output_cache[:, 1:], new_hidden.unsqueeze(1)), dim=1
            )

        branch_outputs = []
        for alpha in (0.0, 1.0):
            model.decoder.GASF.interaction_activation_override = alpha
            branch_outputs.append(model(
                normalized_window,
                0.0,
                covariance,
                None,
                normalized_r,
                encoder_output_cache=rolling_encoder,
                mixture_coeffs_cache=step_mixture,
            ))
        base, expert = branch_outputs
        encoder_output_cache = base.encoder_output_cache.detach()
        encoder_hidden_cache = base.encoder_hidden.detach()

        target = base.target[:, 0]
        scale = float(trainer.config["S_POS"])
        base_error = (
            (base.states[:, 0, 0, :2] - target[:, :2]) * scale
        ).square().sum(dim=-1)
        expert_error = (
            (expert.states[:, 0, 0, :2] - target[:, :2]) * scale
        ).square().sum(dim=-1)
        valid = base.real_mask[:, 0].bool()
        use_expert = valid & (expert_error < base_error)

        selected = base._replace(
            states=_select_nodes(base.states, expert.states, use_expert),
            covariances=_select_nodes(
                base.covariances, expert.covariances, use_expert
            ),
            updated_state=_select_nodes(
                base.updated_state, expert.updated_state, use_expert
            ),
            updated_covariance=_select_nodes(
                base.updated_covariance, expert.updated_covariance, use_expert
            ),
            gasf_confidence=use_expert.to(base.states.dtype).view(-1, 1, 1),
        )
        event_class = trainer._event_node_class(graph_data, time_step)
        accumulator.add(selected, event_class, latest["output"])

        # The common-history oracle measures the pointwise branch envelope on
        # a fixed temporal causal state. The greedy oracle instead propagates
        # each selected prior/posterior. Only the former is a same-state local
        # upper bound; greedy choices need not be globally optimal over time.
        history_output = selected if rollout_selected else base
        posterior_raw = trainer._denormalize_update(
            history_output, centroids, graph_data.batch
        )
        current_valid = graph_data.real_mask[:, time_step, 0].bool()
        previous_valid = graph_data.real_mask[:, time_step - 1, 0].bool()
        continuing = current_valid & previous_valid
        births = current_valid & ~previous_valid
        next_covariance = covariance.detach().clone()
        next_covariance[continuing] = (
            history_output.updated_covariance.detach()[continuing]
        )
        if births.any():
            birth_covariance = trainer.init_state_covariance(measurement_r)
            next_covariance[births] = birth_covariance[births]
        covariance = next_covariance

        if time_step < last_time_step:
            prior_raw = trainer._denormalize_prior(
                history_output, centroids, graph_data.batch
            )
            bias = time_step - input_len + 1
            window = GraphDataProcessor.update_input_window(
                window,
                graph_data,
                trainer.config,
                posterior_raw,
                prior_raw,
                bias,
                trainer.args,
            )


def evaluate_oracle(model, trainer, loader, rollout_selected):
    accumulator = EventAccumulator(trainer.config)
    latest = {}

    def gasf_hook(_module, _inputs, output):
        latest["output"] = output

    hook = model.decoder.GASF.register_forward_hook(gasf_hook)
    model.eval()
    try:
        with torch.no_grad():
            for graph_data in loader:
                _oracle_trajectory(
                    model,
                    trainer,
                    graph_data,
                    accumulator,
                    latest,
                    rollout_selected,
                )
    finally:
        hook.remove()
        model.decoder.GASF.interaction_activation_override = None
        model.decoder.GASF.interaction_gain_override = None
        model.eval()
    return {"events": accumulator.result()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_trajectories", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--skip_oracle", action="store_true",
        help="run only the fixed-alpha sweep",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    if any(alpha < 0.0 or alpha > 1.0 for alpha in cli.alphas):
        raise ValueError("all alpha values must lie in [0, 1]")
    device = torch.device(cli.device)
    payload, args, config, model, trainer, dataset, loader = build_runtime(
        cli.checkpoint,
        cli.data_dir,
        cli.max_trajectories,
        device,
    )
    results = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -999)),
        "trajectory_count": len(dataset),
        "protocol": {
            "online_rollout": True,
            "encoder_mode": "stream",
            "evaluation_batch_size": 1,
            "gasf_fusion_mode": "concat",
            "fixed_alpha_values": cli.alphas,
            "common_history_oracle": (
                "non-causal per-node prior-position choice evaluated on a "
                "fixed alpha=0 causal history; local same-state upper bound"
            ),
            "closed_loop_greedy_oracle": (
                "the same non-causal per-node greedy choice with its selected "
                "prior, EKF posterior, covariance and residual rolled forward; "
                "not a global dynamic-programming upper bound"
            ),
        },
        "fixed_alpha": {},
    }
    for alpha in cli.alphas:
        key = f"{alpha:.6f}"
        results["fixed_alpha"][key] = evaluate_fixed_alpha(
            model, trainer, loader, alpha
        )
    if not cli.skip_oracle:
        results["common_history_oracle"] = evaluate_oracle(
            model, trainer, loader, rollout_selected=False
        )
        results["closed_loop_greedy_oracle"] = evaluate_oracle(
            model, trainer, loader, rollout_selected=True
        )

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
