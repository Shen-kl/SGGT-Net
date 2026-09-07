"""Numerically compare the GASF path with and without SEAN on v3 data."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from model.sggt_net import SGGT_Net
from train.trainer_v3 import Trainer


class TensorStats:
    def __init__(self):
        self.parts = defaultdict(list)

    def add(self, name, value):
        if value is None or not torch.is_tensor(value) or value.numel() == 0:
            return
        flat = value.detach().float().reshape(-1).cpu()
        flat = flat[torch.isfinite(flat)]
        if flat.numel():
            self.parts[name].append(flat)

    def summarize(self):
        result = {}
        for name, parts in self.parts.items():
            values = torch.cat(parts).numpy()
            result[name] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "p01": float(np.quantile(values, 0.01)),
                "p10": float(np.quantile(values, 0.10)),
                "p50": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
                "p99": float(np.quantile(values, 0.99)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        return result


def build_model(args, use_sean):
    motion_model = SecondOrderNeuralODE_groupTrack(
        solver=args.ode_solver,
        dt=args.T,
        mixtures=args.n_mixtures,
        static_f_dim=0,
        n_hidden=args.n_ode_hidden,
        n_layers=args.n_ode_layers,
        use_SEAN=use_sean,
        use_GASF=True,
        gasf_fusion_mode="concat",
        gasf_aux_source="gasf",
    )
    return SGGT_Net(
        args.encoder_input_size,
        args.encoder_hidden_size,
        args.encoder_n_heads,
        args.encoder_n_layers,
        motion_model.mixtures,
        args.encoder_dropout,
        args.encoder_gnn_layer,
        args.encoder_use_edge_features,
        motion_model,
        args.decoder_max_length,
        args.decoder_hidden_size,
        args.decoder_n_heads,
        args.decoder_n_layers,
        args.decoder_alpha,
        args.decoder_dropout,
        args.decoder_residual_length,
        args.decoder_z_dimension,
        args.decoder_gnn_layer,
        True,
        use_sean,
        args.T,
        model_variant="sggt_v2",
        covariance_transition="cv",
        gasf_filter_mode="abc",
        gasf_graph_mode="graph",
        adaptive_q_mode="legacy",
    )


def evaluate(
    checkpoint,
    data_dir,
    use_sean,
    max_trajectories,
    device,
    sean_intervention="none",
):
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = torch.device(device)
    args.decoder_use_GASF = True
    args.decoder_use_SEAN = use_sean
    args.model_variant = "sggt_v2"
    args.covariance_transition = "cv"
    args.gasf_filter_mode = "abc"
    args.gasf_graph_mode = "graph"
    args.gasf_fusion_mode = "concat"
    args.gasf_aux_source = "gasf"
    args.adaptive_q_mode = "legacy"
    args.encoder_execution_mode = "stream"
    args.full_time_supervision = True
    args.eval_batch_size = 1
    args.report_trajectory_quantiles = False

    config = dict(CONFIG)
    config.update({
        "S_VEL": 40.0,
        "S_EDGE_V_DIST": 40.0,
        "output_graph_window_len": 1,
        "window_sampling_ratio": [0.4, 0.3, 0.3],
    })
    CONFIG.update(config)

    model = build_model(args, use_sean).to(args.device)
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if use_sean and sean_intervention != "none":
        original_sean_forward = model.decoder.SEAN.forward

        def intervened_sean(
            batch,
            states,
            hidden,
            mask=None,
            candidate_edge_index=None,
            candidate_edge_weight=None,
            selector_context=None,
        ):
            output = original_sean_forward(
                batch,
                states,
                hidden,
                mask,
                candidate_edge_index,
                candidate_edge_weight,
                selector_context,
            )
            if candidate_edge_index is None or candidate_edge_weight is None:
                return output
            node_entropy = (
                torch.ones_like(output.node_entropy)
                if sean_intervention == "bypass_graph_and_entropy"
                else output.node_entropy
            )
            return output._replace(
                edge_index=candidate_edge_index,
                edge_weight=candidate_edge_weight,
                node_entropy=node_entropy,
            )

        model.decoder.SEAN.forward = intervened_sean
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    trainer = Trainer(model, optimizer, scheduler, None, config, args)

    dataset = GroupTargetTrackingDatasetV3(data_dir, "val", load_structure=True)
    if max_trajectories:
        dataset = Subset(dataset, range(min(max_trajectories, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    stats = TensorStats()
    confusion = defaultdict(int)
    loss_parts = defaultdict(list)

    original_forward = model.forward

    def observed_forward(graph_data, *forward_args, **forward_kwargs):
        output = original_forward(graph_data, *forward_args, **forward_kwargs)
        predictable = output.real_mask[:, 0].bool()
        for name, tensor in (
            ("node_entropy", output.node_entropy[:, 0, 0]),
            ("gasf_confidence", output.gasf_confidence[:, 0, 0]),
            ("film_gate", output.film_gate[:, 0, 0]),
            ("scale_weight_8", output.spectral_weights[:, 0, 0]),
            ("scale_weight_16", output.spectral_weights[:, 0, 1]),
        ):
            stats.add(name, tensor[predictable])

        if output.adjacency.numel() and getattr(graph_data, "struct_feat", None):
            target = graph_data.struct_feat[0].to(output.adjacency).reshape(-1)
            probability = output.adjacency[:, 0]
            valid = output.adjacency_mask[:, 0].bool() & (target >= 0)
            if valid.any():
                probability = probability[valid]
                target = target[valid] > 0.5
                prediction = probability > 0.5
                stats.add("adjacency_probability", probability)
                stats.add("adjacency_positive_probability", probability[target])
                stats.add("adjacency_negative_probability", probability[~target])
                confusion["tp"] += int((prediction & target).sum())
                confusion["fp"] += int((prediction & ~target).sum())
                confusion["fn"] += int((~prediction & target).sum())
                confusion["tn"] += int((~prediction & ~target).sum())
        return output

    model.forward = observed_forward

    def gasf_pre_hook(_module, inputs):
        edge_weight = inputs[2]
        stats.add("gasf_graph_edge_weight", edge_weight)

    def gasf_hook(module, _inputs, output):
        stats.add("gasf_latent_norm", torch.linalg.vector_norm(output.latent, dim=-1))
        stats.add("gasf_raw_confidence", output.confidence)
        for diagnostic in module.last_filter_diagnostics:
            for name, value in diagnostic.items():
                stats.add(f"filter_{name}", value)

    def fusion_hook(_module, inputs, output):
        combined = inputs[0]
        width = combined.size(-1) // 2
        base, gasf = combined[:, :width], combined[:, width:]
        base_norm = torch.linalg.vector_norm(base, dim=-1)
        gasf_norm = torch.linalg.vector_norm(gasf, dim=-1)
        correction_norm = torch.linalg.vector_norm(output, dim=-1)
        stats.add("fusion_base_norm", base_norm)
        stats.add("fusion_gasf_norm", gasf_norm)
        stats.add("fusion_correction_norm", correction_norm)
        stats.add("fusion_correction_to_base", correction_norm / base_norm.clamp_min(1e-8))

    handles = [
        model.decoder.GASF.register_forward_pre_hook(gasf_pre_hook),
        model.decoder.GASF.register_forward_hook(gasf_hook),
        model.decoder.motion_model.gasf_concat_fusion.register_forward_hook(fusion_hook),
    ]

    if use_sean:
        def sean_hook(_module, inputs, output):
            valid = inputs[3].reshape(-1).bool()
            stats.add("sean_activation", output.activation[valid])
            stats.add("sean_edge_weight", output.edge_weight)
            stats.add("sean_adjacency", output.adjacency_vector[output.vector_mask])

        handles.append(model.decoder.SEAN.register_forward_hook(sean_hook))

    original_loss = trainer.compute_loss

    def observed_loss(*loss_args, **loss_kwargs):
        value = original_loss(*loss_args, **loss_kwargs)
        for name, tensor in trainer.last_loss_tensors.items():
            loss_parts[name].append(float(tensor.detach()))
        return value

    trainer.compute_loss = observed_loss
    metrics = trainer.validate(loader, "SEAN numerical diagnostic")
    for handle in handles:
        handle.remove()

    tp, fp, fn, tn = (confusion[name] for name in ("tp", "fp", "fn", "tn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    structure = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_rate": (tp + fn) / max(tp + fp + fn + tn, 1),
        "predicted_positive_rate": (tp + fp) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "specificity": tn / max(tn + fp, 1),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "sean_intervention": sean_intervention,
        "validation_metrics": metrics,
        "structure_confusion": structure,
        "loss_components_unweighted": {
            name: float(np.mean(values)) for name, values in loss_parts.items()
        },
        "tensor_statistics": stats.summarize(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True)
    parser.add_argument("--a2", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_trajectories", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()

    result = {
        "a1_no_sean_gasf": evaluate(
            cli.a1, cli.data_dir, False, cli.max_trajectories, cli.device
        ),
        "a2_sean_gasf": evaluate(
            cli.a2, cli.data_dir, True, cli.max_trajectories, cli.device
        ),
        "a2_bypass_sean_graph": evaluate(
            cli.a2,
            cli.data_dir,
            True,
            cli.max_trajectories,
            cli.device,
            sean_intervention="bypass_graph",
        ),
        "a2_bypass_sean_graph_and_entropy": evaluate(
            cli.a2,
            cli.data_dir,
            True,
            cli.max_trajectories,
            cli.device,
            sean_intervention="bypass_graph_and_entropy",
        ),
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
