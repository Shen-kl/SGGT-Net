"""Audit EKF NIS calibration and causal SEAN innovation decomposition."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
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
    load_compatible_state_dict,
)
from train.trainer_v3 import Trainer


EVENT_NAMES = {-1: "unclassified", 0: "steady", 1: "switch", 2: "strong"}
CHI2_DF2_QUANTILES = {
    "p50": (1.38629436112, 0.50),
    "p90": (4.60517018599, 0.90),
    "p95": (5.99146454711, 0.95),
    "p99": (9.21034037198, 0.99),
}


def node_scalar_or_default(values, node_count, name, empty_value=0.0):
    """Return one scalar per node, including disabled optional outputs.

    Optional SGGT branches use an empty tensor when they are disabled.  In
    particular, a Temporal checkpoint has no causal-innovation eligibility
    tensor, whereas a causal-corrector checkpoint returns ``[N, 1, 1]``.
    Calibration must treat the disabled branch as ineligible instead of
    indexing the empty one-dimensional tensor as if it were populated.
    """
    if values.numel() == 0:
        return values.new_full((int(node_count),), empty_value)
    flattened = values.reshape(-1)
    if flattened.numel() != int(node_count):
        raise ValueError(
            f"{name} must contain one scalar per node: "
            f"expected {node_count}, got shape {tuple(values.shape)}"
        )
    return flattened


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InnovationAccumulator:
    """Collect node-level diagnostics without retaining rollout tensors."""

    def __init__(self):
        self.values = defaultdict(lambda: defaultdict(list))
        self.reconstruction_max_error = 0.0

    def append(self, name, values, valid, event_class):
        values = values.detach().reshape(-1).to("cpu", torch.float64)
        valid = valid.detach().reshape(-1).to("cpu").bool()
        event_class = event_class.detach().reshape(-1).to("cpu")
        finite = torch.isfinite(values)
        valid = valid & finite
        selections = {"all": valid}
        selections.update({
            event_name: valid & (event_class == event_value)
            for event_value, event_name in EVENT_NAMES.items()
        })
        for event_name, selection in selections.items():
            if selection.any():
                self.values[event_name][name].extend(
                    values[selection].tolist()
                )

    def _observe_causal_window(self, window, current_valid, event_class):
        """Audit the actual own/leave-one-out-peer inputs through t-1.

        The current CGIC model keeps own and peer channels separate and does
        not execute the older GASF common/individual decomposition.  Its
        causal input tensors live on the rolling window, so collect them
        directly when the legacy GASF diagnostic dictionary is absent.
        """
        required = (
            "innovation_raw",
            "innovation_whitened",
            "innovation_peer_raw",
            "innovation_peer_whitened",
            "nis_history",
            "innovation_valid",
            "innovation_peer_valid",
        )
        if window is None or not all(hasattr(window, name) for name in required):
            return
        own_valid = window.innovation_valid[:, -1].reshape(-1).bool()
        peer_valid = window.innovation_peer_valid[:, -1].reshape(-1).bool()
        history_valid = current_valid & own_valid
        paired_valid = history_valid & peer_valid
        own_raw = window.innovation_raw[:, -1]
        peer_raw = window.innovation_peer_raw[:, -1]
        own_norm = torch.linalg.vector_norm(own_raw, dim=-1)
        peer_norm = torch.linalg.vector_norm(peer_raw, dim=-1)
        cosine = (own_raw * peer_raw).sum(dim=-1) / (
            own_norm * peer_norm
        ).clamp_min(1e-6)
        peer_nis = window.innovation_peer_whitened[:, -1].square().sum(dim=-1)

        self.append(
            "history_has_neighbor",
            peer_valid.to(own_raw.dtype),
            history_valid,
            event_class,
        )
        self.append(
            "history_own_nis",
            window.nis_history[:, -1],
            history_valid,
            event_class,
        )
        self.append(
            "history_peer_nis",
            peer_nis,
            paired_valid,
            event_class,
        )
        self.append(
            "history_own_innovation_norm_m",
            own_norm,
            history_valid,
            event_class,
        )
        self.append(
            "history_peer_innovation_norm_m",
            peer_norm,
            paired_valid,
            event_class,
        )
        self.append(
            "history_peer_to_own_norm_ratio",
            peer_norm / own_norm.clamp_min(1e-6),
            paired_valid,
            event_class,
        )
        self.append(
            "history_own_peer_cosine",
            cosine,
            paired_valid,
            event_class,
        )

    def observe(self, model, outputs, event_class, window=None):
        current_valid = outputs.real_mask[:, 0].bool()
        innovation = outputs.measurement_innovation[:, 0]
        covariance = outputs.innovation_covariance
        eigenvalues = torch.linalg.eigvalsh(
            0.5 * (covariance + covariance.transpose(-1, -2))
        ).clamp_min(1e-12)
        self.append("current_nis", outputs.nis[:, 0, 0], current_valid, event_class)
        self.append(
            "current_innovation_norm_m",
            torch.linalg.vector_norm(innovation, dim=-1),
            current_valid,
            event_class,
        )
        self.append(
            "current_s_trace_m2",
            covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1),
            current_valid,
            event_class,
        )
        self.append(
            "current_s_logdet",
            torch.log(eigenvalues).sum(dim=-1),
            current_valid,
            event_class,
        )
        self.append(
            "current_s_condition",
            eigenvalues[:, -1] / eigenvalues[:, 0],
            current_valid,
            event_class,
        )
        association_valid = outputs.association_valid[:, 0, 0].bool()
        association_mask = current_valid & association_valid
        self.append(
            "association_detection_mass",
            outputs.association_detection_mass[:, 0, 0],
            current_valid,
            event_class,
        )
        self.append(
            "association_entropy",
            outputs.association_entropy[:, 0, 0],
            association_mask,
            event_class,
        )
        self.append(
            "association_spread_trace_m2",
            outputs.association_spread[:, 0].sum(dim=-1),
            association_mask,
            event_class,
        )
        self.append(
            "causal_history_eligible",
            node_scalar_or_default(
                outputs.causal_innovation_eligible,
                current_valid.numel(),
                "causal_innovation_eligible",
                empty_value=False,
            ),
            current_valid,
            event_class,
        )

        details = getattr(model.decoder.GASF, "last_innovation_details", {})
        if not details:
            self._observe_causal_window(window, current_valid, event_class)
            return
        history_valid = details["valid_latest"].bool() & current_valid
        has_neighbors = details["has_neighbors_latest"].bool() & history_valid
        own_raw = details["own_raw_latest"]
        group_raw = details["group_raw_latest"]
        individual_raw = details["individual_raw_latest"]
        own_norm = torch.linalg.vector_norm(own_raw, dim=-1)
        group_norm = torch.linalg.vector_norm(group_raw, dim=-1)
        individual_norm = torch.linalg.vector_norm(individual_raw, dim=-1)
        energy_denominator = (
            group_norm.square() + individual_norm.square()
        ).clamp_min(1e-12)

        self.append(
            "history_has_neighbor",
            has_neighbors.to(own_raw.dtype),
            history_valid,
            event_class,
        )
        self.append(
            "history_own_nis",
            details["own_nis_latest"],
            history_valid,
            event_class,
        )
        self.append(
            "history_group_nis",
            details["group_nis_latest"],
            has_neighbors,
            event_class,
        )
        self.append(
            "history_individual_nis",
            details["individual_nis_latest"],
            history_valid,
            event_class,
        )
        self.append(
            "history_common_energy_fraction",
            group_norm.square() / energy_denominator,
            has_neighbors,
            event_class,
        )
        self.append(
            "history_individual_to_own_norm_ratio",
            individual_norm / own_norm.clamp_min(1e-6),
            history_valid,
            event_class,
        )
        reconstruction = torch.linalg.vector_norm(
            group_raw + individual_raw - own_raw, dim=-1
        )
        self.append(
            "history_reconstruction_error_m",
            reconstruction,
            history_valid,
            event_class,
        )
        if history_valid.any():
            self.reconstruction_max_error = max(
                self.reconstruction_max_error,
                float(reconstruction[history_valid].max()),
            )

        edge_index = details["edge_index"]
        if edge_index.numel() == 0:
            return
        source, target = edge_index
        edge_valid = history_valid[source] & history_valid[target]
        if not edge_valid.any():
            return
        group_difference = torch.linalg.vector_norm(
            group_raw[source] - group_raw[target], dim=-1
        )
        own_difference = torch.linalg.vector_norm(
            own_raw[source] - own_raw[target], dim=-1
        )
        edge_ratio = group_difference / own_difference.clamp_min(1e-6)
        edge_weight = details["edge_weight"].to(own_raw.dtype)
        node_count = own_raw.size(0)
        incoming_weight = own_raw.new_zeros(node_count)
        incoming_weight_square = own_raw.new_zeros(node_count)
        incoming_count = own_raw.new_zeros(node_count)
        valid_weight = edge_weight * edge_valid.to(edge_weight.dtype)
        incoming_weight.index_add_(0, target, valid_weight)
        incoming_weight_square.index_add_(0, target, valid_weight.square())
        incoming_count.index_add_(0, target, edge_valid.to(own_raw.dtype))
        effective_neighbor_count = incoming_weight.square() / (
            incoming_weight_square + 1e-12
        )
        self.append(
            "history_sean_incoming_weight",
            incoming_weight,
            has_neighbors,
            event_class,
        )
        self.append(
            "history_sean_neighbor_count",
            incoming_count,
            has_neighbors,
            event_class,
        )
        self.append(
            "history_sean_effective_neighbor_count",
            effective_neighbor_count,
            has_neighbors,
            event_class,
        )
        self.append(
            "history_sean_edge_weight",
            edge_weight,
            edge_valid,
            event_class[target],
        )
        # Attribute a directed edge diagnostic to its receiving target so it
        # can use the same event stratification as node-level statistics.
        self.append(
            "history_edge_common_to_own_disagreement_ratio",
            edge_ratio,
            edge_valid,
            event_class[target],
        )

    @staticmethod
    def summarize_values(values):
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {"count": 0}
        return {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "median": float(np.median(array)),
            "p05": float(np.percentile(array, 5)),
            "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    @staticmethod
    def nis_calibration(values):
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {"count": 0}
        array = np.maximum(array, 0.0)
        pit = 1.0 - np.exp(-0.5 * array)
        sorted_pit = np.sort(pit)
        count = array.size
        empirical = np.arange(1, count + 1, dtype=np.float64) / count
        empirical_left = np.arange(0, count, dtype=np.float64) / count
        ks = max(
            float(np.max(np.abs(empirical - sorted_pit))),
            float(np.max(np.abs(sorted_pit - empirical_left))),
        )
        coverage = {}
        calibration_error = []
        for name, (threshold, nominal) in CHI2_DF2_QUANTILES.items():
            observed = float(np.mean(array <= threshold))
            coverage[name] = {
                "threshold": threshold,
                "nominal_below": nominal,
                "observed_below": observed,
                "error": observed - nominal,
            }
            calibration_error.append(abs(observed - nominal))
        return {
            "count": int(count),
            "expected_mean": 2.0,
            "observed_mean": float(np.mean(array)),
            "mean_ratio": float(np.mean(array) / 2.0),
            "observed_median": float(np.median(array)),
            "pit_mean": float(np.mean(pit)),
            "pit_ks_uniform": ks,
            "coverage_mae": float(np.mean(calibration_error)),
            "coverage": coverage,
        }

    def result(self):
        result = {}
        for event_name, event_values in self.values.items():
            result[event_name] = {
                "statistics": {
                    name: self.summarize_values(values)
                    for name, values in event_values.items()
                },
                "current_nis_calibration": self.nis_calibration(
                    event_values.get("current_nis", [])
                ),
                "causal_history_nis_calibration": self.nis_calibration(
                    event_values.get("history_own_nis", [])
                ),
            }
        return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=["val", "iid_test", "ood_test"])
    parser.add_argument("--start_trajectory", type=int, default=0)
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["temporal", "expert", "checkpoint"], default="expert")
    parser.add_argument("--association_augmentation", action="store_true")
    parser.add_argument("--association_seed", type=int, default=20260813)
    return parser.parse_args()


def main():
    cli = parse_args()
    device = torch.device(cli.device)
    checkpoint_path = Path(cli.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = device
    args.decoder_use_GASF = bool(payload.get("decoder_use_GASF", True))
    args.decoder_use_SEAN = bool(payload.get("decoder_use_SEAN", True))
    args.model_variant = payload.get("model_variant", "sggt_v2")
    args.covariance_transition = payload.get("covariance_transition", "cv")
    args.gasf_filter_mode = "abc"
    args.gasf_graph_mode = payload.get("gasf_graph_mode", "graph")
    args.gasf_fusion_mode = payload.get("gasf_fusion_mode", "concat")
    args.gasf_aux_source = "fusion"
    args.counterfactual_training = False
    args.innovation_decomposition = bool(
        payload.get("innovation_decomposition", False)
    )
    args.innovation_whiten_clip = float(
        payload.get("innovation_whiten_clip", 20.0)
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
    args.association_max_clutter = int(
        payload.get("association_max_clutter", 32)
    )
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
    args.association_validation_seed = int(cli.association_seed)
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
    override = {"temporal": 0.0, "expert": 1.0, "checkpoint": None}[cli.mode]
    model.decoder.GASF.interaction_activation_override = override

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
    selected_indices = list(range(start, stop))
    loader = DataLoader(
        Subset(dataset, selected_indices), batch_size=1, shuffle=False,
        num_workers=0,
    )
    accumulator = InnovationAccumulator()

    def observer(**context):
        accumulator.observe(
            model,
            context["outputs"],
            context["event_class"],
            window=context["window"],
        )

    trainer.step_observer = observer
    if cli.association_augmentation:
        global_metrics = trainer.validate_association(
            loader, "Association innovation calibration"
        )
    else:
        global_metrics = trainer.validate(loader, "Innovation calibration")
    trainer.step_observer = None
    report = {
        "protocol": {
            "version": 2,
            "evaluator_sha256": file_sha256(Path(__file__).resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch", -999)),
            "dataset_root": str(Path(cli.data_dir).resolve()),
            "split": cli.split,
            "trajectory_start": start,
            "trajectory_stop_exclusive": stop,
            "trajectory_count": stop - start,
            "mode": cli.mode,
            "interaction_activation_override": override,
            "online_rollout": True,
            "association_augmentation": bool(cli.association_augmentation),
            "association_seed": (
                int(cli.association_seed) if cli.association_augmentation else None
            ),
            "current_nis_time": "measurement at scored step t",
            "decomposition_time": "latest causal innovation through t-1",
            "history_diagnostic": (
                "legacy GASF decomposition when active; otherwise separate "
                "own and leave-one-out-peer rolling-window channels"
            ),
            "state_dimension": 4,
            "measurement_dimension": 2,
            "expected_nis_distribution": "chi-square(df=2)",
        },
        "global_tracking_metrics": {
            key: float(value) for key, value in global_metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        "reconstruction_max_error_m": accumulator.reconstruction_max_error,
        "events": accumulator.result(),
    }
    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
