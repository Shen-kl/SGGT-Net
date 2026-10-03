"""Paired and closed-loop SEAN versus temporal-only GASF evaluation."""

import argparse
from contextlib import nullcontext
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from model.sggt_net import SGGT_Net
from train.trainer_v3 import Trainer


EVENT_NAMES = {-1: "unclassified", 0: "steady", 1: "switch", 2: "strong"}


def build_model(args):
    motion_model = SecondOrderNeuralODE_groupTrack(
        solver=args.ode_solver,
        dt=args.T,
        mixtures=args.n_mixtures,
        static_f_dim=0,
        n_hidden=args.n_ode_hidden,
        n_layers=args.n_ode_layers,
        use_SEAN=args.decoder_use_SEAN,
        use_GASF=args.decoder_use_GASF,
        gasf_fusion_mode=args.gasf_fusion_mode,
        gasf_aux_source=args.gasf_aux_source,
        interaction_physical_residual=args.interaction_physical_residual,
        interaction_physical_max_gain=args.interaction_physical_max_gain,
        interaction_physical_init_gain=args.interaction_physical_init_gain,
        interaction_activation_floor=args.interaction_activation_floor,
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
        args.decoder_use_GASF,
        args.decoder_use_SEAN,
        args.T,
        model_variant="sggt_v2",
        covariance_transition=args.covariance_transition,
        gasf_filter_mode=args.gasf_filter_mode,
        gasf_graph_mode=args.gasf_graph_mode,
        gasf_recurrent_filter=getattr(args, "gasf_recurrent_filter", False),
        gasf_recurrent_gate_initial_update=getattr(
            args, "gasf_recurrent_gate_initial_update", 0.1
        ),
        counterfactual_training=getattr(
            args, 'counterfactual_training', True
        ),
        adaptive_q_mode=args.adaptive_q_mode,
        adaptive_q_max_multiplier=args.adaptive_q_max_multiplier,
        adaptive_q_max_position_variance=args.adaptive_q_max_position_variance,
        adaptive_q_initial_multiplier=getattr(
            args, 'adaptive_q_initial_multiplier', 1.0
        ),
        adaptive_q_initial_position_variance=getattr(
            args, 'adaptive_q_initial_position_variance', 0.01
        ),
        adaptive_q_acceleration=args.adaptive_q_acceleration,
        adaptive_q_tail_boost=args.adaptive_q_tail_boost,
        adaptive_q_tail_threshold=args.adaptive_q_tail_threshold,
        innovation_decomposition=getattr(args, 'innovation_decomposition', False),
        innovation_whiten_clip=getattr(args, 'innovation_whiten_clip', 20.0),
        causal_innovation_correction=getattr(
            args, 'causal_innovation_correction', False
        ),
        causal_innovation_history=getattr(args, 'causal_innovation_history', 8),
        causal_innovation_hidden_size=getattr(
            args, 'causal_innovation_hidden_size', 32
        ),
        causal_innovation_output_mode=getattr(
            args, 'causal_innovation_output_mode', 'acceleration'
        ),
        causal_innovation_max_acceleration_mps2=getattr(
            args, 'causal_innovation_max_acceleration_mps2', 5.0
        ),
        causal_innovation_max_position_m=getattr(
            args, 'causal_innovation_max_position_m', 50.0
        ),
        causal_innovation_max_velocity_mps=getattr(
            args, 'causal_innovation_max_velocity_mps', 5.0
        ),
        causal_innovation_dynamic_activation=getattr(
            args, 'causal_innovation_dynamic_activation', False
        ),
        causal_innovation_activation_floor=getattr(
            args, 'causal_innovation_activation_floor', 0.5
        ),
        causal_innovation_activation_nis_low=getattr(
            args, 'causal_innovation_activation_nis_low', 1.38629436112
        ),
        causal_innovation_activation_nis_high=getattr(
            args, 'causal_innovation_activation_nis_high', 5.99146454711
        ),
        causal_innovation_minimum_valid_history=(
            getattr(args, 'causal_innovation_minimum_valid_history', 0) or None
        ),
        causal_innovation_recent_valid_window=getattr(
            args, 'causal_innovation_recent_valid_window', 3
        ),
        causal_innovation_association_context=getattr(
            args, 'causal_innovation_association_context', False
        ),
        causal_innovation_use_peer=getattr(
            args, 'causal_innovation_use_peer', True
        ),
        association_augmentation=getattr(args, 'association_augmentation', False),
        association_pd_min=getattr(args, 'association_pd_min', 0.75),
        association_pd_max=getattr(args, 'association_pd_max', 0.98),
        association_clutter_mu_min=getattr(args, 'association_clutter_mu_min', 0.0),
        association_clutter_mu_max=getattr(args, 'association_clutter_mu_max', 0.0),
        association_surveillance_side_m=getattr(
            args, 'association_surveillance_side_m', 3000.0
        ),
        association_max_clutter=getattr(args, 'association_max_clutter', 32),
        association_local_clutter_rate=getattr(
            args, 'association_local_clutter_rate', 0.25
        ),
        association_max_local_clutter=getattr(
            args, 'association_max_local_clutter', 2
        ),
        association_local_clutter_scale=getattr(
            args, 'association_local_clutter_scale', 2.5
        ),
        association_gate_threshold=getattr(
            args, 'association_gate_threshold', 9.21034037198
        ),
        association_minimum_detection_mass=getattr(
            args, 'association_minimum_detection_mass', 0.05
        ),
        association_include_target_competitors=getattr(
            args, 'association_include_target_competitors', False
        ),
        association_clutter_requires_detection=getattr(
            args, 'association_clutter_requires_detection', True
        ),
    )


def load_compatible_state_dict(model, state_dict):
    """Load pre-context checkpoints without weakening all compatibility checks."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "decoder.GASF.interaction_innovation_projection.",
        "decoder.motion_model.gasf_interaction_context_embedding.",
        "decoder.motion_model.interaction_physical_gain_raw",
        "decoder.GASF.context_activation_head.",
        "decoder.GASF.group_innovation_encoders.",
        "decoder.GASF.individual_innovation_encoders.",
        "decoder.GASF.innovation_uncertainty_encoder.",
        "decoder.GASF.innovation_decomposition_projection.",
        "decoder.GASF.parameter_update_gates.",
        "decoder.motion_model.causal_innovation_corrector.",
    )
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint incompatibility: "
            f"missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return incompatible


class EventAccumulator:
    def __init__(self, config):
        self.config = config
        self.values = defaultdict(lambda: defaultdict(float))

    def _masks(self, valid, event_class):
        yield "all", valid
        for value, name in EVENT_NAMES.items():
            mask = valid & (event_class == value)
            if mask.any():
                yield name, mask

    @staticmethod
    def _state_sse(state, target, scale, mask):
        error = (state - target) * scale
        return (
            float(error[mask, :2].square().sum()),
            float(error[mask, 2:].square().sum()),
        )

    @staticmethod
    def _graph_signal(signal, edge_index, edge_weight):
        if edge_index.numel() == 0:
            return torch.zeros_like(signal)
        source, target = edge_index
        weight = edge_weight.to(signal).reshape(-1)
        degree = torch.zeros(
            signal.size(0), device=signal.device, dtype=signal.dtype
        )
        degree.index_add_(0, target, weight)
        norm = (
            weight
            * degree[source].clamp_min(1e-6).rsqrt()
            * degree[target].clamp_min(1e-6).rsqrt()
        )
        output = torch.zeros_like(signal)
        output.index_add_(0, target, signal[source] * norm.unsqueeze(-1))
        return output

    @staticmethod
    def _add_signal_alignment(bucket, name, signal, ideal):
        signal_norm = torch.linalg.vector_norm(signal, dim=-1)
        ideal_norm = torch.linalg.vector_norm(ideal, dim=-1)
        usable = (signal_norm > 1e-8) & (ideal_norm > 1e-8)
        if not usable.any():
            return
        signal = signal[usable]
        ideal = ideal[usable]
        signal_norm = signal_norm[usable]
        ideal_norm = ideal_norm[usable]
        dot = (signal * ideal).sum(dim=-1)
        cosine = dot / (signal_norm * ideal_norm).clamp_min(1e-8)
        prefix = f"signal_{name}_"
        bucket[prefix + "count"] += int(usable.sum())
        bucket[prefix + "dot_sum"] += float(dot.sum())
        bucket[prefix + "signal_square_sum"] += float(signal.square().sum())
        bucket[prefix + "ideal_square_sum"] += float(ideal.square().sum())
        bucket[prefix + "cosine_sum"] += float(cosine.sum())
        bucket[prefix + "positive_cosine_count"] += float((cosine > 0).sum())

    def add(self, outputs, event_class, gasf_output, window=None):
        valid = outputs.real_mask[:, 0].bool()
        if not valid.any():
            return
        scale = outputs.states.new_tensor([
            self.config["S_POS"], self.config["S_POS"],
            self.config["S_VEL"], self.config["S_VEL"],
        ])
        target = outputs.target[:, 0]
        prediction = outputs.states[:, 0, 0]
        update = outputs.updated_state[:, 0]
        causal_activation = outputs.causal_innovation_activation.numel() > 0
        activation = (
            outputs.causal_innovation_activation[:, 0, 0]
            if causal_activation else outputs.gasf_confidence[:, 0, 0]
        )
        activation_valid = (
            valid & outputs.causal_innovation_eligible[:, 0, 0].bool()
            if causal_activation else valid
        )

        if gasf_output is None:
            # GASF-free baselines still use this evaluator for the common
            # trajectory and maneuver-event metrics.  Their spectral/graph
            # diagnostics are structurally absent and therefore contribute
            # zeros instead of making the evaluator dereference ``None``.
            activation = torch.zeros_like(activation)
            delta_norm = torch.zeros_like(activation)
            effective_norm = torch.zeros_like(activation)
            effective_ratio = torch.zeros_like(activation)
        else:
            temporal_latent = gasf_output.temporal_latent
            interaction_delta = gasf_output.interaction_delta
            temporal_norm = torch.linalg.vector_norm(temporal_latent, dim=-1)
            delta_norm = torch.linalg.vector_norm(interaction_delta, dim=-1)
            effective_norm = activation * delta_norm
            effective_ratio = effective_norm / temporal_norm.clamp_min(1e-8)

        has_filter_transition = (
            gasf_output is not None
            and gasf_output.recurrent_transition_valid.numel() > 0
        )
        if has_filter_transition:
            filter_transition_valid = (
                gasf_output.recurrent_transition_valid.bool().all(dim=-1)
            )
            filter_gate = gasf_output.parameter_update_gates.mean(dim=-1)
            filter_parameter_step = gasf_output.parameter_step_norms.mean(dim=-1)
            filter_response_step = gasf_output.filter_response_step_rms.mean(dim=-1)

        has_q_diagnostics = (
            outputs.q_acceleration_multiplier.numel() > 0
            and outputs.q_innovation_score.numel() > 0
            and outputs.q_position_noise.numel() > 0
        )
        if has_q_diagnostics:
            q_acceleration = outputs.q_acceleration_multiplier[:, 0, 0]
            q_surprise = outputs.q_innovation_score[:, 0, 0]
            q_position = (
                outputs.q_position_noise[:, 0, 0, [0, 1], [0, 1]].sum(-1)
                * float(self.config["S_POS"]) ** 2
                / 2.0
            )

        has_counterfactual = (
            outputs.counterfactual_base.numel() > 0
            and outputs.counterfactual_expert.numel() > 0
        )
        if has_counterfactual:
            temporal_state = outputs.counterfactual_base[:, 0, 0]
            expert_state = outputs.counterfactual_expert[:, 0, 0]

        causal_signals = None
        if window is not None and has_counterfactual and gasf_output is not None:
            residual = window.residual.to(target)
            latest = residual[:, -1]
            trend = residual[:, -1] - residual[:, -2]
            structure = gasf_output.structure
            graph_latest = self._graph_signal(
                latest, structure.edge_index, structure.edge_weight
            )
            graph_trend = self._graph_signal(
                trend, structure.edge_index, structure.edge_weight
            )
            causal_signals = {
                "own_latest": latest,
                "own_trend": trend,
                "graph_latest": graph_latest,
                "graph_trend": graph_trend,
                "graph_latest_contrast": graph_latest - latest,
                "graph_trend_contrast": graph_trend - trend,
            }

        for name, mask in self._masks(valid, event_class):
            count = int(mask.sum())
            bucket = self.values[name]
            bucket["count"] += count
            pos_sse, vel_sse = self._state_sse(
                prediction, target, scale, mask
            )
            update_pos_sse, update_vel_sse = self._state_sse(
                update, target, scale, mask
            )
            bucket["prediction_position_sse"] += pos_sse
            bucket["prediction_velocity_sse"] += vel_sse
            bucket["update_position_sse"] += update_pos_sse
            bucket["update_velocity_sse"] += update_vel_sse
            activation_mask = mask & activation_valid
            bucket["activation_count"] += int(activation_mask.sum())
            bucket["activation_sum"] += float(activation[activation_mask].sum())
            bucket["activation_square_sum"] += float(
                activation[activation_mask].square().sum()
            )
            bucket["interaction_delta_norm_sum"] += float(delta_norm[mask].sum())
            bucket["effective_interaction_norm_sum"] += float(
                effective_norm[mask].sum()
            )
            bucket["effective_interaction_ratio_sum"] += float(
                effective_ratio[mask].sum()
            )

            if has_filter_transition:
                filter_mask = mask & filter_transition_valid
                if filter_mask.any():
                    bucket["filter_transition_count"] += int(filter_mask.sum())
                    for prefix, values in (
                        ("filter_gate", filter_gate),
                        ("filter_parameter_step", filter_parameter_step),
                        ("filter_response_step", filter_response_step),
                    ):
                        selected = values[filter_mask].double()
                        bucket[prefix + "_sum"] += float(selected.sum())
                        bucket[prefix + "_square_sum"] += float(
                            selected.square().sum()
                        )

            if has_q_diagnostics:
                surprise_values = q_surprise[mask].double()
                acceleration_values = q_acceleration[mask].double()
                position_values = q_position[mask].double()
                finite = (
                    torch.isfinite(surprise_values)
                    & torch.isfinite(acceleration_values)
                    & torch.isfinite(position_values)
                )
                if finite.any():
                    surprise_values = surprise_values[finite]
                    acceleration_values = acceleration_values[finite]
                    position_values = position_values[finite]
                    bucket["q_count"] += int(finite.sum())
                    bucket["q_surprise_sum"] += float(surprise_values.sum())
                    bucket["q_surprise_square_sum"] += float(
                        surprise_values.square().sum()
                    )
                    for prefix, values in (
                        ("q_acceleration", acceleration_values),
                        ("q_position", position_values),
                    ):
                        bucket[prefix + "_sum"] += float(values.sum())
                        bucket[prefix + "_square_sum"] += float(
                            values.square().sum()
                        )
                        bucket[prefix + "_surprise_product_sum"] += float(
                            (values * surprise_values).sum()
                        )

            if has_counterfactual:
                base_pos_sse, base_vel_sse = self._state_sse(
                    temporal_state, target, scale, mask
                )
                expert_pos_sse, expert_vel_sse = self._state_sse(
                    expert_state, target, scale, mask
                )
                bucket["cf_temporal_position_sse"] += base_pos_sse
                bucket["cf_temporal_velocity_sse"] += base_vel_sse
                bucket["cf_expert_position_sse"] += expert_pos_sse
                bucket["cf_expert_velocity_sse"] += expert_vel_sse
                base_node_error = (
                    (temporal_state[mask, :2] - target[mask, :2])
                    * scale[:2]
                ).square().sum(dim=-1)
                expert_node_error = (
                    (expert_state[mask, :2] - target[mask, :2])
                    * scale[:2]
                ).square().sum(dim=-1)
                bucket["cf_expert_position_wins"] += float(
                    (expert_node_error < base_node_error).sum()
                )
                correction = (expert_state[mask] - temporal_state[mask]) * scale
                ideal = (target[mask] - temporal_state[mask]) * scale
                for suffix, coordinate_slice in (
                    ("position", slice(0, 2)),
                    ("velocity", slice(2, 4)),
                ):
                    correction_part = correction[:, coordinate_slice]
                    ideal_part = ideal[:, coordinate_slice]
                    correction_norm = torch.linalg.vector_norm(
                        correction_part, dim=-1
                    )
                    ideal_norm = torch.linalg.vector_norm(ideal_part, dim=-1)
                    direction_valid = ideal_norm > 1e-8
                    if direction_valid.any():
                        dot = (
                            correction_part[direction_valid]
                            * ideal_part[direction_valid]
                        ).sum(dim=-1)
                        cosine = dot / (
                            correction_norm[direction_valid]
                            * ideal_norm[direction_valid]
                        ).clamp_min(1e-8)
                        projection = dot / ideal_norm[direction_valid].square().clamp_min(1e-8)
                        bucket[f"cf_{suffix}_direction_count"] += int(
                            direction_valid.sum()
                        )
                        bucket[f"cf_{suffix}_cosine_sum"] += float(cosine.sum())
                        bucket[f"cf_{suffix}_ideal_projection_sum"] += float(
                            projection.sum()
                        )
                    bucket[f"cf_{suffix}_correction_norm_sum"] += float(
                        correction_norm.sum()
                    )
                    bucket[f"cf_{suffix}_ideal_norm_sum"] += float(
                        ideal_norm.sum()
                    )
                if causal_signals is not None:
                    ideal_position = ideal[:, :2]
                    for signal_name, signal in causal_signals.items():
                        self._add_signal_alignment(
                            bucket,
                            signal_name,
                            signal[mask],
                            ideal_position,
                        )

    def result(self):
        result = {}
        for name, bucket in self.values.items():
            count = max(bucket["count"], 1.0)
            activation_count = max(bucket["activation_count"], 1.0)
            activation_mean = bucket["activation_sum"] / activation_count
            activation_variance = max(
                bucket["activation_square_sum"] / activation_count
                - activation_mean ** 2,
                0.0,
            )
            row = {
                "count": int(bucket["count"]),
                "position_prediction_rmse": math.sqrt(
                    bucket["prediction_position_sse"] / count
                ),
                "velocity_prediction_rmse": math.sqrt(
                    bucket["prediction_velocity_sse"] / count
                ),
                "position_update_rmse": math.sqrt(
                    bucket["update_position_sse"] / count
                ),
                "velocity_update_rmse": math.sqrt(
                    bucket["update_velocity_sse"] / count
                ),
                "activation_mean": activation_mean,
                "activation_std": math.sqrt(activation_variance),
                "activation_count": int(bucket["activation_count"]),
                "interaction_delta_norm_mean": (
                    bucket["interaction_delta_norm_sum"] / count
                ),
                "effective_interaction_norm_mean": (
                    bucket["effective_interaction_norm_sum"] / count
                ),
                "effective_to_temporal_norm_ratio": (
                    bucket["effective_interaction_ratio_sum"] / count
                ),
            }
            if bucket["filter_transition_count"] > 0:
                transition_count = float(bucket["filter_transition_count"])
                row["filter_transition_count"] = int(transition_count)
                for prefix, output_name in (
                    ("filter_gate", "gasf_parameter_update_gate"),
                    ("filter_parameter_step", "gasf_parameter_step_norm"),
                    ("filter_response_step", "gasf_filter_response_step_rms"),
                ):
                    mean = bucket[prefix + "_sum"] / transition_count
                    mean_square = (
                        bucket[prefix + "_square_sum"] / transition_count
                    )
                    row[output_name + "_mean"] = mean
                    row[output_name + "_rms"] = math.sqrt(mean_square)
                    row[output_name + "_std"] = math.sqrt(
                        max(mean_square - mean ** 2, 0.0)
                    )
            if bucket["q_count"] > 0:
                q_count = float(bucket["q_count"])
                surprise_mean = bucket["q_surprise_sum"] / q_count
                surprise_var = max(
                    bucket["q_surprise_square_sum"] / q_count
                    - surprise_mean ** 2,
                    0.0,
                )
                row["q_surprise_mean"] = surprise_mean
                row["q_count"] = int(bucket["q_count"])
                for prefix, output_name in (
                    ("q_acceleration", "q_acceleration_multiplier"),
                    ("q_position", "q_position_variance_m2"),
                ):
                    value_mean = bucket[prefix + "_sum"] / q_count
                    value_var = max(
                        bucket[prefix + "_square_sum"] / q_count
                        - value_mean ** 2,
                        0.0,
                    )
                    covariance = (
                        bucket[prefix + "_surprise_product_sum"] / q_count
                        - value_mean * surprise_mean
                    )
                    denominator = math.sqrt(value_var * surprise_var)
                    row[output_name + "_mean"] = value_mean
                    row[output_name + "_std"] = math.sqrt(value_var)
                    row[output_name + "_surprise_correlation"] = (
                        covariance / denominator if denominator > 1e-12 else 0.0
                    )
            if bucket["cf_temporal_position_sse"] > 0:
                temporal_position = math.sqrt(
                    bucket["cf_temporal_position_sse"] / count
                )
                expert_position = math.sqrt(
                    bucket["cf_expert_position_sse"] / count
                )
                temporal_velocity = math.sqrt(
                    bucket["cf_temporal_velocity_sse"] / count
                )
                expert_velocity = math.sqrt(
                    bucket["cf_expert_velocity_sse"] / count
                )
                row.update({
                    "paired_temporal_position_rmse": temporal_position,
                    "paired_expert_position_rmse": expert_position,
                    "paired_position_improvement_percent": 100.0 * (
                        temporal_position - expert_position
                    ) / max(temporal_position, 1e-12),
                    "paired_temporal_velocity_rmse": temporal_velocity,
                    "paired_expert_velocity_rmse": expert_velocity,
                    "paired_velocity_improvement_percent": 100.0 * (
                        temporal_velocity - expert_velocity
                    ) / max(temporal_velocity, 1e-12),
                    "paired_expert_position_win_rate": (
                        bucket["cf_expert_position_wins"] / count
                    ),
                })
                for suffix in ("position", "velocity"):
                    direction_count = max(
                        bucket[f"cf_{suffix}_direction_count"], 1.0
                    )
                    row[f"paired_{suffix}_correction_norm_mean"] = (
                        bucket[f"cf_{suffix}_correction_norm_sum"] / count
                    )
                    row[f"paired_{suffix}_ideal_norm_mean"] = (
                        bucket[f"cf_{suffix}_ideal_norm_sum"] / count
                    )
                    row[f"paired_{suffix}_correction_cosine_mean"] = (
                        bucket[f"cf_{suffix}_cosine_sum"] / direction_count
                    )
                    row[f"paired_{suffix}_ideal_projection_mean"] = (
                        bucket[f"cf_{suffix}_ideal_projection_sum"]
                        / direction_count
                    )
                for signal_name in (
                    "own_latest",
                    "own_trend",
                    "graph_latest",
                    "graph_trend",
                    "graph_latest_contrast",
                    "graph_trend_contrast",
                ):
                    prefix = f"signal_{signal_name}_"
                    signal_count = bucket[prefix + "count"]
                    if signal_count <= 0:
                        continue
                    dot_sum = bucket[prefix + "dot_sum"]
                    signal_square_sum = bucket[prefix + "signal_square_sum"]
                    ideal_square_sum = bucket[prefix + "ideal_square_sum"]
                    optimal_beta = dot_sum / max(signal_square_sum, 1e-12)
                    explained = dot_sum * dot_sum / max(
                        signal_square_sum * ideal_square_sum, 1e-12
                    )
                    row[f"{prefix}cosine_mean"] = (
                        bucket[prefix + "cosine_sum"] / signal_count
                    )
                    row[f"{prefix}positive_cosine_fraction"] = (
                        bucket[prefix + "positive_cosine_count"] / signal_count
                    )
                    row[f"{prefix}optimal_scalar"] = optimal_beta
                    row[f"{prefix}linear_explained_fraction"] = explained
            result[name] = row
        return result


class TrajectorySSEAccumulator:
    """Accumulate bootstrap sufficient statistics without forcing batch=1."""

    METRICS = (
        "position_prediction",
        "position_update",
        "velocity_prediction",
        "velocity_update",
    )

    def __init__(self, config):
        self.config = config
        self.values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

    def add(self, graph_data, window, outputs, centroids, event_class):
        valid = outputs.real_mask[:, 0].bool()
        if not valid.any():
            return
        graph_batch = graph_data.batch
        if hasattr(graph_data, "trajectory_id"):
            trajectory_ids = graph_data.trajectory_id.reshape(-1)
        else:
            # Safe only for legacy batch=1 data.  New V3 data always provides
            # trajectory_id because PyG offsets names containing ``index``.
            if int(graph_data.num_graphs) != 1:
                raise ValueError(
                    "batched trajectory SSE requires graph_data.trajectory_id"
                )
            trajectory_ids = graph_data.trajectory_index.reshape(-1)
        # ``outputs.target`` is normalized by minmax_scaler; the formal RMSE
        # evaluator compares denormalized states against the raw online-window
        # target.  Use that same raw target so accumulated SSE exactly
        # reproduces the global metric.
        target = torch.nan_to_num(window.y[:, 0])
        weights = torch.softmax(outputs.mixture_coeffs, dim=-1)
        prediction = (
            outputs.states[:, 0] * weights.unsqueeze(-1)
        ).sum(dim=1)
        update = outputs.updated_state[:, 0]
        centroid = centroids[graph_batch]

        def raw(state):
            result = state.new_zeros(state.shape)
            result[:, 0] = (
                state[:, 0] * self.config["S_POS"] + centroid[:, 0]
            )
            result[:, 1] = state[:, 2] * self.config["S_VEL"]
            result[:, 2] = (
                state[:, 1] * self.config["S_POS"] + centroid[:, 1]
            )
            result[:, 3] = state[:, 3] * self.config["S_VEL"]
            return result

        prediction_error = raw(prediction) - target
        update_error = raw(update) - target
        errors = {
            "position_prediction": prediction_error[:, [0, 2]],
            "position_update": update_error[:, [0, 2]],
            "velocity_prediction": prediction_error[:, [1, 3]],
            "velocity_update": update_error[:, [1, 3]],
        }
        event_masks = {"all": valid}
        event_masks.update({
            name: valid & (event_class == value)
            for value, name in EVENT_NAMES.items()
        })
        for local_graph, trajectory_id in enumerate(trajectory_ids.tolist()):
            graph_nodes = graph_batch == local_graph
            for event_name, event_mask in event_masks.items():
                mask = graph_nodes & event_mask
                count = int(mask.sum())
                if count == 0:
                    continue
                bucket = self.values[int(trajectory_id)][event_name]
                bucket["count"] += count
                for metric, error in errors.items():
                    bucket[f"{metric}_sse"] += float(
                        error[mask].square().sum()
                    )

    def result(self):
        rows = []
        for trajectory_id in sorted(self.values):
            events = {}
            for event_name, bucket in self.values[trajectory_id].items():
                row = {
                    "count": int(bucket["count"]),
                }
                for metric in self.METRICS:
                    row[f"{metric}_sse"] = float(
                        bucket[f"{metric}_sse"]
                    )
                events[event_name] = row
            rows.append({
                "trajectory_id": int(trajectory_id),
                "trajectory_index": int(trajectory_id),
                "events": events,
            })
        return rows


def evaluate_mode(
    model,
    trainer,
    loader,
    mode,
    collect_trajectory_sse=False,
    association_augmentation=False,
    association_seed=20260813,
):
    override = {"temporal": 0.0, "learned": None, "expert": 1.0}[mode]
    gasf = getattr(model.decoder, "GASF", None)
    if gasf is not None:
        gasf.interaction_activation_override = override
    latest = {}
    event_accumulator = EventAccumulator(trainer.config)
    trajectory_accumulator = (
        TrajectorySSEAccumulator(trainer.config)
        if collect_trajectory_sse else None
    )

    def gasf_hook(_module, _inputs, output):
        latest["output"] = output

    hook = gasf.register_forward_hook(gasf_hook) if gasf is not None else None

    def observer(**values):
        event_accumulator.add(
            values["outputs"], values["event_class"], latest.get("output"),
            values.get("window"),
        )
        if trajectory_accumulator is not None:
            trajectory_accumulator.add(
                values["graph_data"], values["window"], values["outputs"],
                values["centroids"], values["event_class"],
            )

    trainer.step_observer = observer
    # Enable the two extra ODE counterfactual branches only for the learned
    # common-state pass. Submodules remain in eval mode, so dropout is disabled.
    counterfactual_hook = None
    if mode == "learned":
        counterfactual_hook = model.register_forward_pre_hook(
            lambda module, _inputs: setattr(module, "training", True)
        )
    devices = []
    if trainer.device.type == "cuda":
        devices = [
            trainer.device.index
            if trainer.device.index is not None
            else torch.cuda.current_device()
        ]
    rng_context = (
        torch.random.fork_rng(devices=devices)
        if association_augmentation else nullcontext()
    )
    try:
        with rng_context:
            if association_augmentation:
                torch.manual_seed(int(association_seed))
                if trainer.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(association_seed))
                global_metrics = trainer.validate_association(
                    loader, f"Association comparison: {mode}"
                )
            else:
                global_metrics = trainer.validate(
                    loader, f"SEAN comparison: {mode}"
                )
    finally:
        if hook is not None:
            hook.remove()
        if counterfactual_hook is not None:
            counterfactual_hook.remove()
        trainer.step_observer = None
        if gasf is not None:
            gasf.interaction_activation_override = None
        model.eval()
    result = {"global": global_metrics, "events": event_accumulator.result()}
    if trajectory_accumulator is not None:
        result["trajectories"] = trajectory_accumulator.result()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_trajectories", type=int, default=8)
    parser.add_argument("--start_trajectory", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--gasf_fusion_mode", choices=["add", "concat"], default=None,
        help="override checkpoint fusion mode (legacy V3 checkpoints default to concat)",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["temporal", "learned", "expert"],
        default=["temporal", "learned", "expert"],
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    device = torch.device(cli.device)
    payload = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = device
    args.decoder_use_GASF = True
    args.decoder_use_SEAN = True
    args.model_variant = payload.get("model_variant", "sggt_v2")
    args.covariance_transition = payload.get("covariance_transition", "ode_jacobian")
    args.gasf_filter_mode = "abc"
    args.gasf_graph_mode = "graph"
    args.gasf_recurrent_filter = payload.get(
        "gasf_recurrent_filter", False
    )
    args.gasf_recurrent_gate_initial_update = payload.get(
        "gasf_recurrent_gate_initial_update", 0.1
    )
    args.gasf_fusion_mode = (
        cli.gasf_fusion_mode
        or payload.get("gasf_fusion_mode", "concat")
    )
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
    args.innovation_decomposition = bool(
        payload.get("innovation_decomposition", False)
    )
    args.innovation_whiten_clip = float(
        payload.get("innovation_whiten_clip", 20.0)
    )
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
        cli.data_dir, "val", load_structure=False
    )
    start = max(int(cli.start_trajectory), 0)
    stop = len(dataset)
    if cli.max_trajectories > 0:
        stop = min(start + int(cli.max_trajectories), len(dataset))
    if start >= stop:
        raise ValueError(
            f"empty validation range: start={start}, stop={stop}, "
            f"dataset_size={len(dataset)}"
        )
    dataset = Subset(dataset, range(start, stop))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    results = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -999)),
        "trajectory_count": len(dataset),
        "protocol": {
            "online_rollout": True,
            "encoder_mode": "stream",
            "evaluation_batch_size": 1,
            "validation_start_index": start,
            "gasf_fusion_mode": args.gasf_fusion_mode,
            "temporal_override": 0.0,
            "learned_override": None,
            "expert_override": 1.0,
        },
        "modes": {},
    }
    for mode in cli.modes:
        results["modes"][mode] = evaluate_mode(model, trainer, loader, mode)

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
