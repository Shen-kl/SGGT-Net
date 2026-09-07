"""Causal whole-trajectory training for legacy and Dataset2D v3 packs."""

import math
import os
import time
import json
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.training_utils_v3 import CovarianceNormalizer, GraphDataProcessor, TrainingMetrics
from utils.losses_v3 import caculate_evaluation, edge_f1_score, focal_loss
from utils.minmax_scaler import minmax_scaler


class Trainer:
    def __init__(self, model, optimizer, lr_schedule, logger, config, args):
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.logger = logger
        self.config = config
        self.args = args
        self.device = torch.device(args.device)
        self.train_epoches = int(getattr(args, "train_epochs", 100))
        self.P_change_epochs = int(getattr(args, "covariance_reset_epochs", 0))
        self.tf_prob_start = 1.0
        self.tf_prob_min = 0.0
        self.tf_decay_epochs = 20
        self.min_evaluation = float("inf")
        self.min_position_evaluation = float("inf")
        self.min_association_position_evaluation = float("inf")
        self.min_expert_evaluation = float("inf")
        self.min_strong_expert_evaluation = float("inf")
        self.min_gate_evaluation = float("inf")
        self.best_model_path = None

    def get_tf_prob(self, current_epoch):
        return max(
            self.tf_prob_min,
            self.tf_prob_start - current_epoch / self.tf_decay_epochs,
        )

    def compute_measurement_noise(self, graph_data, frame_index=0):
        """Build node-wise Cartesian R from each v3 scene's radar noise."""
        frame_index = min(int(frame_index), graph_data.measurement.size(1) - 1)
        position = graph_data.measurement[:, frame_index]
        x, y = position[:, 0], position[:, 1]
        radius = torch.sqrt(x.square() + y.square()).clamp_min(1.0)
        azimuth = torch.atan2(x, y)

        graph_index = graph_data.batch
        if hasattr(graph_data, "range_noise_std"):
            sigma_r = graph_data.range_noise_std.reshape(-1)[graph_index]
        else:
            sigma_r = torch.full_like(radius, self.config["measurement_noise_std"][0])
        if hasattr(graph_data, "azimuth_noise_std_deg"):
            sigma_a = torch.deg2rad(
                graph_data.azimuth_noise_std_deg.reshape(-1)[graph_index]
            )
        else:
            sigma_a = torch.full_like(radius, self.config["measurement_noise_std"][1])

        sin_a, cos_a = torch.sin(azimuth), torch.cos(azimuth)
        j11, j12 = sin_a, radius * cos_a
        j21, j22 = cos_a, -radius * sin_a
        var_r, var_a = sigma_r.square(), sigma_a.square()
        covariance = position.new_zeros(position.size(0), 2, 2)
        covariance[:, 0, 0] = j11.square() * var_r + j12.square() * var_a
        covariance[:, 1, 1] = j21.square() * var_r + j22.square() * var_a
        covariance[:, 0, 1] = j11 * j21 * var_r + j12 * j22 * var_a
        covariance[:, 1, 0] = covariance[:, 0, 1]
        normalized = CovarianceNormalizer.normalize_measurement_cov(
            covariance, self.config
        )
        return covariance, normalized

    def init_state_covariance(self, measurement_covariance):
        covariance = GraphDataProcessor.init_covariance_from_measurement(
            measurement_covariance,
            self.args.T,
            self.config,
            device=measurement_covariance.device,
        )
        return covariance * float(getattr(self.args, "initial_covariance_scale", 1.0))

    def _event_node_weight(self, graph_data, time_step):
        """Return inverse-frequency scene weights for a legal window start."""
        input_len = self.config["input_graph_window_len"]
        output_len = self.config["output_graph_window_len"]
        legal_count = graph_data.y.size(1) - input_len - output_len + 1
        start = int(time_step) - input_len
        masks = [
            graph_data.steady_window_mask.bool(),
            graph_data.switch_window_mask.bool(),
            graph_data.strong_window_mask.bool(),
        ]
        ratios = self.config.get("window_sampling_ratio", [0.4, 0.3, 0.3])
        graph_weight = graph_data.y.new_zeros(int(graph_data.num_graphs))
        for ratio, mask in zip(ratios, masks):
            mask = mask.reshape(int(graph_data.num_graphs), -1)[:, :legal_count]
            count = mask.sum(dim=1).clamp_min(1).to(graph_weight.dtype)
            graph_weight += float(ratio) * mask[:, start].to(graph_weight.dtype) / count
        graph_weight *= float(legal_count)
        # Metadata-free legacy data are uniform; malformed uncovered starts also
        # receive a neutral weight instead of silently dropping their loss.
        graph_weight = torch.where(
            graph_weight > 0, graph_weight, torch.ones_like(graph_weight)
        )
        return graph_weight[graph_data.batch]

    def _event_node_class(self, graph_data, time_step):
        """Return 0=steady, 1=switch, 2=strong with strong taking priority."""
        input_len = self.config["input_graph_window_len"]
        start = int(time_step) - input_len
        graph_class = torch.full(
            (int(graph_data.num_graphs),), -1, dtype=torch.long,
            device=graph_data.x.device,
        )
        for value, name in ((0, "steady_window_mask"),
                            (1, "switch_window_mask"),
                            (2, "strong_window_mask")):
            mask = getattr(graph_data, name, None)
            if mask is not None and 0 <= start < mask.reshape(int(graph_data.num_graphs), -1).size(1):
                active = mask.reshape(int(graph_data.num_graphs), -1)[:, start].bool()
                graph_class[active] = value
        return graph_class[graph_data.batch]

    @staticmethod
    def _weighted_wta(
        mu, target, mask, node_weight, winners=1, coordinate_weight=None
    ):
        expanded = target.unsqueeze(2).expand_as(mu)
        coordinate_error = F.huber_loss(mu, expanded, reduction="none")
        if coordinate_weight is not None:
            coordinate_error = coordinate_error * coordinate_weight
        error = coordinate_error.sum(dim=-1)
        weighted_mask = mask.float() * node_weight[:, None]
        mixture_error = (error * weighted_mask[..., None]).sum(dim=1)
        values = torch.topk(
            mixture_error, k=min(winners, mu.size(2)), dim=-1, largest=False
        ).values
        denominator = weighted_mask.sum().clamp_min(1.0)
        return values.sum() / denominator

    def compute_loss(
        self,
        outputs,
        graph_window,
        node_weight,
        previous_adjacency=None,
        previous_adjacency_mask=None,
        event_class=None,
        previous_q_score=None,
        demand_positive_prior=None,
    ):
        prediction, covariance = outputs.states, outputs.covariances
        target, dec_mask = outputs.target, outputs.real_mask
        update = outputs.updated_state
        adjacency, adjacency_mask = outputs.adjacency, outputs.adjacency_mask
        winners = min(1, outputs.mixture_coeffs.size(-1))

        coordinate_weight = target.new_tensor([
            getattr(self.args, "position_loss_weight", 1.0),
            getattr(self.args, "position_loss_weight", 1.0),
            getattr(self.args, "velocity_loss_weight", 1.0),
            getattr(self.args, "velocity_loss_weight", 1.0),
        ])
        prediction_loss = self._weighted_wta(
            prediction, target, dec_mask, node_weight, winners,
            coordinate_weight,
        )
        update_loss = self._weighted_wta(
            update.unsqueeze(1), target[:, :1], dec_mask[:, :1], node_weight,
            1, coordinate_weight,
        )
        zero = prediction_loss.new_zeros(())

        structure_loss = zero
        if (
            float(getattr(self.args, "lambda_struct", 0.0)) > 0.0
            and adjacency.numel()
            and getattr(graph_window, "struct_feat", None)
        ):
            structure_target = graph_window.struct_feat[0].to(adjacency).reshape(-1)
            if structure_target.numel() == adjacency.size(0):
                valid = structure_target >= 0
                expanded_target = structure_target.clamp_min(0).view(-1, 1).expand_as(adjacency)
                valid_mask = adjacency_mask & valid.view(-1, 1).expand_as(adjacency)
                if valid_mask.any():
                    structure_loss = focal_loss(
                        adjacency.clamp(1e-6, 1 - 1e-6),
                        expanded_target,
                        valid_mask,
                    )

        temporal_loss = zero
        if (
            float(getattr(self.args, "lambda_temporal", 0.0)) > 0.0
            and previous_adjacency is not None
            and previous_adjacency.shape == adjacency.shape
        ):
            temporal_mask = adjacency_mask & previous_adjacency_mask
            if temporal_mask.any():
                temporal_loss = F.smooth_l1_loss(
                    adjacency[temporal_mask], previous_adjacency[temporal_mask]
                )

        auxiliary_loss = zero
        interaction_aux_expert_loss = zero
        interaction_aux_rank_loss = zero
        interaction_demand_loss = zero
        interaction_demand_balanced_active = zero
        interaction_demand_positive_fraction = zero
        interaction_demand_accuracy = zero
        interaction_demand_tp_count = zero
        interaction_demand_tn_count = zero
        interaction_demand_positive_count = zero
        interaction_demand_negative_count = zero
        interaction_demand_positive_score_sum = zero
        interaction_demand_negative_score_sum = zero
        velocity_delta = None
        maneuver_gain = None
        auxiliary_weighted_mask = None
        if outputs.auxiliary_delta_v.numel() and getattr(self.model.decoder, "use_GASF", False):
            # `outputs.target` is already normalized and reordered to
            # [x, y, vx, vy].  Construct every horizon increment in that same
            # coordinate system to avoid broadcasting a one-step label over
            # the three-step decoder output.
            last_velocity = (
                graph_window.x[:, -1:, [1, 3]] / self.config["S_VEL"]
            ).to(target)
            previous_velocity = torch.cat(
                (last_velocity, target[:, :-1, 2:]), dim=1
            )
            velocity_delta = target[:, :, 2:] - previous_velocity
            maneuver_gain = 1.0 + velocity_delta.norm(dim=-1).clamp(max=4.0)
            auxiliary_error = F.smooth_l1_loss(
                outputs.auxiliary_delta_v, velocity_delta, reduction="none"
            ).mean(dim=-1)
            weighted_mask = dec_mask * node_weight[:, None]
            auxiliary_weighted_mask = weighted_mask
            auxiliary_loss = (
                auxiliary_error * maneuver_gain * weighted_mask
            ).sum() / weighted_mask.sum().clamp_min(1.0)

        if (
            velocity_delta is not None
            and getattr(self.model.decoder, "use_SEAN", False)
            and outputs.auxiliary_delta_v_temporal.numel()
            and outputs.auxiliary_delta_v_expert.numel()
        ):
            temporal_aux_error = F.smooth_l1_loss(
                outputs.auxiliary_delta_v_temporal,
                velocity_delta,
                reduction="none",
            ).mean(dim=-1)
            expert_aux_error = F.smooth_l1_loss(
                outputs.auxiliary_delta_v_expert,
                velocity_delta,
                reduction="none",
            ).mean(dim=-1)
            auxiliary_denominator = auxiliary_weighted_mask.sum().clamp_min(1.0)
            interaction_aux_expert_loss = (
                expert_aux_error * maneuver_gain * auxiliary_weighted_mask
            ).sum() / auxiliary_denominator
            auxiliary_rank_margin = float(
                getattr(self.args, "interaction_aux_rank_margin", 1e-4)
            )
            # Detach the temporal comparator: this term must form a useful
            # interaction increment, not degrade the temporal reference until
            # the inequality becomes true.
            auxiliary_rank = F.relu(
                expert_aux_error
                - temporal_aux_error.detach()
                + auxiliary_rank_margin
            )
            interaction_aux_rank_loss = (
                auxiliary_rank * maneuver_gain * auxiliary_weighted_mask
            ).sum() / auxiliary_denominator

            if getattr(self.model.decoder, "use_SEAN", False):
                activation = outputs.gasf_confidence[:, 0, 0].clamp(
                    1e-5, 1.0 - 1e-5
                )
                # Maneuver demand is a fixed physical trajectory label.  Do
                # not derive it from `velocity_delta`: that auxiliary target
                # uses the online posterior as its reference and therefore
                # changes as the model learns.
                maneuver_delta_v = graph_window.maneuver_delta_v[:, 0].to(
                    activation
                )
                physical_delta_v = torch.linalg.vector_norm(
                    maneuver_delta_v, dim=-1
                )
                demand_target = (
                    physical_delta_v
                    >= float(getattr(
                        self.args, "interaction_maneuver_threshold", 2.0
                    ))
                ).to(activation.dtype)
                demand_weight = dec_mask[:, 0].float() * node_weight
                demand_error = F.binary_cross_entropy(
                    activation, demand_target, reduction="none"
                )
                valid_demand = demand_weight > 0
                if valid_demand.any():
                    if demand_positive_prior is None:
                        positive_prior = demand_target[valid_demand].mean()
                    else:
                        positive_prior = torch.as_tensor(
                            demand_positive_prior,
                            dtype=demand_error.dtype,
                            device=demand_error.device,
                        )
                    positive_prior = positive_prior.clamp(0.05, 0.95)
                    class_weight = (
                        demand_target * (0.5 / positive_prior)
                        + (1.0 - demand_target) * (0.5 / (1.0 - positive_prior))
                    )
                    balanced_weight = demand_weight * class_weight
                    interaction_demand_loss = (
                        demand_error * balanced_weight
                    ).sum() / balanced_weight.sum().clamp_min(1.0)
                    interaction_demand_balanced_active = demand_error.new_ones(())
                    interaction_demand_positive_fraction = demand_target[
                        valid_demand
                    ].mean()
                    demand_prediction = activation[valid_demand] >= 0.5
                    demand_truth = demand_target[valid_demand] >= 0.5
                    interaction_demand_accuracy = (
                        demand_prediction == demand_truth
                    ).float().mean()
                    interaction_demand_tp_count = (
                        demand_prediction & demand_truth
                    ).sum().to(demand_error.dtype)
                    interaction_demand_tn_count = (
                        ~demand_prediction & ~demand_truth
                    ).sum().to(demand_error.dtype)
                    interaction_demand_positive_count = demand_truth.sum().to(
                        demand_error.dtype
                    )
                    interaction_demand_negative_count = (~demand_truth).sum().to(
                        demand_error.dtype
                    )
                    interaction_demand_positive_score_sum = activation[
                        valid_demand
                    ][demand_truth].sum()
                    interaction_demand_negative_score_sum = activation[
                        valid_demand
                    ][~demand_truth].sum()

        interaction_base_loss = zero
        interaction_expert_loss = zero
        interaction_gate_loss = zero
        interaction_regret_loss = zero
        interaction_supervision_active = zero
        interaction_advantage_mean = zero
        interaction_advantage_abs_mean = zero
        interaction_advantage_std = zero
        interaction_expert_better_fraction = zero
        interaction_gate_target_mean = zero
        interaction_gate_target_std = zero
        interaction_gate_balanced_active = zero
        interaction_gate_tp_count = zero
        interaction_gate_tn_count = zero
        interaction_gate_positive_count = zero
        interaction_gate_negative_count = zero
        interaction_gate_positive_score_sum = zero
        interaction_gate_negative_score_sum = zero
        causal_gate_mae = zero
        causal_gate_correlation = zero
        interaction_state_alignment_loss = zero
        interaction_state_improvement_loss = zero
        causal_velocity_guard_loss = zero
        interaction_strong_base_loss = zero
        interaction_strong_expert_loss = zero
        interaction_strong_advantage = zero
        if outputs.counterfactual_base.numel() and outputs.counterfactual_expert.numel():
            base_state = outputs.counterfactual_base[:, 0, 0]
            expert_state = outputs.counterfactual_expert[:, 0, 0]
            mixed_state = prediction[:, 0, 0]
            first_target = target[:, 0]
            training_stage = getattr(self.args, "training_stage", "joint")
            graph_expert_stage = training_stage in {
                "innovation_expert", "graph_expert", "graph_context", "graph_context_rollout",
                "graph_context_bridge", "physical_gain", "causal_innovation_expert"
            }
            gate_stage = training_stage in {"gate", "causal_innovation_gate"}
            if graph_expert_stage or gate_stage:
                # Preprocessing scales (1500 m, 300 m/s) are deliberately
                # broad enough for the complete dataset, but make the local
                # expert residual almost zero numerically.  Train this stage
                # in task-scale physical units instead.
                physical_scale = first_target.new_tensor([
                    float(self.config["S_POS"]),
                    float(self.config["S_POS"]),
                    float(self.config["S_VEL"]),
                    float(self.config["S_VEL"]),
                ])
                task_scale = first_target.new_tensor([
                    float(getattr(
                        self.args, "interaction_expert_position_scale", 25.0
                    )),
                    float(getattr(
                        self.args, "interaction_expert_position_scale", 25.0
                    )),
                    float(getattr(
                        self.args, "interaction_expert_velocity_scale", 5.0
                    )),
                    float(getattr(
                        self.args, "interaction_expert_velocity_scale", 5.0
                    )),
                ]).clamp_min(1e-6)
                if gate_stage:
                    # The paper's primary endpoint is position estimation.  A
                    # velocity-only local win must not activate an expert that
                    # worsens the position rollout.
                    branch_weight = first_target.new_tensor(
                        [1.0, 1.0, 0.0, 0.0]
                    ).view(1, -1)
                else:
                    branch_weight = first_target.new_tensor([
                        float(getattr(
                            self.args, "interaction_alignment_position_weight", 1.0
                        )),
                        float(getattr(
                            self.args, "interaction_alignment_position_weight", 1.0
                        )),
                        float(getattr(
                            self.args, "interaction_alignment_velocity_weight", 1.0
                        )),
                        float(getattr(
                            self.args, "interaction_alignment_velocity_weight", 1.0
                        )),
                    ]).view(1, -1)
            else:
                physical_scale = first_target.new_ones(4)
                task_scale = first_target.new_ones(4)
                branch_weight = coordinate_weight.view(1, -1)

            def branch_error(state):
                normalized_error = (
                    (state - first_target) * physical_scale / task_scale
                )
                return (
                    F.huber_loss(
                        normalized_error,
                        torch.zeros_like(normalized_error),
                        reduction="none",
                    )
                    * branch_weight
                ).sum(dim=-1)

            base_error = branch_error(base_state)
            expert_error = branch_error(expert_state)
            mixed_error = branch_error(mixed_state)
            first_mask = dec_mask[:, 0].float() * node_weight
            denominator = first_mask.sum().clamp_min(1.0)
            interaction_base_loss = (base_error * first_mask).sum() / denominator
            expert_weight = first_mask.clone()
            if event_class is not None:
                event_weight = torch.ones_like(expert_weight)
                event_weight = torch.where(
                    event_class == 0,
                    event_weight.new_full((), float(getattr(
                        self.args, "interaction_expert_steady_weight", 1.0
                    ))),
                    event_weight,
                )
                event_weight = torch.where(
                    event_class == 1,
                    event_weight.new_full((), float(getattr(
                        self.args, "interaction_expert_switch_weight", 1.0
                    ))),
                    event_weight,
                )
                event_weight = torch.where(
                    event_class == 2,
                    event_weight.new_full((), float(getattr(
                        self.args, "interaction_expert_strong_weight", 1.0
                    ))),
                    event_weight,
                )
                expert_weight = expert_weight * event_weight
            expert_denominator = expert_weight.sum().clamp_min(1.0)
            interaction_expert_loss = (
                expert_error * expert_weight
            ).sum() / expert_denominator
            if event_class is not None:
                strong_weight = first_mask * (event_class == 2).to(first_mask.dtype)
                strong_denominator = strong_weight.sum().clamp_min(1.0)
                interaction_strong_base_loss = (
                    base_error * strong_weight
                ).sum() / strong_denominator
                interaction_strong_expert_loss = (
                    expert_error * strong_weight
                ).sum() / strong_denominator
                interaction_strong_advantage = (
                    (base_error.detach() - expert_error) * strong_weight
                ).sum() / strong_denominator

            correction = expert_state - base_state.detach()
            ideal_correction = first_target.detach() - base_state.detach()

            def directional_error(coordinate_slice, physical_scale):
                correction_part = correction[:, coordinate_slice] * physical_scale
                ideal_part = ideal_correction[:, coordinate_slice] * physical_scale
                valid_direction = torch.linalg.vector_norm(
                    ideal_part, dim=-1
                ) > 1e-8
                cosine = F.cosine_similarity(
                    correction_part, ideal_part, dim=-1, eps=1e-8
                )
                return (1.0 - cosine) * valid_direction.to(cosine.dtype)

            position_direction_error = directional_error(
                slice(0, 2), float(self.config["S_POS"])
            )
            velocity_direction_error = directional_error(
                slice(2, 4), float(self.config["S_VEL"])
            )
            position_alignment_weight = float(getattr(
                self.args, "interaction_alignment_position_weight", 1.0
            ))
            velocity_alignment_weight = float(getattr(
                self.args, "interaction_alignment_velocity_weight", 1.0
            ))
            alignment_weight_sum = max(
                position_alignment_weight + velocity_alignment_weight, 1e-8
            )
            state_alignment_error = (
                position_alignment_weight * position_direction_error
                + velocity_alignment_weight * velocity_direction_error
            ) / alignment_weight_sum
            interaction_state_alignment_loss = (
                state_alignment_error * expert_weight
            ).sum() / expert_denominator

            expert_margin = float(getattr(
                self.args, "interaction_gate_margin", 1e-5
            ))
            state_improvement_error = F.relu(
                expert_error - base_error.detach() + expert_margin
            )
            interaction_state_improvement_loss = (
                state_improvement_error * expert_weight
            ).sum() / expert_denominator
            if training_stage == "causal_innovation_expert":
                velocity_scale = max(float(getattr(
                    self.args, "interaction_expert_velocity_scale", 5.0
                )), 1e-6)
                base_velocity_error = F.huber_loss(
                    (base_state[:, 2:] - first_target[:, 2:])
                    * float(self.config["S_VEL"]) / velocity_scale,
                    torch.zeros_like(base_state[:, 2:]),
                    reduction="none",
                ).sum(dim=-1)
                expert_velocity_error = F.huber_loss(
                    (expert_state[:, 2:] - first_target[:, 2:])
                    * float(self.config["S_VEL"]) / velocity_scale,
                    torch.zeros_like(expert_state[:, 2:]),
                    reduction="none",
                ).sum(dim=-1)
                velocity_regret = F.relu(
                    expert_velocity_error - base_velocity_error.detach()
                    + float(getattr(
                        self.args, "causal_velocity_guard_margin", 0.0
                    ))
                )
                causal_velocity_guard_loss = (
                    velocity_regret * expert_weight
                ).sum() / expert_denominator

            if gate_stage:
                position_scale = float(self.config["S_POS"])
                base_gate_error = torch.linalg.vector_norm(
                    (base_state[:, :2] - first_target[:, :2]) * position_scale,
                    dim=-1,
                )
                expert_gate_error = torch.linalg.vector_norm(
                    (expert_state[:, :2] - first_target[:, :2]) * position_scale,
                    dim=-1,
                )
                mixed_gate_error = torch.linalg.vector_norm(
                    (mixed_state[:, :2] - first_target[:, :2]) * position_scale,
                    dim=-1,
                )
                advantage = base_gate_error - expert_gate_error
                margin = float(getattr(
                    self.args, "interaction_gate_position_margin_m", 1e-4
                ))
                regret_reference = torch.minimum(
                    base_gate_error, expert_gate_error
                ).detach()
                regret_error = mixed_gate_error
            else:
                advantage = base_error - expert_error
                margin = float(getattr(
                    self.args, "interaction_gate_margin", 1e-5
                ))
                regret_reference = torch.minimum(
                    base_error, expert_error
                ).detach()
                regret_error = mixed_error
            informative = advantage.abs() > margin
            gate_event_weight = torch.ones_like(first_mask)
            if gate_stage and event_class is not None:
                gate_event_weight = torch.where(
                    event_class == 0,
                    gate_event_weight.new_full((), float(getattr(
                        self.args, "interaction_gate_steady_weight", 1.0
                    ))),
                    gate_event_weight,
                )
                gate_event_weight = torch.where(
                    event_class == 1,
                    gate_event_weight.new_full((), float(getattr(
                        self.args, "interaction_gate_switch_weight", 1.5
                    ))),
                    gate_event_weight,
                )
                gate_event_weight = torch.where(
                    event_class == 2,
                    gate_event_weight.new_full((), float(getattr(
                        self.args, "interaction_gate_strong_weight", 2.0
                    ))),
                    gate_event_weight,
                )
            supervised_weight = (
                first_mask * gate_event_weight * informative.to(first_mask.dtype)
            )
            # The sign of a sufficiently large counterfactual advantage is
            # the task label. The former fixed-temperature sigmoid compressed
            # the observed 1e-6--1e-5 differences into targets near 0.5 and
            # therefore trained a global bias instead of a dynamic selector.
            gate_target = (advantage.detach() > margin).to(advantage.dtype)
            valid_counterfactual = first_mask > 0
            if valid_counterfactual.any():
                valid_advantage = advantage.detach()[valid_counterfactual]
                valid_gate_target = gate_target[valid_counterfactual]
                interaction_advantage_mean = valid_advantage.mean()
                interaction_advantage_abs_mean = valid_advantage.abs().mean()
                interaction_advantage_std = valid_advantage.std(unbiased=False)
                interaction_expert_better_fraction = (
                    valid_advantage > margin
                ).float().mean()
                interaction_gate_target_mean = valid_gate_target.mean()
                interaction_gate_target_std = valid_gate_target.std(unbiased=False)
            activation = (
                outputs.causal_innovation_activation[:, 0, 0]
                if training_stage == "causal_innovation_gate"
                else outputs.gasf_confidence[:, 0, 0]
            ).clamp(1e-5, 1.0 - 1e-5)
            if training_stage == "causal_innovation_gate":
                # The instantaneous branch winner is dominated by current
                # measurement noise and proved non-predictable on held-out
                # trajectories.  Train the selector against a deployable,
                # continuous demand derived strictly from past NIS and
                # own/peer innovation agreement.
                demand_target = outputs.causal_innovation_demand[:, 0, 0].detach()
                supervised_weight = first_mask * outputs.causal_innovation_eligible[
                    :, 0, 0
                ].to(first_mask.dtype)
                supervised_denominator = supervised_weight.sum().clamp_min(1.0)
                gate_error = F.smooth_l1_loss(
                    activation, demand_target, reduction="none", beta=0.1
                )
                interaction_gate_loss = (
                    gate_error * supervised_weight
                ).sum() / supervised_denominator
                interaction_gate_balanced_active = (
                    supervised_weight.sum() > 0
                ).to(activation.dtype)
                interaction_supervision_active = (
                    supervised_weight > 0
                ).float().mean()
                supervised = supervised_weight > 0
                if supervised.any():
                    target_positive = demand_target[supervised] >= 0.75
                    predicted_positive = activation[supervised] >= 0.75
                    interaction_gate_tp_count = (
                        predicted_positive & target_positive
                    ).sum().to(activation.dtype)
                    interaction_gate_tn_count = (
                        ~predicted_positive & ~target_positive
                    ).sum().to(activation.dtype)
                    interaction_gate_positive_count = target_positive.sum().to(
                        activation.dtype
                    )
                    interaction_gate_negative_count = (~target_positive).sum().to(
                        activation.dtype
                    )
                    supervised_activation = activation[supervised]
                    interaction_gate_positive_score_sum = supervised_activation[
                        target_positive
                    ].sum()
                    interaction_gate_negative_score_sum = supervised_activation[
                        ~target_positive
                    ].sum()
                interaction_gate_target_mean = demand_target[
                    supervised
                ].mean() if supervised.any() else zero
                interaction_gate_target_std = demand_target[
                    supervised
                ].std(unbiased=False) if supervised.any() else zero
                if supervised.any():
                    supervised_activation = activation[supervised]
                    supervised_target = demand_target[supervised]
                    causal_gate_mae = (
                        supervised_activation - supervised_target
                    ).abs().mean()
                    centered_activation = (
                        supervised_activation - supervised_activation.mean()
                    )
                    centered_target = supervised_target - supervised_target.mean()
                    correlation_denominator = torch.sqrt(
                        centered_activation.square().sum()
                        * centered_target.square().sum()
                    ).clamp_min(1e-8)
                    causal_gate_correlation = (
                        centered_activation * centered_target
                    ).sum() / correlation_denominator
                interaction_regret_loss = zero
            else:
                gate_error = F.binary_cross_entropy(
                    activation, gate_target, reduction="none"
                )
                positive_weight = supervised_weight * gate_target
                negative_weight = supervised_weight * (1.0 - gate_target)
                positive_count = positive_weight.sum()
                negative_count = negative_weight.sum()
                # A balanced loss removes the global class-prior shortcut. Skip a
                # local step unless it contains evidence for both decisions; the
                # trajectory-batch optimizer will aggregate many such steps.
                if positive_count > 0 and negative_count > 0:
                    interaction_gate_loss = 0.5 * (
                        (gate_error * positive_weight).sum() / positive_count
                        + (gate_error * negative_weight).sum() / negative_count
                    )
                    interaction_gate_balanced_active = interaction_gate_loss.new_ones(())
                interaction_supervision_active = informative.float().mean()
                supervised = supervised_weight > 0
                if supervised.any():
                    predicted_positive = activation[supervised] >= 0.5
                    target_positive = gate_target[supervised].bool()
                    interaction_gate_tp_count = (
                        predicted_positive & target_positive
                    ).sum().to(activation.dtype)
                    interaction_gate_tn_count = (
                        ~predicted_positive & ~target_positive
                    ).sum().to(activation.dtype)
                    interaction_gate_positive_count = target_positive.sum().to(
                        activation.dtype
                    )
                    interaction_gate_negative_count = (~target_positive).sum().to(
                        activation.dtype
                    )
                    supervised_activation = activation[supervised]
                    interaction_gate_positive_score_sum = supervised_activation[
                        target_positive
                    ].sum()
                    interaction_gate_negative_score_sum = supervised_activation[
                        ~target_positive
                    ].sum()
                regret = F.relu(
                    regret_error - regret_reference - margin
                )
                interaction_regret_loss = (
                    regret * first_mask
                ).sum() / denominator

        nll_loss = zero
        if covariance.numel():
            difference = (target.unsqueeze(2) - prediction).unsqueeze(-1)
            eye = torch.eye(covariance.size(-1), device=covariance.device)
            stable_covariance = covariance + 1e-5 * eye
            solution = torch.linalg.solve(stable_covariance, difference)
            mahalanobis = (
                difference.transpose(-1, -2) @ solution
            ).squeeze(-1).squeeze(-1)
            logdet = torch.linalg.slogdet(stable_covariance).logabsdet
            best_nll = (0.5 * (mahalanobis + logdet)).min(dim=2).values
            best_nll = best_nll.clamp(min=-20.0, max=100.0)
            weighted_mask = dec_mask * node_weight[:, None]
            nll_loss = (best_nll * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

        q_rank_loss = zero
        q_surprise_loss = zero
        q_surprise_active = zero
        q_smooth_loss = zero
        q_baseline_loss = zero
        if outputs.q_acceleration_multiplier.numel():
            multiplier = outputs.q_acceleration_multiplier[:, 0, 0].clamp_min(1e-6)
            position_variance = outputs.q_position_noise[:, 0, 0, [0, 1], [0, 1]].sum(-1)
            position_variance = position_variance * float(self.config["S_POS"]) ** 2
            position_ceiling = max(
                2.0 * float(getattr(
                    self.args, "adaptive_q_max_position_variance", 400.0
                )), 1e-6
            )
            # Combine two dimensionless fractions. The former implementation
            # added a multiplier to variance in m^2, so the position head
            # dominated ranking purely because of units.
            q_score = (
                torch.log(multiplier)
                + torch.log1p(position_variance / position_ceiling)
            )
            if event_class is not None:
                steady = event_class == 0
                maneuver = event_class >= 1
                if steady.any() and maneuver.any():
                    q_rank_loss = F.relu(
                        float(getattr(self.args, "q_rank_margin", 0.25))
                        + q_score[steady].mean() - q_score[maneuver].mean()
                    )
                if steady.any():
                    # At steady times the legacy acceleration scale (1) and no
                    # extra position noise are the physically conservative base.
                    q_baseline_loss = (
                        q_score[steady].square().mean()
                        + (
                            position_variance[steady] / position_ceiling
                        ).square().mean()
                    )
            if previous_q_score is not None:
                q_smooth_loss = F.smooth_l1_loss(q_score, previous_q_score.detach())
            surprise = outputs.q_innovation_score[:, 0, 0].detach()
            finite = torch.isfinite(surprise) & torch.isfinite(q_score)
            if finite.sum() >= 4:
                surprise_valid = surprise[finite]
                q_valid = q_score[finite]
                fraction = min(max(
                    float(getattr(self.args, "q_surprise_quantile", 0.25)), 0.05
                ), 0.5)
                high_threshold = float(getattr(
                    self.args, "q_surprise_high_threshold", 0.0
                ))
                low_threshold = float(getattr(
                    self.args, "q_surprise_low_threshold", 0.0
                ))
                if high_threshold > 0.0:
                    high_index = torch.where(surprise_valid >= high_threshold)[0]
                    low_index = torch.where(surprise_valid <= low_threshold)[0]
                else:
                    count = max(1, int(fraction * surprise_valid.numel()))
                    order = torch.argsort(surprise_valid)
                    low_index, high_index = order[:count], order[-count:]
                if high_index.numel() == 0 or low_index.numel() == 0:
                    high_index = low_index = surprise_valid.new_empty(0, dtype=torch.long)
                surprise_gap = (
                    surprise_valid[high_index].mean() - surprise_valid[low_index].mean()
                    if high_index.numel() and low_index.numel()
                    else surprise_valid.new_zeros(())
                )
                if high_index.numel() and low_index.numel() and float(surprise_gap) >= float(
                    getattr(self.args, "q_surprise_min_gap", 0.1)
                ):
                    q_surprise_active = q_score.new_ones(())
                    q_surprise_loss = F.relu(
                        float(getattr(self.args, "q_surprise_margin", 0.25))
                        + q_valid[low_index].mean() - q_valid[high_index].mean()
                    )
                    high_target = float(getattr(
                        self.args, "q_surprise_high_target", 0.0
                    ))
                    low_target = float(getattr(
                        self.args, "q_surprise_low_target", 0.0
                    ))
                    anchor = q_score.new_zeros(())
                    if high_target > 0.0:
                        anchor = anchor + F.relu(
                            math.log(high_target) - q_valid[high_index].mean()
                        )
                    if low_target > 0.0:
                        anchor = anchor + F.relu(
                            q_valid[low_index].mean() - math.log(low_target)
                        )
                    q_surprise_loss = q_surprise_loss + float(getattr(
                        self.args, "q_surprise_anchor_weight", 1.0
                    )) * anchor

        # A learned residual should not be rewarded for moving the causal CV
        # initialization farther from the label.  This is a one-sided regret
        # term: improvements over CV are free, and only degradation beyond the
        # configured margin is penalized.  The reference uses only the last
        # posterior state in the causal input window, never future measurements.
        cv_guard_loss = zero
        if float(getattr(self.args, "lambda_cv_guard", 0.0)) > 0.0:
            last = graph_window.x[:, -1]
            centroid = torch.zeros_like(last[:, [0, 2]])
            # The target/model coordinates are centered on the final input
            # group centroid. Recover it directly from the raw causal state.
            for graph_idx in range(int(graph_window.num_graphs)):
                members = graph_window.batch == graph_idx
                if members.any():
                    centroid[members] = last[members][:, [0, 2]].mean(dim=0)
            cv = target.new_zeros(target[:, :1].shape)
            cv[:, 0, 0] = (
                last[:, 0] + float(self.args.T) * last[:, 1] - centroid[:, 0]
            ) / self.config["S_POS"]
            cv[:, 0, 1] = (
                last[:, 2] + float(self.args.T) * last[:, 3] - centroid[:, 1]
            ) / self.config["S_POS"]
            cv[:, 0, 2] = last[:, 1] / self.config["S_VEL"]
            cv[:, 0, 3] = last[:, 3] / self.config["S_VEL"]
            model_error = (prediction[:, :1] - target[:, :1].unsqueeze(2)).square()
            cv_error = (cv - target[:, :1]).square().unsqueeze(2)
            coordinate_weight = target.new_tensor([
                getattr(self.args, "cv_guard_position_weight", 1.0),
                getattr(self.args, "cv_guard_position_weight", 1.0),
                getattr(self.args, "cv_guard_velocity_weight", 1.0),
                getattr(self.args, "cv_guard_velocity_weight", 1.0),
            ])
            regret = F.relu(
                model_error - cv_error.detach()
                - float(getattr(self.args, "cv_guard_margin", 0.0))
            ) * coordinate_weight
            best_regret = regret.sum(dim=-1).min(dim=2).values
            weighted_mask = dec_mask[:, :1].float() * node_weight[:, None]
            cv_guard_loss = (
                best_regret * weighted_mask
            ).sum() / weighted_mask.sum().clamp_min(1.0)

        total = (
            prediction_loss
            + update_loss
            + getattr(self.args, "lambda_struct", 0.5) * structure_loss
            + getattr(self.args, "lambda_temporal", 0.05) * temporal_loss
            + getattr(self.args, "lambda_aux", 0.1) * auxiliary_loss
            + getattr(self.args, "lambda_nll", 0.05) * nll_loss
            + getattr(self.args, "lambda_cv_guard", 0.0) * cv_guard_loss
            + getattr(self.args, "lambda_q_rank", 0.0) * q_rank_loss
            + getattr(self.args, "lambda_q_smooth", 0.0) * q_smooth_loss
            + getattr(self.args, "lambda_q_baseline", 0.0) * q_baseline_loss
            + getattr(self.args, "lambda_q_surprise", 0.0) * q_surprise_loss
            + getattr(self.args, "lambda_interaction_base", 0.05)
            * interaction_base_loss
            + getattr(self.args, "lambda_interaction_expert", 0.1)
            * interaction_expert_loss
            + getattr(self.args, "lambda_interaction_gate", 1e-3)
            * interaction_gate_loss
            + getattr(self.args, "lambda_interaction_regret", 0.1)
            * interaction_regret_loss
            + getattr(self.args, "lambda_interaction_aux_expert", 0.25)
            * interaction_aux_expert_loss
            + getattr(self.args, "lambda_interaction_aux_rank", 0.1)
            * interaction_aux_rank_loss
            + getattr(self.args, "lambda_interaction_demand", 0.01)
            * interaction_demand_loss
            + getattr(self.args, "lambda_interaction_state_alignment", 0.0)
            * interaction_state_alignment_loss
            + getattr(self.args, "lambda_interaction_state_improvement", 0.0)
            * interaction_state_improvement_loss
        )
        if getattr(self.args, "training_stage", "joint") in {
            "innovation_expert", "graph_expert", "graph_context", "graph_context_rollout",
            "graph_context_bridge", "physical_gain", "causal_innovation_expert"
        }:
            total = (
                getattr(self.args, "lambda_interaction_expert", 1.0)
                * interaction_expert_loss
                + getattr(
                    self.args, "lambda_interaction_state_alignment", 1.0
                ) * interaction_state_alignment_loss
                + getattr(
                    self.args, "lambda_interaction_state_improvement", 1.0
                ) * interaction_state_improvement_loss
                + getattr(
                    self.args, "lambda_causal_velocity_guard", 0.0
                ) * causal_velocity_guard_loss
            )
        elif getattr(self.args, "training_stage", "joint") == "q_adapt":
            # Prediction state is frozen in this stage. Q is identified by
            # the differentiable EKF posterior loss and state Gaussian NLL;
            # optional causal ranking/smoothness terms only regularize it.
            total = (
                update_loss
                + getattr(self.args, "lambda_nll", 0.0) * nll_loss
                + getattr(self.args, "lambda_q_rank", 0.0) * q_rank_loss
                + getattr(self.args, "lambda_q_smooth", 0.0) * q_smooth_loss
                + getattr(self.args, "lambda_q_baseline", 0.0) * q_baseline_loss
                + getattr(self.args, "lambda_q_surprise", 0.0) * q_surprise_loss
            )
        elif getattr(self.args, "training_stage", "joint") in {
            "gate", "causal_innovation_gate"
        }:
            # Only the selector is trainable. Keep its objective explicit so
            # unrelated frozen-model losses cannot obscure whether the
            # activation policy itself learned.
            total = (
                interaction_gate_loss
                + getattr(self.args, "lambda_interaction_regret", 0.1)
                * interaction_regret_loss
            )
        self.last_loss_tensors = {
            "prediction": prediction_loss,
            "update": update_loss,
            "structure": structure_loss,
            "temporal": temporal_loss,
            "auxiliary": auxiliary_loss,
            "nll": nll_loss,
            "cv_guard": cv_guard_loss,
            "q_rank": q_rank_loss,
            "q_smooth": q_smooth_loss,
            "q_baseline": q_baseline_loss,
            "q_surprise": q_surprise_loss,
            "q_surprise_active": q_surprise_active,
            "interaction_base": interaction_base_loss,
            "interaction_expert": interaction_expert_loss,
            "interaction_strong_base": interaction_strong_base_loss,
            "interaction_strong_expert": interaction_strong_expert_loss,
            "interaction_strong_advantage": interaction_strong_advantage,
            "interaction_gate": interaction_gate_loss,
            "interaction_regret": interaction_regret_loss,
            "interaction_aux_expert": interaction_aux_expert_loss,
            "interaction_aux_rank": interaction_aux_rank_loss,
            "interaction_state_alignment": interaction_state_alignment_loss,
            "interaction_state_improvement": interaction_state_improvement_loss,
            "causal_velocity_guard": causal_velocity_guard_loss,
            "interaction_demand": interaction_demand_loss,
            "interaction_demand_balanced_active": interaction_demand_balanced_active,
            "interaction_demand_positive_fraction": interaction_demand_positive_fraction,
            "interaction_demand_accuracy": interaction_demand_accuracy,
            "interaction_demand_tp_count": interaction_demand_tp_count,
            "interaction_demand_tn_count": interaction_demand_tn_count,
            "interaction_demand_positive_count": interaction_demand_positive_count,
            "interaction_demand_negative_count": interaction_demand_negative_count,
            "interaction_demand_positive_score_sum": interaction_demand_positive_score_sum,
            "interaction_demand_negative_score_sum": interaction_demand_negative_score_sum,
            "interaction_gate_balanced_active": interaction_gate_balanced_active,
            "interaction_supervision_active": interaction_supervision_active,
            "interaction_advantage_mean": interaction_advantage_mean,
            "interaction_advantage_abs_mean": interaction_advantage_abs_mean,
            "interaction_advantage_std": interaction_advantage_std,
            "interaction_expert_better_fraction": interaction_expert_better_fraction,
            "interaction_gate_target_mean": interaction_gate_target_mean,
            "interaction_gate_target_std": interaction_gate_target_std,
            "interaction_gate_tp_count": interaction_gate_tp_count,
            "interaction_gate_tn_count": interaction_gate_tn_count,
            "interaction_gate_positive_count": interaction_gate_positive_count,
            "interaction_gate_negative_count": interaction_gate_negative_count,
            "interaction_gate_positive_score_sum": interaction_gate_positive_score_sum,
            "interaction_gate_negative_score_sum": interaction_gate_negative_score_sum,
            "causal_gate_mae": causal_gate_mae,
            "causal_gate_correlation": causal_gate_correlation,
        }
        return total

    def _denormalize_prior(self, outputs, centroids, graph_batch):
        weights = torch.softmax(outputs.mixture_coeffs, dim=-1)
        prior = (outputs.states[:, 0] * weights.unsqueeze(-1)).sum(dim=1)
        raw = prior.new_zeros(prior.shape)
        raw[:, 0] = prior[:, 0] * self.config["S_POS"] + centroids[graph_batch, 0]
        raw[:, 2] = prior[:, 1] * self.config["S_POS"] + centroids[graph_batch, 1]
        raw[:, 1] = prior[:, 2] * self.config["S_VEL"]
        raw[:, 3] = prior[:, 3] * self.config["S_VEL"]
        return raw.unsqueeze(1).detach()

    def _denormalize_update(self, outputs, centroids, graph_batch):
        """Convert the first-step EKF posterior to raw [x, vx, y, vy]."""
        update = outputs.updated_state[:, 0].detach()
        raw = update.new_zeros(update.shape)
        raw[:, 0] = update[:, 0] * self.config["S_POS"] + centroids[graph_batch, 0]
        raw[:, 2] = update[:, 1] * self.config["S_POS"] + centroids[graph_batch, 1]
        raw[:, 1] = update[:, 2] * self.config["S_VEL"]
        raw[:, 3] = update[:, 3] * self.config["S_VEL"]
        return raw.unsqueeze(1)

    def compute_evaluation_metrics(self, graph_window, outputs, centroids):
        values = caculate_evaluation(
            graph_window,
            outputs.states,
            outputs.updated_state,
            centroids,
            self.config,
            1,
        )
        (loc_pred, vel_pred, loc_update, vel_update,
         update_raw, loc_measurement) = values
        prediction_count = float(
            graph_window.real_mask[..., 0].to(torch.float32).sum().item()
        )
        update_count = float(
            graph_window.real_mask[:, 0, 0].to(torch.float32).sum().item()
        )
        f1 = 0.0
        if outputs.adjacency.numel() and getattr(graph_window, "struct_feat", None):
            target = graph_window.struct_feat[0].to(outputs.adjacency).reshape(-1)
            prediction = outputs.adjacency.detach().reshape(outputs.adjacency.size(0), -1)[:, 0]
            valid = target >= 0
            if valid.any() and target.numel() == prediction.numel():
                f1, _, _ = edge_f1_score(prediction[valid], target[valid])
        return {
            "location_rmse_prediction": loc_pred,
            "velocity_rmse_prediction": vel_pred,
            "location_rmse_update": loc_update,
            "velocity_rmse_update": vel_update,
            "location_rmse_measurement": loc_measurement,
            "f1": f1,
            # Preserve sufficient statistics so epoch metrics are pooled over
            # valid target points instead of averaging per-batch RMSE values.
            # The latter changes when --eval_batch_size changes.
            "location_rmse_prediction_sse": loc_pred ** 2 * prediction_count,
            "velocity_rmse_prediction_sse": vel_pred ** 2 * prediction_count,
            "location_rmse_update_sse": loc_update ** 2 * update_count,
            "velocity_rmse_update_sse": vel_update ** 2 * update_count,
            "location_rmse_measurement_sse": loc_measurement ** 2 * prediction_count,
            "prediction_count": prediction_count,
            "update_count": update_count,
        }, update_raw

    def _run_trajectory(
        self,
        graph_data,
        training,
        current_epoch=0,
        association_augmentation_override=None,
    ):
        graph_data = graph_data.to(self.device, non_blocking=True)
        measurement_covariance, _ = self.compute_measurement_noise(graph_data, 0)
        covariance = self.init_state_covariance(measurement_covariance)
        initial_covariance = covariance.clone()
        metrics = {key: [] for key in TrainingMetrics().metrics}
        previous_adjacency = previous_adjacency_mask = None
        input_len = self.config["input_graph_window_len"]
        output_len = self.config["output_graph_window_len"]
        last_time_step = graph_data.y.size(1) - output_len
        legal = torch.arange(input_len, last_time_step + 1, device=self.device)
        num_graphs = int(graph_data.batch.max().item()) + 1
        selected_graph_steps = torch.ones(
            num_graphs, graph_data.y.size(1), dtype=torch.bool, device=self.device
        )
        if training and not bool(getattr(self.args, 'full_time_supervision', True)):
            selected_graph_steps.zero_()
            counts = list(getattr(self.args, 'v3_window_counts', [8, 6, 6]))
            names = ('steady_window_mask', 'switch_window_mask', 'strong_window_mask')
            event_masks = {
                name: getattr(graph_data, name).reshape(num_graphs, -1)
                for name in names if hasattr(graph_data, name)
            }
            # Window labels index causal starts; decoder time indexes the first
            # target frame, hence the input_len offset. Sample independently
            # for every scene so batching does not change the class mixture.
            legal_starts = legal - input_len
            desired = min(sum(counts), legal.numel())
            for graph_idx in range(num_graphs):
                selected = torch.zeros_like(legal, dtype=torch.bool)
                # Give rare/important classes priority when labels overlap.
                for name, count in reversed(list(zip(names, counts))):
                    event = event_masks.get(name)
                    if event is None:
                        continue
                    candidates = torch.nonzero(
                        event[graph_idx, legal_starts] & ~selected,
                        as_tuple=False,
                    ).flatten()
                    if candidates.numel():
                        take = min(int(count), candidates.numel())
                        chosen = candidates[
                            torch.randperm(candidates.numel(), device=self.device)[:take]
                        ]
                        selected[chosen] = True
                remaining = torch.nonzero(~selected, as_tuple=False).flatten()
                if int(selected.sum()) < desired and remaining.numel():
                    take = min(desired - int(selected.sum()), remaining.numel())
                    chosen = remaining[
                        torch.randperm(remaining.numel(), device=self.device)[:take]
                    ]
                    selected[chosen] = True
                selected_graph_steps[graph_idx, legal[selected]] = True
        truth_delta_v = (
            graph_data.y[:, legal, :][:, :, [1, 3]]
            - graph_data.y[:, legal - 1, :][:, :, [1, 3]]
        )
        demand_truth = torch.linalg.vector_norm(truth_delta_v, dim=-1) >= float(
            getattr(self.args, 'interaction_maneuver_threshold', 2.0)
        )
        demand_valid = (
            graph_data.real_mask[:, legal, 0].bool()
            & graph_data.real_mask[:, legal - 1, 0].bool()
        )
        if training:
            demand_valid = demand_valid & selected_graph_steps[
                graph_data.batch
            ][:, legal]
        demand_positive_prior = (
            demand_truth[demand_valid].float().mean()
            if demand_valid.any()
            else graph_data.y.new_tensor(0.5)
        )
        start_bias = 0
        window = GraphDataProcessor.init_input_window(
            graph_data, self.config, self.args, start_bias=start_bias
        )
        association_active = (
            training and getattr(self.args, 'association_augmentation', False)
            if association_augmentation_override is None
            else bool(association_augmentation_override)
        )
        self.model.association_augmentation_active = bool(
            association_active
            and getattr(self.model, 'association_augmentation_enabled', False)
        )
        normalized_window = window.clone()
        tf_prob = self.get_tf_prob(current_epoch) if training else 0.0

        loop_start = input_len
        loop_end = last_time_step
        step_mode = getattr(self.args, 'optimizer_step_mode', 'trajectory_batch')
        accumulated_gradient = False
        total_selected_graph_steps = selected_graph_steps[:, legal].sum().clamp_min(1)
        if training and step_mode == 'trajectory_batch':
            self.optimizer.zero_grad(set_to_none=True)
        previous_q_score = None
        encoder_output_cache = None
        encoder_hidden_cache = None
        gasf_recurrent_state = None
        stream_encoder = (
            getattr(self.args, 'encoder_execution_mode', 'stream') == 'stream'
        )
        stream_reset_interval = max(
            0, int(getattr(self.args, 'encoder_stream_reset_interval', 0))
        )
        for time_step in range(loop_start, loop_end + 1):
            normalized_window, centroids = minmax_scaler(
                window, normalized_window, self.config, self.args.neural_net
            )
            measurement_r, normalized_r = self.compute_measurement_noise(
                graph_data, time_step
            )
            selected_graphs = selected_graph_steps[:, time_step]
            scored = bool(selected_graphs.any())
            with torch.set_grad_enabled(training and scored):
                stream_offset = time_step - loop_start
                reset_stream = (
                    encoder_output_cache is None
                    or not stream_encoder
                    or (
                        stream_reset_interval > 0
                        and stream_offset % stream_reset_interval == 0
                    )
                )
                if reset_stream:
                    outputs = self.model(
                        normalized_window,
                        tf_prob,
                        covariance,
                        None,
                        normalized_r,
                        gasf_recurrent_state=gasf_recurrent_state,
                    )
                else:
                    encoder_step = normalized_window.clone()
                    encoder_step.x = normalized_window.x[:, -1:, :]
                    encoder_step.edge_index = normalized_window.edge_index[-1:]
                    encoder_step.edge_features = normalized_window.edge_features[-1:]
                    step_output, step_mixture = self.model.encoder(
                        encoder_step, encoder_hidden_cache
                    )
                    new_hidden = step_output[:, -1]
                    rolling_encoder = torch.cat(
                        (encoder_output_cache[:, 1:], new_hidden.unsqueeze(1)),
                        dim=1,
                    )
                    outputs = self.model(
                        normalized_window,
                        tf_prob,
                        covariance,
                        None,
                        normalized_r,
                        encoder_output_cache=rolling_encoder,
                        mixture_coeffs_cache=step_mixture,
                        gasf_recurrent_state=gasf_recurrent_state,
                    )
                encoder_output_cache = outputs.encoder_output_cache.detach()
                encoder_hidden_cache = outputs.encoder_hidden.detach()
                gasf_recurrent_state = (
                    outputs.gasf_recurrent_state.detach()
                    if outputs.gasf_recurrent_state is not None
                    else None
                )
            interval = int(getattr(self.args, 'finite_check_interval', 0))
            run_diagnostics = interval > 0 and (time_step - loop_start) % interval == 0
            diagnostic_tensors = {
                'window_x': normalized_window.x,
                'residual': normalized_window.residual,
                'R': normalized_r,
                'P_in': covariance,
                'states': outputs.states,
                'P_pred': outputs.covariances,
                'updated_state': outputs.updated_state,
                'P_update': outputs.updated_covariance,
                'process_aux_dv': outputs.auxiliary_delta_v,
                'encoder_cache': outputs.encoder_output_cache,
                'decoder_cache': outputs.decoder_hidden_cache,
                'mixture_coeffs': outputs.mixture_coeffs_cache,
                'q_multiplier': outputs.q_acceleration_multiplier,
                'q_position': outputs.q_position_noise,
                'interaction_activation': outputs.gasf_confidence,
                'gasf_parameter_update_gates': (
                    outputs.gasf_parameter_update_gates
                ),
                'counterfactual_base': outputs.counterfactual_base,
                'counterfactual_expert': outputs.counterfactual_expert,
                'measurement_innovation': outputs.measurement_innovation,
                'innovation_covariance': outputs.innovation_covariance,
                'whitened_innovation': outputs.whitened_innovation,
                'nis': outputs.nis,
                'causal_innovation_correction': outputs.causal_innovation_correction,
                'causal_innovation_activation': outputs.causal_innovation_activation,
                'causal_innovation_demand': outputs.causal_innovation_demand,
                'association_detection_mass': outputs.association_detection_mass,
                'association_entropy': outputs.association_entropy,
                'association_spread': outputs.association_spread,
            }
            if run_diagnostics:
                bad = []
                for name, tensor in diagnostic_tensors.items():
                    if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                        finite = tensor[torch.isfinite(tensor)]
                        bad.append(
                            f"{name}(finite={finite.numel()}/{tensor.numel()},"
                            f"max={float(finite.abs().max()) if finite.numel() else float('nan')})"
                        )
                if bad:
                    raise FloatingPointError(
                        f"Non-finite forward at epoch={current_epoch}, time={time_step}: "
                        + ', '.join(bad)
                    )
            loss = None
            if scored:
                # Full-time supervision targets the same uniform valid-point
                # population as global validation RMSE. Event reweighting is
                # retained only for the legacy sampled-window protocol.
                uniform_training = bool(
                    getattr(self.args, 'full_time_supervision', True)
                )
                node_weight = (
                    self._event_node_weight(graph_data, time_step)
                    if training and not uniform_training
                    else torch.ones(graph_data.x.size(0), device=self.device)
                )
                if training:
                    node_weight = node_weight * selected_graphs[graph_data.batch].to(
                        node_weight.dtype
                    )
                loss = self.compute_loss(
                    outputs,
                    window,
                    node_weight,
                    previous_adjacency,
                    previous_adjacency_mask,
                    self._event_node_class(graph_data, time_step),
                    previous_q_score,
                    demand_positive_prior,
                )
            if outputs.q_acceleration_multiplier.numel():
                multiplier = outputs.q_acceleration_multiplier[:, 0, 0].clamp_min(1e-6)
                pos_var = outputs.q_position_noise[:, 0, 0, [0, 1], [0, 1]].sum(-1)
                position_ceiling = max(
                    2.0 * float(getattr(
                        self.args, "adaptive_q_max_position_variance", 400.0
                    )), 1e-6
                )
                previous_q_score = (
                    torch.log(multiplier)
                    + torch.log1p(
                        pos_var * float(self.config["S_POS"]) ** 2
                        / position_ceiling
                    )
                ).detach()
                q_valid = outputs.real_mask[:, 0].bool()
                if q_valid.any():
                    multiplier_valid = multiplier[q_valid].detach()
                    position_valid = (
                        pos_var[q_valid].detach()
                        * float(self.config["S_POS"]) ** 2
                        / 2.0
                    )
                    surprise_valid = outputs.q_innovation_score[
                        q_valid, 0, 0
                    ].detach()
                    metrics["q_acceleration_multiplier_mean"].append(
                        float(multiplier_valid.mean())
                    )
                    metrics["q_acceleration_multiplier_std"].append(
                        float(multiplier_valid.std(unbiased=False))
                    )
                    metrics["q_position_variance_mean_m2"].append(
                        float(position_valid.mean())
                    )
                    metrics["q_position_variance_std_m2"].append(
                        float(position_valid.std(unbiased=False))
                    )
                    metrics["q_causal_surprise_mean"].append(
                        float(surprise_valid.mean())
                    )
                    metrics["q_causal_surprise_std"].append(
                        float(surprise_valid.std(unbiased=False))
                    )
            if outputs.gasf_parameter_update_gates.numel():
                gate_valid = outputs.real_mask[:, 0].bool()
                if gate_valid.any():
                    recurrent_gate = outputs.gasf_parameter_update_gates[
                        gate_valid, 0
                    ].detach()
                    metrics["gasf_parameter_update_gate_mean"].append(
                        float(recurrent_gate.mean())
                    )
                    metrics["gasf_parameter_update_gate_std"].append(
                        float(recurrent_gate.std(unbiased=False))
                    )
            previous_adjacency = outputs.adjacency.detach()
            previous_adjacency_mask = outputs.adjacency_mask.detach()
            if scored:
                metrics["loss"].append(float(loss.detach()))
                for metric_name, tensor_name in (
                    ("loss_q_rank", "q_rank"),
                    ("loss_q_surprise", "q_surprise"),
                    ("loss_q_smooth", "q_smooth"),
                    ("loss_q_baseline", "q_baseline"),
                    ("q_surprise_active", "q_surprise_active"),
                    ("loss_interaction_base", "interaction_base"),
                    ("loss_interaction_expert", "interaction_expert"),
                    ("loss_interaction_strong_base", "interaction_strong_base"),
                    ("loss_interaction_strong_expert", "interaction_strong_expert"),
                    ("interaction_strong_advantage", "interaction_strong_advantage"),
                    ("loss_interaction_gate", "interaction_gate"),
                    ("loss_interaction_regret", "interaction_regret"),
                    ("loss_interaction_aux_expert", "interaction_aux_expert"),
                    ("loss_interaction_aux_rank", "interaction_aux_rank"),
                    ("loss_interaction_state_alignment", "interaction_state_alignment"),
                    ("loss_interaction_state_improvement", "interaction_state_improvement"),
                    ("loss_causal_velocity_guard", "causal_velocity_guard"),
                    ("loss_interaction_demand", "interaction_demand"),
                    ("interaction_demand_balanced_active", "interaction_demand_balanced_active"),
                    ("interaction_demand_positive_fraction", "interaction_demand_positive_fraction"),
                    ("interaction_demand_accuracy", "interaction_demand_accuracy"),
                    ("interaction_demand_tp_count", "interaction_demand_tp_count"),
                    ("interaction_demand_tn_count", "interaction_demand_tn_count"),
                    ("interaction_demand_positive_count", "interaction_demand_positive_count"),
                    ("interaction_demand_negative_count", "interaction_demand_negative_count"),
                    ("interaction_demand_positive_score_sum", "interaction_demand_positive_score_sum"),
                    ("interaction_demand_negative_score_sum", "interaction_demand_negative_score_sum"),
                    ("interaction_gate_balanced_active", "interaction_gate_balanced_active"),
                    ("interaction_supervision_active", "interaction_supervision_active"),
                    ("interaction_advantage_mean", "interaction_advantage_mean"),
                    ("interaction_advantage_abs_mean", "interaction_advantage_abs_mean"),
                    ("interaction_advantage_std", "interaction_advantage_std"),
                    ("interaction_expert_better_fraction", "interaction_expert_better_fraction"),
                    ("interaction_gate_target_mean", "interaction_gate_target_mean"),
                    ("interaction_gate_target_std", "interaction_gate_target_std"),
                    ("interaction_gate_tp_count", "interaction_gate_tp_count"),
                    ("interaction_gate_tn_count", "interaction_gate_tn_count"),
                    ("interaction_gate_positive_count", "interaction_gate_positive_count"),
                    ("interaction_gate_negative_count", "interaction_gate_negative_count"),
                    ("interaction_gate_positive_score_sum", "interaction_gate_positive_score_sum"),
                    ("interaction_gate_negative_score_sum", "interaction_gate_negative_score_sum"),
                    ("causal_gate_mae", "causal_gate_mae"),
                    ("causal_gate_correlation", "causal_gate_correlation"),
                ):
                    # Balanced gate BCE is only defined when the current
                    # valid population contains both target classes.  The
                    # loss routine deliberately returns zero otherwise; do
                    # not let those sentinel zeros bias the reported mean or
                    # a gate-based checkpoint score downward.
                    if (
                        metric_name == "loss_interaction_gate"
                        and float(
                            self.last_loss_tensors[
                                "interaction_gate_balanced_active"
                            ].detach()
                        ) <= 0.0
                    ):
                        continue
                    metrics[metric_name].append(
                        float(self.last_loss_tensors[tensor_name].detach())
                    )
                activation = (
                    outputs.causal_innovation_activation[:, 0, 0].detach()
                    if outputs.causal_innovation_activation.numel()
                    else outputs.gasf_confidence[:, 0, 0].detach()
                )
                activation_valid = outputs.real_mask[:, 0].bool()
                if activation_valid.any():
                    activation = activation[activation_valid]
                    metrics["interaction_activation_mean"].append(
                        float(activation.mean())
                    )
                    metrics["interaction_activation_std"].append(
                        float(activation.std(unbiased=False))
                    )
                    metrics["interaction_activation_below_005"].append(
                        float((activation < 0.05).float().mean())
                    )
                    metrics["interaction_activation_above_095"].append(
                        float((activation > 0.95).float().mean())
                    )
                if outputs.causal_innovation_correction.numel():
                    causal = outputs.causal_innovation_correction[:, 0].detach()
                    causal_valid = outputs.real_mask[:, 0].bool()
                    eligible = (
                        outputs.causal_innovation_eligible[:, 0, 0].bool()
                        & causal_valid
                    )
                    denominator = causal_valid.sum().clamp_min(1)
                    metrics["causal_innovation_eligible_fraction"].append(
                        float(eligible.sum() / denominator)
                    )
                    if eligible.any():
                        position = causal[eligible, :2] * float(
                            self.config["S_POS"]
                        )
                        velocity = causal[eligible, 2:] * float(
                            self.config["S_VEL"]
                        )
                        metrics["causal_innovation_position_rms_m"].append(
                            float(position.square().mean().sqrt())
                        )
                        metrics["causal_innovation_velocity_rms_mps"].append(
                            float(velocity.square().mean().sqrt())
                        )
                        metrics["causal_innovation_position_max_m"].append(
                            float(position.abs().max())
                        )
                association_population = outputs.real_mask[:, 0].bool()
                if association_population.any():
                    mass = outputs.association_detection_mass[
                        association_population, 0, 0
                    ].detach()
                    entropy = outputs.association_entropy[
                        association_population, 0, 0
                    ].detach()
                    spread_trace = outputs.association_spread[
                        association_population
                    ].diagonal(dim1=-2, dim2=-1).sum(-1).detach()
                    valid_association = outputs.association_valid[
                        association_population, 0, 0
                    ].detach()
                    metrics["association_detection_mass_mean"].append(
                        float(mass.mean())
                    )
                    metrics["association_entropy_mean"].append(
                        float(entropy.mean())
                    )
                    metrics["association_missed_fraction"].append(
                        float((~valid_association).float().mean())
                    )
                    metrics["association_spread_trace_mean"].append(
                        float(spread_trace.mean())
                    )
            if training and scored:
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f'Non-finite v3 loss at epoch={current_epoch}, time={time_step}'
                    )
                if step_mode == 'time_step':
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    clip = float(getattr(self.args, 'grad_clip_norm', 10.0))
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip, error_if_nonfinite=True
                        )
                    self.optimizer.step()
                else:
                    # Weight by the number of selected scenes at this time so
                    # every sampled scene-window contributes equally.
                    gradient_weight = (
                        selected_graphs.sum().to(loss.dtype)
                        / total_selected_graph_steps.to(loss.dtype)
                    )
                    (loss * gradient_weight).backward()
                    accumulated_gradient = True

            metric_interval = max(
                1, int(getattr(self.args, 'train_metric_interval', 8))
            )
            record_evaluation = (
                scored
                and (
                    not training
                    or (time_step - loop_start) % metric_interval == 0
                    or time_step == loop_end
                )
            )
            if record_evaluation:
                evaluation, posterior_raw = self.compute_evaluation_metrics(
                    window, outputs, centroids
                )
                for key, value in evaluation.items():
                    metrics[key].append(value)
                step_observer = getattr(self, 'step_observer', None)
                if step_observer is not None:
                    step_observer(
                        graph_data=graph_data,
                        time_step=time_step,
                        window=window,
                        outputs=outputs,
                        centroids=centroids,
                        event_class=self._event_node_class(graph_data, time_step),
                    )
            else:
                posterior_raw = self._denormalize_update(
                    outputs, centroids, graph_data.batch
                )

            current_valid = graph_data.real_mask[:, time_step, 0].bool()
            previous_valid = graph_data.real_mask[:, time_step - 1, 0].bool()
            continuing = current_valid & previous_valid
            births = current_valid & ~previous_valid
            next_covariance = covariance.detach().clone()
            next_covariance[continuing] = (
                outputs.updated_covariance.detach()[continuing]
            )
            if births.any():
                birth_covariance = self.init_state_covariance(measurement_r)
                next_covariance[births] = birth_covariance[births]
            if training and current_epoch < self.P_change_epochs:
                # Preserve lifecycle initialization even under the legacy
                # covariance-reset curriculum.
                next_covariance[continuing] = initial_covariance[continuing]
            covariance = next_covariance

            if time_step < loop_end:
                prior_raw = self._denormalize_prior(
                    outputs, centroids, graph_data.batch
                )
                bias = time_step - input_len + 1
                window = GraphDataProcessor.update_input_window(
                    window,
                    graph_data,
                    self.config,
                    posterior_raw,
                    prior_raw,
                    bias,
                    self.args,
                    measurement_innovation=outputs.measurement_innovation,
                    innovation_covariance=outputs.innovation_covariance,
                    whitened_innovation=outputs.whitened_innovation,
                    nis=outputs.nis,
                    association_detection_mass=outputs.association_detection_mass,
                    association_entropy=outputs.association_entropy,
                    association_spread=outputs.association_spread,
                    association_valid=outputs.association_valid,
                )

        if training and step_mode == 'trajectory_batch' and accumulated_gradient:
            clip = float(getattr(self.args, 'grad_clip_norm', 10.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip, error_if_nonfinite=True
                )
            self.optimizer.step()
            if int(getattr(self.args, 'finite_check_interval', 0)) > 0:
                bad_parameters = [
                    name for name, parameter in self.model.named_parameters()
                    if not torch.isfinite(parameter).all()
                ]
                if bad_parameters:
                    raise FloatingPointError(
                        'Non-finite parameters after trajectory-batch step: '
                        + ', '.join(bad_parameters[:12])
                    )

        additive_suffixes = ("_sse", "_count", "_sum")
        return {
            key: (
                float(np.sum(values))
                if key.endswith(additive_suffixes)
                else (float(np.mean(values)) if values else 0.0)
            )
            for key, values in metrics.items()
        }

    def train_epoch(self, dataloader, current_epoch):
        self.model.train()
        if getattr(self.args, "training_stage", "joint") == "q_adapt":
            # Frozen dropout/normalization layers must be deterministic during
            # Q-only calibration; only the new adapter remains in train mode.
            self.model.eval()
            self.model.decoder.Q_innovation_encoder.train()
        elif getattr(self.args, "training_stage", "joint") == "expert":
            # Fixed-graph/Temporal feature adaptation keeps every frozen
            # stochastic backbone deterministic. Only the explicitly
            # trainable GASF/fusion/auxiliary modules enter train mode.
            self.model.eval()
            self.model.decoder.GASF.train()
            fusion_mode = getattr(
                self.model.decoder.motion_model, "gasf_fusion_mode", "concat"
            )
            if fusion_mode == "concat":
                self.model.decoder.motion_model.gasf_concat_fusion.train()
            elif fusion_mode == "film":
                self.model.decoder.motion_model.gasf_film.train()
            self.model.decoder.motion_model.aux_velocity_head.train()
        elif getattr(self.args, "training_stage", "joint") in {
            "innovation_expert", "graph_expert", "graph_context", "graph_context_rollout",
            "graph_context_bridge", "physical_gain", "causal_innovation_expert"
        }:
            # Keep the fixed temporal comparator and ODE deterministic. Only
            # the graph construction and interaction projection are trained.
            self.model.eval()
            self.model.training = True
            if getattr(self.args, "training_stage", "joint") == "causal_innovation_expert":
                self.model.causal_innovation_corrector.train()
            elif getattr(self.args, "training_stage", "joint") == "innovation_expert":
                for module in (
                    self.model.decoder.GASF.group_innovation_encoders,
                    self.model.decoder.GASF.individual_innovation_encoders,
                    self.model.decoder.GASF.innovation_uncertainty_encoder,
                    self.model.decoder.GASF.innovation_decomposition_projection,
                    self.model.decoder.SEAN.node_projection,
                    self.model.decoder.SEAN.edge_network,
                ):
                    module.train()
            elif getattr(self.args, "training_stage", "joint") == "graph_context_bridge":
                self.model.decoder.motion_model.gasf_interaction_context_embedding.train()
            elif getattr(self.args, "training_stage", "joint") == "physical_gain":
                pass
            elif getattr(self.args, "training_stage", "joint") in {
                "graph_context", "graph_context_rollout"
            }:
                self.model.decoder.GASF.interaction_innovation_projection.train()
            else:
                self.model.decoder.GASF.interaction_projection.train()
                self.model.decoder.GASF.interaction_innovation_projection.train()
                self.model.decoder.SEAN.node_projection.train()
                self.model.decoder.SEAN.edge_network.train()
        elif getattr(self.args, "training_stage", "joint") == "graph_fusion_adapt":
            # Keep every frozen stochastic layer deterministic. Gradients only
            # calibrate the graph projection and concat fusion path.
            self.model.eval()
            self.model.decoder.GASF.interaction_projection.train()
            self.model.decoder.motion_model.gasf_concat_fusion.train()
        elif getattr(self.args, "training_stage", "joint") == "gate":
            # Frozen dropout must not inject noise into paired labels. Eval
            # mode remains differentiable; counterfactual_evaluation keeps the
            # two fixed expert branches enabled.
            self.model.eval()
            if bool(getattr(
                self.args, "interaction_physical_residual", False
            )):
                self.model.decoder.GASF.context_activation_head.train()
            else:
                self.model.decoder.SEAN.activation_head.train()
        elif getattr(self.args, "training_stage", "joint") == "causal_innovation_gate":
            # The expert and all tracking dynamics remain deterministic; only
            # the causal selector receives counterfactual classification and
            # mixed-branch regret gradients.
            self.model.eval()
            self.model.causal_innovation_corrector.selector.train()
        metrics = TrainingMetrics()
        with tqdm(dataloader, desc=f"Train {current_epoch + 1}/{self.train_epoches}") as progress:
            for graph_data in progress:
                batch_metrics = self._run_trajectory(graph_data, True, current_epoch)
                metrics.add_batch_metrics(batch_metrics)
                progress.set_postfix(metrics.get_batch_mean())
        return metrics.get_epoch_mean()

    def validate(self, dataloader, label="Val"):
        self.model.eval()
        metrics = TrainingMetrics()
        report_quantiles = bool(
            getattr(self.args, 'report_trajectory_quantiles', False)
        )
        trajectory_metrics = {
            'location_rmse_prediction': [],
            'location_rmse_update': [],
        }
        trajectory_records = []
        with torch.no_grad(), tqdm(dataloader, desc=label) as progress:
            for graph_data in progress:
                if report_quantiles and int(graph_data.num_graphs) != 1:
                    raise ValueError(
                        '--report_trajectory_quantiles requires --eval_batch_size 1'
                    )
                batch_metrics = self._run_trajectory(graph_data, False)
                metrics.add_batch_metrics(batch_metrics)
                if report_quantiles:
                    for key in trajectory_metrics:
                        trajectory_metrics[key].append(float(batch_metrics[key]))
                    trajectory_index = int(
                        graph_data.trajectory_index.reshape(-1)[0].item()
                    ) if hasattr(graph_data, 'trajectory_index') else len(trajectory_records)
                    input_len = self.config["input_graph_window_len"]
                    output_len = self.config["output_graph_window_len"]
                    last_time_step = graph_data.y.size(1) - output_len
                    current_valid = graph_data.real_mask[
                        :, input_len:last_time_step + 1, 0
                    ].bool()
                    previous_valid = graph_data.real_mask[
                        :, input_len - 1:last_time_step, 0
                    ].bool()
                    valid_prediction_points = int(
                        (current_valid & previous_valid).sum().item()
                    )
                    trajectory_records.append({
                        'trajectory_index': trajectory_index,
                        'target_file': f'batch_data_{trajectory_index:05d}.pt',
                        'node_count': int(graph_data.x.size(0)),
                        'valid_prediction_points': valid_prediction_points,
                        'location_rmse_measurement': float(
                            batch_metrics['location_rmse_measurement']
                        ),
                        'location_rmse_prediction': float(
                            batch_metrics['location_rmse_prediction']
                        ),
                        'location_rmse_update': float(
                            batch_metrics['location_rmse_update']
                        ),
                        'velocity_rmse_prediction': float(
                            batch_metrics['velocity_rmse_prediction']
                        ),
                        'velocity_rmse_update': float(
                            batch_metrics['velocity_rmse_update']
                        ),
                        'steady_windows': int(
                            graph_data.steady_window_mask.sum().item()
                        ),
                        'switch_windows': int(
                            graph_data.switch_window_mask.sum().item()
                        ),
                        'strong_windows': int(
                            graph_data.strong_window_mask.sum().item()
                        ),
                    })
                progress.set_postfix(metrics.get_batch_mean())
        result = metrics.get_epoch_mean()
        if report_quantiles:
            for key, values in trajectory_metrics.items():
                array = np.asarray(values, dtype=np.float64)
                for percentile in (50, 90, 95, 99):
                    result[f'{key}_p{percentile}'] = float(
                        np.percentile(array, percentile)
                    )
                result[f'{key}_max'] = float(array.max())
            self.last_trajectory_diagnostics = trajectory_records
        return result

    def validate_association(self, dataloader, label="Association validation"):
        """Evaluate the same causal augmentation realization every epoch."""
        self.model.eval()
        metrics = TrainingMetrics()
        seed = int(getattr(self.args, 'association_validation_seed', 20260813))
        devices = []
        if self.device.type == 'cuda':
            devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad(), tqdm(dataloader, desc=label) as progress:
                for graph_data in progress:
                    batch_metrics = self._run_trajectory(
                        graph_data,
                        False,
                        association_augmentation_override=True,
                    )
                    metrics.add_batch_metrics(batch_metrics)
                    progress.set_postfix(metrics.get_batch_mean())
        return metrics.get_epoch_mean()

    def _save_trajectory_diagnostics(self, path, epoch, metrics):
        records = getattr(self, 'last_trajectory_diagnostics', None)
        if not records:
            return
        ranked = sorted(
            records,
            key=lambda item: item['location_rmse_prediction'],
            reverse=True,
        )
        payload = {
            'epoch': int(epoch),
            'protocol_sha256': (
                getattr(self.args, 'evaluation_protocol', {}) or {}
            ).get('protocol_sha256'),
            'global_metrics': {
                key: float(value) for key, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            },
            'worst_prediction_trajectories': ranked[:20],
            'trajectories': records,
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8'
        )

    @staticmethod
    def _evaluation_score(metrics):
        return sum(metrics.get(key, 0.0) for key in (
            "location_rmse_prediction",
            "velocity_rmse_prediction",
            "location_rmse_update",
            "velocity_rmse_update",
        ))

    @staticmethod
    def _position_score(metrics):
        return sum(metrics.get(key, 0.0) for key in (
            "location_rmse_prediction",
            "location_rmse_update",
        ))

    def _selection_score(self, metrics):
        metric = getattr(self.args, "checkpoint_metric", "composite")
        if metric == "association_position":
            association_metrics = getattr(
                self, "last_association_validation_metrics", None
            )
            if association_metrics is None:
                return float("inf")
            return self._position_score(association_metrics)
        if metric == "prediction":
            return metrics.get("location_rmse_prediction", float("inf"))
        if metric == "position":
            return self._position_score(metrics)
        if metric == "expert":
            return metrics.get("loss_interaction_expert", float("inf"))
        if metric == "expert_strong":
            return metrics.get("loss_interaction_strong_expert", float("inf"))
        if metric == "gate":
            return (
                metrics.get("loss_interaction_gate", float("inf"))
                + float(getattr(self.args, "lambda_interaction_regret", 0.1))
                * metrics.get("loss_interaction_regret", float("inf"))
            )
        return self._evaluation_score(metrics)

    def _checkpoint_payload(self, current_epoch, val_metrics):
        return {
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_schedule": self.lr_schedule.state_dict(),
            "epoch": current_epoch,
            "metrics": val_metrics,
            "association_validation_metrics": getattr(
                self, 'last_association_validation_metrics', None
            ),
            "config": self.config,
            "model_variant": getattr(self.model, "model_variant", "sggt_v1"),
            "covariance_transition": getattr(self.model, "covariance_transition", "cv"),
            "evaluation_protocol": getattr(self.args, "evaluation_protocol", None),
            "interaction_physical_residual": bool(getattr(
                self.args, "interaction_physical_residual", False
            )),
            "interaction_physical_max_gain": float(getattr(
                self.args, "interaction_physical_max_gain", 0.075
            )),
            "interaction_physical_init_gain": float(getattr(
                self.args, "interaction_physical_init_gain", 0.05
            )),
            "interaction_activation_floor": float(getattr(
                self.args, "interaction_activation_floor", 0.0
            )),
            "gasf_fusion_mode": getattr(
                self.args, "gasf_fusion_mode", "concat"
            ),
            "gasf_graph_mode": getattr(
                self.args, "gasf_graph_mode", "graph"
            ),
            "gasf_filter_mode": getattr(
                self.args, "gasf_filter_mode", "abc"
            ),
            "gasf_recurrent_filter": getattr(
                self.args, "gasf_recurrent_filter", True
            ),
            "gasf_recurrent_gate_initial_update": getattr(
                self.args, "gasf_recurrent_gate_initial_update", 0.1
            ),
            "gasf_aux_source": getattr(
                self.args, "gasf_aux_source", "gasf"
            ),
            "adaptive_q_mode": getattr(
                self.args, "adaptive_q_mode", "legacy"
            ),
            "adaptive_q_max_multiplier": float(getattr(
                self.args, "adaptive_q_max_multiplier", 5.0
            )),
            "adaptive_q_max_position_variance": float(getattr(
                self.args, "adaptive_q_max_position_variance", 400.0
            )),
            "adaptive_q_initial_multiplier": float(getattr(
                self.args, "adaptive_q_initial_multiplier", 1.0
            )),
            "adaptive_q_initial_position_variance": float(getattr(
                self.args, "adaptive_q_initial_position_variance", 0.01
            )),
            "adaptive_q_acceleration": bool(getattr(
                self.args, "adaptive_q_acceleration", True
            )),
            "decoder_use_SEAN": bool(getattr(
                self.args, "decoder_use_SEAN", False
            )),
            "decoder_use_GASF": bool(getattr(
                self.args, "decoder_use_GASF", False
            )),
            "innovation_decomposition": bool(getattr(
                self.args, "innovation_decomposition", False
            )),
            "innovation_whiten_clip": float(getattr(
                self.args, "innovation_whiten_clip", 20.0
            )),
            "causal_innovation_correction": bool(getattr(
                self.args, "causal_innovation_correction", False
            )),
            "causal_innovation_history": int(getattr(
                self.args, "causal_innovation_history", 8
            )),
            "causal_innovation_hidden_size": int(getattr(
                self.args, "causal_innovation_hidden_size", 32
            )),
            "causal_innovation_max_position_m": float(getattr(
                self.args, "causal_innovation_max_position_m", 50.0
            )),
            "causal_innovation_max_velocity_mps": float(getattr(
                self.args, "causal_innovation_max_velocity_mps", 5.0
            )),
            "causal_innovation_dynamic_activation": bool(getattr(
                self.args, "causal_innovation_dynamic_activation", False
            )),
            "causal_innovation_activation_floor": float(getattr(
                self.args, "causal_innovation_activation_floor", 0.5
            )),
            "causal_innovation_activation_nis_low": float(getattr(
                self.args, "causal_innovation_activation_nis_low", 1.38629436112
            )),
            "causal_innovation_activation_nis_high": float(getattr(
                self.args, "causal_innovation_activation_nis_high", 5.99146454711
            )),
            "causal_innovation_minimum_valid_history": int(getattr(
                self.args, "causal_innovation_minimum_valid_history", 0
            )),
            "causal_innovation_recent_valid_window": int(getattr(
                self.args, "causal_innovation_recent_valid_window", 3
            )),
            "causal_innovation_association_context": bool(getattr(
                self.args, "causal_innovation_association_context", False
            )),
            "causal_innovation_use_peer": bool(getattr(
                self.args, "causal_innovation_use_peer", True
            )),
            "association_augmentation": bool(getattr(
                self.args, "association_augmentation", False
            )),
            "association_validation": bool(getattr(
                self.args, "association_validation", False
            )),
            "association_validation_seed": int(getattr(
                self.args, "association_validation_seed", 20260813
            )),
            "association_pd_min": float(getattr(self.args, "association_pd_min", 0.75)),
            "association_pd_max": float(getattr(self.args, "association_pd_max", 0.98)),
            "association_clutter_mu_min": float(getattr(
                self.args, "association_clutter_mu_min", 0.0
            )),
            "association_clutter_mu_max": float(getattr(
                self.args, "association_clutter_mu_max", 0.0
            )),
            "association_surveillance_side_m": float(getattr(
                self.args, "association_surveillance_side_m", 3000.0
            )),
            "association_max_clutter": int(getattr(
                self.args, "association_max_clutter", 32
            )),
            "association_local_clutter_rate": float(getattr(
                self.args, "association_local_clutter_rate", 0.25
            )),
            "association_max_local_clutter": int(getattr(
                self.args, "association_max_local_clutter", 2
            )),
            "association_local_clutter_scale": float(getattr(
                self.args, "association_local_clutter_scale", 2.5
            )),
            "association_gate_threshold": float(getattr(
                self.args, "association_gate_threshold", 9.21034037198
            )),
            "association_minimum_detection_mass": float(getattr(
                self.args, "association_minimum_detection_mass", 0.05
            )),
            "association_include_target_competitors": bool(getattr(
                self.args, "association_include_target_competitors", False
            )),
            "association_clutter_requires_detection": bool(getattr(
                self.args, "association_clutter_requires_detection", True
            )),
            "lambda_causal_velocity_guard": float(getattr(
                self.args, "lambda_causal_velocity_guard", 0.0
            )),
            "causal_velocity_guard_margin": float(getattr(
                self.args, "causal_velocity_guard_margin", 0.0
            )),
        }

    def save_checkpoint(self, current_epoch, val_metrics):
        checkpoint_dir = Path(self.args.output_dir) / "GNN_groupTargetTracking"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload(current_epoch, val_metrics)
        torch.save(payload, checkpoint_dir / "last.pth")

        composite = self._evaluation_score(val_metrics)
        if composite < self.min_evaluation:
            self.min_evaluation = composite
            path = checkpoint_dir / "best_composite.pth"
            torch.save(payload, path)
            if getattr(self.args, "checkpoint_metric", "composite") == "composite":
                self.best_model_path = path

        position = self._position_score(val_metrics)
        if position < self.min_position_evaluation:
            self.min_position_evaluation = position
            path = checkpoint_dir / "best_position.pth"
            torch.save(payload, path)
            if getattr(self.args, "checkpoint_metric", "composite") == "position":
                self.best_model_path = path

        association_metrics = getattr(
            self, "last_association_validation_metrics", None
        )
        if association_metrics is not None:
            association_position = self._position_score(association_metrics)
            if association_position < self.min_association_position_evaluation:
                self.min_association_position_evaluation = association_position
                path = checkpoint_dir / "best_association_position.pth"
                torch.save(payload, path)
                if (
                    getattr(self.args, "checkpoint_metric", "composite")
                    == "association_position"
                ):
                    self.best_model_path = path

        prediction = val_metrics.get("location_rmse_prediction", float("inf"))
        if not hasattr(self, "min_prediction_evaluation"):
            self.min_prediction_evaluation = float("inf")
        if prediction < self.min_prediction_evaluation:
            self.min_prediction_evaluation = prediction
            path = checkpoint_dir / "best_prediction.pth"
            torch.save(payload, path)
            self._save_trajectory_diagnostics(
                checkpoint_dir / 'best_prediction_trajectory_diagnostics.json',
                current_epoch,
                val_metrics,
            )
            if getattr(self.args, "checkpoint_metric", "composite") == "prediction":
                self.best_model_path = path

        expert = val_metrics.get("loss_interaction_expert", float("inf"))
        if expert < self.min_expert_evaluation:
            self.min_expert_evaluation = expert
            path = checkpoint_dir / "best_expert.pth"
            torch.save(payload, path)
            if getattr(self.args, "checkpoint_metric", "composite") == "expert":
                self.best_model_path = path

        strong_expert = val_metrics.get(
            "loss_interaction_strong_expert", float("inf")
        )
        if strong_expert < self.min_strong_expert_evaluation:
            self.min_strong_expert_evaluation = strong_expert
            path = checkpoint_dir / "best_expert_strong.pth"
            torch.save(payload, path)
            if getattr(self.args, "checkpoint_metric", "composite") == "expert_strong":
                self.best_model_path = path

        gate = (
            val_metrics.get("loss_interaction_gate", float("inf"))
            + float(getattr(self.args, "lambda_interaction_regret", 0.1))
            * val_metrics.get("loss_interaction_regret", float("inf"))
        )
        if gate < self.min_gate_evaluation:
            self.min_gate_evaluation = gate
            path = checkpoint_dir / "best_gate.pth"
            torch.save(payload, path)
            if getattr(self.args, "checkpoint_metric", "composite") == "gate":
                self.best_model_path = path

    def log_metrics(self, current_epoch, train_metrics, val_metrics):
        for prefix, values in (("train", train_metrics), ("validation", val_metrics)):
            for key, value in values.items():
                self.logger.debug(f"{prefix}_{key}:{float(value):.6f},index:{current_epoch}")

    def train_and_evaluation(
        self, train_dataloader, val_dataloader, iid_dataloader=None, ood_dataloader=None
    ):
        process = psutil.Process(os.getpid())
        start_epoch = int(getattr(self.args, "start_epoch", 0))
        # Record the untrained physical-CV initialization explicitly for a new
        # run.  On resume, label this as the preceding epoch instead of
        # incorrectly calling an already-trained checkpoint epoch -1.
        val_metrics = self.validate(val_dataloader, "Validation epoch -1")
        if bool(getattr(self.args, 'association_validation', False)):
            self.last_association_validation_metrics = self.validate_association(
                val_dataloader, "Association validation epoch -1"
            )
            for key, value in self.last_association_validation_metrics.items():
                self.logger.debug(
                    f"association_validation_{key}:{float(value):.6f},index:-1"
                )
        reference_epoch = -1 if start_epoch == 0 else start_epoch - 1
        for key, value in val_metrics.items():
            self.logger.debug(
                f"validation_{key}:{float(value):.6f},index:{reference_epoch}"
            )
        self.save_checkpoint(reference_epoch, val_metrics)
        if start_epoch == 0:
            checkpoint_dir = (
                Path(self.args.output_dir) / "GNN_groupTargetTracking"
            )
            initial_payload = self._checkpoint_payload(reference_epoch, val_metrics)
            torch.save(initial_payload, checkpoint_dir / "initial.pth")
            baseline_record = {
                "epoch": -1,
                "protocol_sha256": (
                    getattr(self.args, "evaluation_protocol", {}) or {}
                ).get("protocol_sha256"),
                "clean_validation": {
                    key: float(value) for key, value in val_metrics.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                },
                "association_validation": {
                    key: float(value) for key, value in (
                        getattr(self, "last_association_validation_metrics", {}) or {}
                    ).items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                },
            }
            (checkpoint_dir / "epoch_minus_one_metrics.json").write_text(
                json.dumps(baseline_record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        best_score = self._selection_score(val_metrics)
        bad_epochs = 0
        patience = int(getattr(self.args, "early_stopping_patience", 10))
        min_delta = float(getattr(self.args, "early_stopping_min_delta", 1e-3))
        for epoch in range(start_epoch, self.train_epoches):
            if (
                bool(getattr(self.args, "staged_joint_finetune", False))
                and epoch == int(getattr(self.args, "fusion_adapt_epochs", 0))
            ):
                for parameter in self.model.parameters():
                    parameter.requires_grad = True
                print(
                    f"Fusion adaptation complete at epoch {epoch}; "
                    "unfroze the full network"
                )
            print(f"RAM: {process.memory_info().rss / 1024 ** 3:.2f} GB")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            train_metrics = self.train_epoch(train_dataloader, epoch)
            val_metrics = self.validate(val_dataloader, "Validation")
            if bool(getattr(self.args, 'association_validation', False)):
                self.last_association_validation_metrics = self.validate_association(
                    val_dataloader, "Association validation"
                )
                for key, value in self.last_association_validation_metrics.items():
                    self.logger.debug(
                        "association_validation_"
                        f"{key}:{float(value):.6f},index:{epoch}"
                    )
            if isinstance(self.lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_schedule.step(val_metrics.get("loss", 0.0))
            else:
                self.lr_schedule.step()
            self.log_metrics(epoch, train_metrics, val_metrics)
            self.save_checkpoint(epoch, val_metrics)
            score = self._selection_score(val_metrics)
            if score < best_score - min_delta:
                best_score = score
                bad_epochs = 0
            else:
                bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    print(
                        f"Early stopping at epoch {epoch}: no validation "
                        f"improvement for {bad_epochs} epochs"
                    )
                    break

        results = {"validation": val_metrics}
        if self.best_model_path is not None:
            try:
                checkpoint = torch.load(
                    self.best_model_path, map_location=self.device, weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            print(f"Best model: {self.best_model_path}")
        if iid_dataloader is not None:
            results["iid_test"] = self.validate(iid_dataloader, "IID test")
        if ood_dataloader is not None:
            results["ood_test"] = self.validate(ood_dataloader, "OOD test")
        for split, values in results.items():
            print(f"{split}: {values}")
        return results


def train_and_evaluation(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    lr_schedule,
    logger,
    config,
    args,
    iid_dataloader=None,
    ood_dataloader=None,
):
    trainer = Trainer(model, optimizer, lr_schedule, logger, config, args)
    return trainer.train_and_evaluation(
        train_dataloader, val_dataloader, iid_dataloader, ood_dataloader
    )
