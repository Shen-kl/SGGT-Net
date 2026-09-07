"""Differentiable state update from externally computed soft associations.

The association probabilities are deliberately treated as tracker-side
quantities.  Gradients flow through the state update, but not through the
probability calculation used by the training-time augmentation.  This mirrors
the intended deployment in which belief propagation supplies association
probabilities to the learned motion prior.
"""

from typing import NamedTuple

import torch

from config import CONFIG
from model.kalman_utils import CovarianceUtils, MatrixBuilder


class AssociationUpdate(NamedTuple):
    state: torch.Tensor
    covariance: torch.Tensor
    innovation: torch.Tensor
    innovation_covariance: torch.Tensor
    measurement: torch.Tensor
    detection_mass: torch.Tensor
    entropy: torch.Tensor
    spread: torch.Tensor
    valid: torch.Tensor


def _normalise_hypotheses(probabilities, miss_probability, candidate_mask):
    probabilities = torch.where(
        candidate_mask, probabilities.clamp_min(0.0),
        torch.zeros_like(probabilities),
    )
    miss_probability = miss_probability.reshape(-1, 1).clamp_min(0.0)
    total = probabilities.sum(dim=-1, keepdim=True) + miss_probability
    empty = total <= 0.0
    probabilities = probabilities / total.clamp_min(1e-12)
    miss_probability = miss_probability / total.clamp_min(1e-12)
    probabilities = torch.where(empty, torch.zeros_like(probabilities), probabilities)
    miss_probability = torch.where(
        empty, torch.ones_like(miss_probability), miss_probability
    )
    return probabilities, miss_probability


def probabilistic_measurement_update(
    state,
    covariance,
    candidates,
    probabilities,
    miss_probability,
    measurement_covariance,
    candidate_mask=None,
    minimum_detection_mass=0.05,
):
    """Moment-match a missed-detection/measurement posterior mixture.

    Args:
        state: normalized prior state ``[N, 1, 4]`` in ``[x,y,vx,vy]`` order.
        covariance: normalized prior covariance ``[N, 4, 4]``.
        candidates: normalized Cartesian measurements ``[N, C, 2]``.
        probabilities: probability of each measurement hypothesis ``[N, C]``.
        miss_probability: probability of the missed-detection hypothesis.
        measurement_covariance: normalized Cartesian measurement covariance.
        candidate_mask: optional Boolean mask for padded candidates.

    For one candidate with probability one this reduces to the ordinary EKF
    measurement update (up to floating-point roundoff).
    """
    if state.ndim != 3 or state.size(1) != 1:
        raise ValueError(f"state must be [N,1,4], got {tuple(state.shape)}")
    if candidates.ndim != 3 or candidates.size(-1) != 2:
        raise ValueError(
            f"candidates must be [N,C,2], got {tuple(candidates.shape)}"
        )
    if candidate_mask is None:
        candidate_mask = torch.ones_like(probabilities, dtype=torch.bool)
    probabilities, miss_probability = _normalise_hypotheses(
        probabilities, miss_probability, candidate_mask.bool()
    )

    node_count = state.size(0)
    device, dtype = state.device, state.dtype
    scale = MatrixBuilder.build_scaling_matrix(device).to(dtype=dtype)
    observation = MatrixBuilder.build_observation_matrix(device).to(dtype=dtype)
    observation_normalized = observation @ scale
    observation_batch = observation_normalized.unsqueeze(0).expand(
        node_count, -1, -1
    )
    measurement_covariance_physical = (
        measurement_covariance * float(CONFIG["S_POS"]) ** 2
    )
    hp = observation_batch @ covariance
    innovation_covariance = (
        hp @ observation_batch.transpose(-1, -2)
        + measurement_covariance_physical
    )
    pht = covariance @ observation_batch.transpose(-1, -2)
    gain = torch.linalg.solve(
        innovation_covariance, pht.transpose(-1, -2)
    ).transpose(-1, -2)

    prior_state = state.squeeze(1)
    predicted_measurement = state @ observation.transpose(-1, -2)
    innovations = float(CONFIG["S_POS"]) * (
        candidates - predicted_measurement
    )
    candidate_states = prior_state.unsqueeze(1) + (
        gain.unsqueeze(1) @ innovations.unsqueeze(-1)
    ).squeeze(-1)

    identity = torch.eye(
        covariance.size(-1), dtype=dtype, device=device
    ).unsqueeze(0).expand(node_count, -1, -1)
    residual_map = identity - gain @ observation_normalized
    conditional_covariance = (
        residual_map @ covariance @ residual_map.transpose(-1, -2)
        + gain @ measurement_covariance_physical @ gain.transpose(-1, -2)
    )
    conditional_covariance = CovarianceUtils.ensure_symmetry(
        conditional_covariance, eps=0.0
    )

    detection_mass = probabilities.sum(dim=-1, keepdim=True)
    posterior_state = (
        miss_probability * prior_state
        + (probabilities.unsqueeze(-1) * candidate_states).sum(dim=1)
    )
    prior_difference = prior_state - posterior_state
    candidate_difference = candidate_states - posterior_state.unsqueeze(1)
    posterior_covariance = (
        miss_probability.unsqueeze(-1)
        * (
            covariance
            + prior_difference.unsqueeze(-1) @ prior_difference.unsqueeze(-2)
        )
        + (
            probabilities.unsqueeze(-1).unsqueeze(-1)
            * (
                conditional_covariance.unsqueeze(1)
                + candidate_difference.unsqueeze(-1)
                @ candidate_difference.unsqueeze(-2)
            )
        ).sum(dim=1)
    )
    posterior_covariance = CovarianceUtils.ensure_symmetry(
        posterior_covariance, eps=1e-7
    )

    conditional_weights = probabilities / detection_mass.clamp_min(1e-12)
    associated_measurement = (
        conditional_weights.unsqueeze(-1) * candidates
    ).sum(dim=1)
    conditional_innovation = (
        conditional_weights.unsqueeze(-1) * innovations
    ).sum(dim=1)
    measurement_difference = float(CONFIG["S_POS"]) * (
        candidates - associated_measurement.unsqueeze(1)
    )
    spread = (
        conditional_weights.unsqueeze(-1).unsqueeze(-1)
        * measurement_difference.unsqueeze(-1)
        @ measurement_difference.unsqueeze(-2)
    ).sum(dim=1)
    effective_innovation_covariance = CovarianceUtils.ensure_symmetry(
        innovation_covariance + spread, eps=1e-7
    )

    all_probabilities = torch.cat((miss_probability, probabilities), dim=-1)
    entropy = -(
        all_probabilities
        * torch.log(all_probabilities.clamp_min(1e-12))
    ).sum(dim=-1, keepdim=True)
    support_size = candidate_mask.sum(dim=-1, keepdim=True).to(dtype) + 1.0
    entropy = entropy / support_size.log().clamp_min(1e-12)
    valid = detection_mass >= float(minimum_detection_mass)

    associated_measurement = torch.where(
        valid, associated_measurement, predicted_measurement.squeeze(1)
    )
    conditional_innovation = torch.where(
        valid, conditional_innovation, torch.zeros_like(conditional_innovation)
    )
    spread = torch.where(
        valid.unsqueeze(-1), spread, torch.zeros_like(spread)
    )
    effective_innovation_covariance = torch.where(
        valid.unsqueeze(-1),
        effective_innovation_covariance,
        innovation_covariance,
    )
    return AssociationUpdate(
        posterior_state.unsqueeze(1),
        posterior_covariance,
        conditional_innovation.unsqueeze(1),
        effective_innovation_covariance,
        associated_measurement.unsqueeze(1),
        detection_mass.unsqueeze(1),
        entropy.unsqueeze(1),
        spread,
        valid.unsqueeze(1),
    )


class CausalAssociationAugmentor:
    """Generate online missed detections and ambiguous candidate sets.

    Every target receives all detections in its packed scene, shared uniform
    clutter, and optional nearby competitors.  Hypothesis probabilities use
    only the detached current prior, its covariance and the tracker noise
    assumptions; the identity of the originating target is never exposed to
    the neural network.
    """

    def __init__(
        self,
        pd_min=0.75,
        pd_max=0.98,
        clutter_mu_min=0.0,
        clutter_mu_max=0.0,
        surveillance_side_m=3000.0,
        max_clutter=32,
        local_clutter_rate=0.25,
        max_local_clutter=2,
        local_clutter_scale=2.5,
        gate_threshold=9.21034037198,
        minimum_detection_mass=0.05,
        bp_iterations=20,
        include_target_competitors=False,
        clutter_requires_detection=True,
    ):
        self.pd_min = float(pd_min)
        self.pd_max = float(pd_max)
        self.clutter_mu_min = float(clutter_mu_min)
        self.clutter_mu_max = float(clutter_mu_max)
        self.surveillance_side_m = float(surveillance_side_m)
        self.max_clutter = int(max_clutter)
        self.local_clutter_rate = float(local_clutter_rate)
        self.max_local_clutter = int(max_local_clutter)
        self.local_clutter_scale = float(local_clutter_scale)
        self.gate_threshold = float(gate_threshold)
        self.minimum_detection_mass = float(minimum_detection_mass)
        self.bp_iterations = int(bp_iterations)
        self.include_target_competitors = bool(include_target_competitors)
        self.clutter_requires_detection = bool(clutter_requires_detection)
        if not 0.0 < self.pd_min <= self.pd_max <= 1.0:
            raise ValueError("detection probability interval must lie in (0,1]")
        if self.surveillance_side_m <= 0.0:
            raise ValueError("surveillance side must be positive")

    @staticmethod
    def exclusive_association_probabilities(
        log_weight,
        log_miss,
        candidate_mask,
        graph_batch,
        iterations=20,
    ):
        """Approximate one-to-one target/measurement marginals with loopy BP.

        Candidate columns are shared within each packed scene. A column that is
        valid for only one row (for example an importance-sampled local clutter
        point) naturally receives a unit exclusion message and is otherwise an
        ordinary association hypothesis.
        """
        probabilities = torch.zeros_like(log_weight)
        miss_probabilities = torch.ones(
            log_weight.size(0), dtype=log_weight.dtype, device=log_weight.device
        )
        graph_count = int(graph_batch.max().item()) + 1 if graph_batch.numel() else 0
        for graph_index in range(graph_count):
            rows = torch.nonzero(
                graph_batch == graph_index, as_tuple=False
            ).squeeze(-1)
            if not rows.numel():
                continue
            active_columns = candidate_mask[rows].any(dim=0)
            columns = torch.nonzero(active_columns, as_tuple=False).squeeze(-1)
            if not columns.numel():
                continue
            valid = candidate_mask[rows][:, columns]
            local_log_weight = log_weight[rows][:, columns]
            local_log_miss = log_miss[rows]
            row_maximum = torch.maximum(
                local_log_miss,
                torch.where(
                    valid, local_log_weight,
                    torch.full_like(local_log_weight, -torch.inf),
                ).amax(dim=-1),
            )
            weight = torch.where(
                valid,
                torch.exp(local_log_weight - row_maximum.unsqueeze(-1)),
                torch.zeros_like(local_log_weight),
            )
            miss = torch.exp(local_log_miss - row_maximum)
            measurement_to_track = torch.ones_like(weight)
            for _ in range(max(1, int(iterations))):
                total = miss + (weight * measurement_to_track).sum(dim=-1)
                track_to_measurement = torch.where(
                    valid,
                    weight / (
                        total.unsqueeze(-1) - weight * measurement_to_track
                    ).clamp_min(1e-12),
                    torch.zeros_like(weight),
                )
                column_total = track_to_measurement.sum(dim=0, keepdim=True)
                measurement_to_track = torch.where(
                    valid,
                    1.0 / (
                        1.0 + column_total - track_to_measurement
                    ).clamp_min(1e-12),
                    torch.zeros_like(weight),
                )
            associated = weight * measurement_to_track
            normalizer = miss + associated.sum(dim=-1)
            local_probability = associated / normalizer.unsqueeze(-1).clamp_min(1e-12)
            local_miss_probability = miss / normalizer.clamp_min(1e-12)
            for local_row, global_row in enumerate(rows):
                probabilities[global_row, columns] = local_probability[local_row]
            miss_probabilities[rows] = local_miss_probability
        return probabilities, miss_probabilities

    @torch.no_grad()
    def hypotheses(
        self, state, covariance, measurement, measurement_covariance,
        graph_batch, source_valid, position_centroid=None,
    ):
        state = state.detach()
        covariance = covariance.detach()
        measurement = measurement.detach()
        measurement_covariance = measurement_covariance.detach()
        graph_batch = graph_batch.long()
        source_valid = source_valid.bool().reshape(-1)
        node_count = state.size(0)
        graph_count = int(graph_batch.max().item()) + 1 if node_count else 0
        dtype, device = state.dtype, state.device

        pd_graph = torch.empty(graph_count, device=device, dtype=dtype).uniform_(
            self.pd_min, self.pd_max
        )
        pd_node = pd_graph[graph_batch]
        detected = (torch.rand(node_count, device=device) < pd_node) & source_valid

        # All target-originated detections from the same packed scene are
        # candidates.  This is the source of target-to-target ambiguity.
        if position_centroid is None:
            position_centroid = measurement.new_zeros(node_count, 2)
        else:
            position_centroid = position_centroid.detach().reshape(node_count, 2)
        measurement_global = measurement.squeeze(1) + position_centroid
        target_candidates_global = measurement_global.unsqueeze(0).expand(
            node_count, -1, -1
        )
        target_candidates = (
            target_candidates_global - position_centroid.unsqueeze(1)
        )
        if self.include_target_competitors:
            target_mask = (
                graph_batch.unsqueeze(1) == graph_batch.unsqueeze(0)
            ) & detected.unsqueeze(0)
        else:
            target_mask = torch.eye(
                node_count, dtype=torch.bool, device=device
            ) & detected.unsqueeze(0)

        clutter_mu_graph = torch.empty(
            graph_count, device=device, dtype=dtype
        ).uniform_(self.clutter_mu_min, self.clutter_mu_max)
        clutter_count_graph = torch.poisson(clutter_mu_graph).long().clamp(
            0, self.max_clutter
        )
        side = self.surveillance_side_m / float(CONFIG["S_POS"])
        scene_center = measurement_global.new_zeros(graph_count, 2)
        for graph_index in range(graph_count):
            members = (graph_batch == graph_index) & source_valid
            if members.any():
                scene_center[graph_index] = measurement_global[members].mean(dim=0)
        global_clutter = scene_center.unsqueeze(1) + (
            torch.rand(
                graph_count, self.max_clutter, 2, device=device, dtype=dtype
            ) - 0.5
        ) * side
        global_candidates = (
            global_clutter[graph_batch] - position_centroid.unsqueeze(1)
        )
        clutter_index = torch.arange(self.max_clutter, device=device)
        global_mask = clutter_index.unsqueeze(0) < clutter_count_graph[
            graph_batch
        ].unsqueeze(1)
        if self.clutter_requires_detection:
            global_mask = global_mask & detected.unsqueeze(1)

        if self.max_local_clutter:
            local_count = torch.poisson(torch.full(
                (node_count,), self.local_clutter_rate,
                device=device, dtype=dtype,
            )).long().clamp(0, self.max_local_clutter)
            covariance_safe = CovarianceUtils.ensure_symmetry(
                measurement_covariance, eps=1e-8
            )
            root = torch.linalg.cholesky(covariance_safe)
            noise = torch.randn(
                node_count, self.max_local_clutter, 2,
                device=device, dtype=dtype,
            )
            offsets = (
                root.unsqueeze(1) @ noise.unsqueeze(-1)
            ).squeeze(-1) * self.local_clutter_scale
            local_candidates_global = measurement_global.unsqueeze(1) + offsets
            local_index = torch.arange(self.max_local_clutter, device=device)
            local_source_mask = (
                local_index.unsqueeze(0) < local_count.unsqueeze(1)
            ) & source_valid.unsqueeze(1)
            # Give every sampled clutter point one scene-global candidate ID.
            # Each target then expresses that same point in its own local
            # centroid frame; BP exclusion is therefore applied to a genuine
            # shared measurement rather than unrelated row-local columns.
            local_candidates_global = local_candidates_global.reshape(-1, 2)
            local_candidates = (
                local_candidates_global.unsqueeze(0)
                - position_centroid.unsqueeze(1)
            )
            local_owner_graph = graph_batch.repeat_interleave(
                self.max_local_clutter
            )
            local_mask = (
                graph_batch.unsqueeze(1) == local_owner_graph.unsqueeze(0)
            ) & local_source_mask.reshape(1, -1)
            if self.clutter_requires_detection:
                local_mask = local_mask & detected.unsqueeze(1)
        else:
            local_candidates = measurement.new_empty(node_count, 0, 2)
            local_mask = torch.empty(
                node_count, 0, dtype=torch.bool, device=device
            )

        candidates = torch.cat(
            (target_candidates, global_candidates, local_candidates), dim=1
        )
        candidate_mask = torch.cat(
            (target_mask, global_mask, local_mask), dim=1
        )

        scale = MatrixBuilder.build_scaling_matrix(device).to(dtype=dtype)
        observation = MatrixBuilder.build_observation_matrix(device).to(dtype=dtype)
        observation_normalized = observation @ scale
        predicted_measurement = state @ observation.transpose(-1, -2)
        innovations = float(CONFIG["S_POS"]) * (
            candidates - predicted_measurement
        )
        innovation_covariance = (
            observation_normalized.unsqueeze(0) @ covariance
            @ observation_normalized.transpose(-1, -2).unsqueeze(0)
            + measurement_covariance * float(CONFIG["S_POS"]) ** 2
        )
        solved = torch.linalg.solve(
            innovation_covariance.unsqueeze(1), innovations.unsqueeze(-1)
        ).squeeze(-1)
        mahalanobis = (innovations * solved).sum(dim=-1)
        _, logdet = torch.linalg.slogdet(innovation_covariance)
        log_likelihood = -0.5 * (
            mahalanobis + logdet.unsqueeze(-1)
            + 2.0 * torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=dtype))
        )
        clutter_density = (
            clutter_mu_graph[graph_batch]
            / (self.surveillance_side_m ** 2)
        ).clamp_min(1e-12)
        log_weight = (
            pd_node.clamp_min(1e-6).log().unsqueeze(-1)
            + log_likelihood
            - clutter_density.log().unsqueeze(-1)
        )
        # The augmentation knows which simulated detection originated from a
        # target only to construct a valid candidate set. Do not let the
        # augmentation's optional hard gate convert an actually generated
        # detection into an unintended extra miss. Its identity is not exposed
        # to the network or used to assign probability one: likelihood and BP
        # still determine the soft weight among all candidates.
        originated_detection = torch.eye(
            node_count, candidate_mask.size(1),
            dtype=torch.bool, device=device,
        ) & detected.unsqueeze(1)
        gated = candidate_mask & (
            (mahalanobis <= self.gate_threshold) | originated_detection
        )
        log_weight = torch.where(
            gated, log_weight, torch.full_like(log_weight, -torch.inf)
        )
        log_miss = (1.0 - pd_node).clamp_min(1e-6).log().unsqueeze(-1)
        probability, miss_probability = self.exclusive_association_probabilities(
            log_weight,
            log_miss.squeeze(-1),
            gated,
            graph_batch,
            iterations=self.bp_iterations,
        )
        return (
            candidates,
            probability,
            miss_probability,
            gated,
        )

    def update(
        self, state, covariance, measurement, measurement_covariance,
        graph_batch, source_valid, position_centroid=None,
    ):
        candidates, probability, miss, mask = self.hypotheses(
            state, covariance, measurement, measurement_covariance,
            graph_batch, source_valid, position_centroid,
        )
        return probabilistic_measurement_update(
            state,
            covariance,
            candidates,
            probability,
            miss,
            measurement_covariance,
            candidate_mask=mask,
            minimum_detection_mass=self.minimum_detection_mass,
        )
