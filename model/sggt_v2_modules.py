import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kalman_utils import CovarianceUtils


class StructureOutput(NamedTuple):
    adjacency_vector: torch.Tensor
    structural_features: torch.Tensor
    vector_mask: torch.Tensor
    group_count: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    node_entropy: torch.Tensor
    activation: torch.Tensor


class SpectralFilterState(NamedTuple):
    """Per-node, per-scale causal state for the recurrent ABC filter."""

    filter_parameters: torch.Tensor
    valid: torch.Tensor

    def detach(self):
        return SpectralFilterState(
            self.filter_parameters.detach(),
            self.valid.detach(),
        )


class SpectralOutput(NamedTuple):
    latent: torch.Tensor
    confidence: torch.Tensor
    scale_weights: torch.Tensor
    temporal_latent: torch.Tensor
    interaction_delta: torch.Tensor
    interaction_context: torch.Tensor
    structure: StructureOutput
    recurrent_state: SpectralFilterState
    parameter_update_gates: torch.Tensor
    parameter_step_norms: torch.Tensor
    filter_response_step_rms: torch.Tensor
    recurrent_transition_valid: torch.Tensor
    spectral_calibration_loss: torch.Tensor


def _upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    row, col = torch.triu_indices(
        matrix.size(0), matrix.size(1), offset=1, device=matrix.device
    )
    return matrix[row, col]


class TemporalProbabilisticSEAN(nn.Module):
    """Infer a symmetric soft interaction graph from temporal node features."""

    def __init__(self, hidden_dim: int, struct_dim: int, context_dim: int = 4):
        super().__init__()
        self.context_dim = int(context_dim)
        self.node_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_network = nn.Sequential(
            nn.Linear(2 * hidden_dim + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.struct_projection = nn.Sequential(
            nn.Linear(hidden_dim, struct_dim),
            nn.LayerNorm(struct_dim),
            nn.SiLU(),
        )
        self.activation_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 + self.context_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # The old -2 bias produced alpha ~= 0.119 and, while the activation was
        # disconnected, made a frozen head look like a learned sparse policy.
        # Start at the neutral alpha=0.5. The temporal counterfactual and regret
        # objective protect the base path, while this point maximizes the
        # selector's sigmoid derivative and avoids a long bias-only calibration.
        # Give the selector observable sample-to-sample variation from the
        # first update. A zero output projection also blocked the preceding
        # layer's gradient on the first optimization step.
        nn.init.xavier_uniform_(self.activation_head[-1].weight, gain=0.02)
        nn.init.constant_(
            self.activation_head[-1].bias,
            0.0,
        )
        self.temperature_raw = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _pair_kinematics(states: torch.Tensor) -> torch.Tensor:
        positions = states[:, :2]
        velocities = states[:, 2:4]
        delta_p = positions[:, None, :] - positions[None, :, :]
        delta_v = velocities[:, None, :] - velocities[None, :, :]

        eps = 1e-6
        distance = torch.linalg.vector_norm(delta_p, dim=-1)
        velocity_distance = torch.linalg.vector_norm(delta_v, dim=-1)
        relative_alignment = (delta_p * delta_v).sum(-1) / (
            distance * velocity_distance + eps
        )

        speed = torch.linalg.vector_norm(velocities, dim=-1)
        speed_difference = torch.abs(speed[:, None] - speed[None, :])
        velocity_cosine = (velocities[:, None, :] * velocities[None, :, :]).sum(-1) / (
            speed[:, None] * speed[None, :] + eps
        )
        radial_rate = (delta_p * delta_v).sum(-1) / (distance + eps)
        return torch.stack(
            [
                distance,
                velocity_distance,
                relative_alignment,
                speed_difference,
                velocity_cosine,
                radial_rate,
            ],
            dim=-1,
        )

    @staticmethod
    def _count_groups(adjacency: torch.Tensor, valid: torch.Tensor) -> int:
        active = torch.where(valid)[0].tolist()
        if not active:
            return 0
        binary = adjacency.detach() >= 0.5
        visited = set()
        groups = 0
        for start in active:
            if start in visited:
                continue
            groups += 1
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                neighbors = torch.where(binary[node] & valid)[0].tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        return groups

    def forward(
        self,
        batch: torch.Tensor,
        states: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        candidate_edge_index: torch.Tensor | None = None,
        candidate_edge_weight: torch.Tensor | None = None,
        selector_context: torch.Tensor | None = None,
    ) -> StructureOutput:
        if states.dim() == 3:
            states = states.mean(dim=1)
        if mask is None:
            valid_nodes = torch.ones(hidden.size(0), dtype=torch.bool, device=hidden.device)
        else:
            valid_nodes = mask.reshape(mask.size(0), -1).all(dim=-1).bool()

        projected = self.node_projection(hidden)
        if selector_context is None:
            selector_context = hidden.new_zeros(hidden.size(0), self.context_dim)
        elif selector_context.shape != (hidden.size(0), self.context_dim):
            raise ValueError(
                "selector_context must have shape "
                f"[{hidden.size(0)}, {self.context_dim}], got "
                f"{tuple(selector_context.shape)}"
            )
        structural_features = torch.zeros_like(projected)
        node_entropy = torch.ones(hidden.size(0), 1, device=hidden.device)
        activation = torch.zeros(hidden.size(0), 1, device=hidden.device)
        vector_parts = []
        vector_mask_parts = []
        edge_indices = []
        edge_weights = []
        total_groups = 0
        temperature = F.softplus(self.temperature_raw) + 0.25

        for graph_id in torch.unique(batch, sorted=True):
            node_indices = torch.where(batch == graph_id)[0]
            graph_hidden = projected[node_indices]
            graph_states = states[node_indices]
            graph_valid = valid_nodes[node_indices]
            node_count = node_indices.numel()

            hidden_sum = graph_hidden[:, None, :] + graph_hidden[None, :, :]
            hidden_difference = torch.abs(
                graph_hidden[:, None, :] - graph_hidden[None, :, :]
            )
            pair_features = torch.cat(
                [hidden_sum, hidden_difference, self._pair_kinematics(graph_states)],
                dim=-1,
            )
            logits = self.edge_network(pair_features).squeeze(-1) / temperature
            adjacency = torch.sigmoid(logits)

            pair_valid = graph_valid[:, None] & graph_valid[None, :]
            diagonal = torch.eye(node_count, dtype=torch.bool, device=hidden.device)
            pair_valid = pair_valid & ~diagonal
            adjacency = adjacency * pair_valid.to(adjacency.dtype)

            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            aggregated = adjacency @ graph_hidden / degree
            structural_features[node_indices] = self.struct_projection(aggregated)

            binary_entropy = -adjacency.clamp(1e-6, 1 - 1e-6) * torch.log(
                adjacency.clamp(1e-6, 1 - 1e-6)
            ) - (1 - adjacency).clamp(1e-6, 1 - 1e-6) * torch.log(
                (1 - adjacency).clamp(1e-6, 1 - 1e-6)
            )
            neighbor_count = pair_valid.sum(dim=-1, keepdim=True).clamp_min(1)
            entropy = (binary_entropy * pair_valid).sum(dim=-1, keepdim=True) / neighbor_count
            entropy = entropy / torch.log(torch.tensor(2.0, device=hidden.device))
            node_entropy[node_indices] = torch.where(
                graph_valid[:, None], entropy, torch.ones_like(entropy)
            )
            contrast = structural_features[node_indices] - graph_hidden
            activation_input = torch.cat(
                [
                    graph_hidden,
                    contrast,
                    node_entropy[node_indices],
                    graph_valid[:, None],
                    selector_context[node_indices],
                ],
                dim=-1,
            )
            activation[node_indices] = torch.sigmoid(self.activation_head(activation_input))

            vector_parts.append(_upper_triangle(adjacency))
            vector_mask_parts.append(_upper_triangle(pair_valid).bool())

            # SEAN is an interaction controller over the encoder's communication
            # graph, not a second all-to-all message-passing network.
            if candidate_edge_index is None:
                candidate = pair_valid
            else:
                candidate = torch.zeros_like(pair_valid)
                global_to_local = torch.full(
                    (hidden.size(0),), -1, dtype=torch.long, device=hidden.device
                )
                global_to_local[node_indices] = torch.arange(node_count, device=hidden.device)
                candidate_source, candidate_target = candidate_edge_index
                belongs = (batch[candidate_source] == graph_id) & (batch[candidate_target] == graph_id)
                local_source = global_to_local[candidate_source[belongs]]
                local_target = global_to_local[candidate_target[belongs]]
                usable = (local_source >= 0) & (local_target >= 0)
                candidate[local_source[usable], local_target[usable]] = True
                candidate &= pair_valid
            source, target = torch.where(candidate)
            if source.numel() > 0:
                edge_indices.append(
                    torch.stack([node_indices[source], node_indices[target]], dim=0)
                )
                # Preserve the causal geometric graph as a prior and let SEAN
                # learn only a bounded residual reweighting.  Replacing the
                # prior by a free sigmoid made small-sample expert training
                # suppress or overfit otherwise useful neighbour consensus.
                task_weight = 0.5 + adjacency[source, target]
                if candidate_edge_index is not None and candidate_edge_weight is not None:
                    # Candidate weights have already been projected to [0, 1].
                    # Look them up in their original order without allowing SEAN
                    # to introduce edges absent from the encoder graph.
                    lookup = torch.zeros_like(adjacency)
                    cs, ct = candidate_edge_index
                    belongs = (batch[cs] == graph_id) & (batch[ct] == graph_id)
                    ls, lt = global_to_local[cs[belongs]], global_to_local[ct[belongs]]
                    usable = (ls >= 0) & (lt >= 0)
                    lookup[ls[usable], lt[usable]] = candidate_edge_weight[belongs][usable].to(adjacency.dtype)
                    task_weight = task_weight * lookup[source, target]
                edge_weights.append(task_weight)
            total_groups += self._count_groups(adjacency, graph_valid)

        empty_vector = hidden.new_empty(0)
        empty_mask = torch.empty(0, dtype=torch.bool, device=hidden.device)
        adjacency_vector = torch.cat(vector_parts) if vector_parts else empty_vector
        vector_mask = torch.cat(vector_mask_parts) if vector_mask_parts else empty_mask
        if edge_indices:
            learned_edge_index = torch.cat(edge_indices, dim=1)
            learned_edge_weight = torch.cat(edge_weights)
        else:
            learned_edge_index = torch.empty(2, 0, dtype=torch.long, device=hidden.device)
            learned_edge_weight = empty_vector

        return StructureOutput(
            adjacency_vector,
            structural_features,
            vector_mask,
            total_groups,
            learned_edge_index,
            learned_edge_weight,
            node_entropy,
            activation,
        )


class JointGraphTemporalSpectralFilter(nn.Module):
    """Causal temporal FFT with optional task-driven spectral interaction.

    Temporal filtering is completed before graph interaction.  SEAN therefore
    controls a normalized graph-feature *increment* rather than raw edge
    magnitudes, whose global scaling would largely cancel during degree
    normalization.  In ABC mode, the previous scale latent conditions a new
    parameter candidate and a learned scalar gate applies
    ``p_k = p_{k-1} + z_k * (candidate_k - p_{k-1})``.  First/reborn frames
    bypass this recurrence and establish a fresh causal state.

    The parameterized ABC and GLP/Wiener MPSIF branches use ``fft_points``
    samples for a zero-padded one-sided FFT (64 by default), while the
    unconstrained learned ablation keeps its historical window-length transform.
    """

    def __init__(
        self,
        hidden_dim: int,
        windows: tuple[int, ...] = (8, 16),
        signal_dim: int = 2,
        chebyshev_order: int = 2,
        filter_mode: str = "abc",
        graph_mode: str = "graph",
        use_sean: bool = False,
        use_context_activation: bool = False,
        interaction_activation_floor: float = 0.0,
        innovation_decomposition: bool = False,
        innovation_whiten_clip: float = 20.0,
        recurrent_filter: bool = False,
        recurrent_gate_initial_update: float = 0.1,
        abc_c_limit: float | None = None,
        glp_p_min: float = 1.0,
        glp_p_max: float = 3.0,
        fft_points: int = 64,
    ):
        super().__init__()
        # Evaluation-only causal intervention. `None` uses the learned SEAN
        # activation, 0 removes only the graph increment, and 1 selects the
        # full interaction expert. It is intentionally not a Parameter and is
        # therefore absent from checkpoints and optimization.
        self.interaction_activation_override = None
        # Evaluation-only intervention for a fixed ABC recurrence coefficient.
        # None uses the learned parameter-update network; 1 exactly recovers
        # the feed-forward ABC response after every valid transition.
        self.parameter_update_gate_override = None
        # Evaluation-only calibration: alpha' = floor + (1-floor) * alpha.
        self.interaction_activation_floor_override = None
        # Evaluation-only stress-test multiplier applied after activation.
        # Unlike activation, this may exceed one so we can distinguish an
        # under-sensitive frozen fusion path from a wrongly directed expert.
        self.interaction_gain_override = None
        self.windows = tuple(int(window) for window in windows)
        self.chebyshev_order = int(chebyshev_order)
        self.signal_dim = signal_dim
        self.filter_mode = filter_mode
        self.use_sean = bool(use_sean)
        self.use_context_activation = bool(use_context_activation)
        self.innovation_decomposition = bool(innovation_decomposition)
        self.innovation_whiten_clip = float(innovation_whiten_clip)
        self.recurrent_filter = bool(recurrent_filter)
        # Evaluation may override these two attributes when loading an ABC
        # checkpoint produced before the 256-point, adaptive-DC response was
        # introduced. New training always uses the current definition.
        self.abc_zero_pad_fft = True
        self.abc_normalize_dc = False
        self.abc_c_limit = (
            None if abc_c_limit is None else float(abc_c_limit)
        )
        if self.abc_c_limit is not None and self.abc_c_limit <= 0.0:
            raise ValueError("ABC c limit must be positive when enabled")
        self.glp_p_min = float(glp_p_min)
        self.glp_p_max = float(glp_p_max)
        if not 0.0 < self.glp_p_min < self.glp_p_max:
            raise ValueError("GLP exponent bounds must satisfy 0 < min < max")
        self.fft_points = int(fft_points)
        if self.fft_points < max(self.windows) or self.fft_points < 2:
            raise ValueError(
                "GLP FFT points must be at least the largest causal window"
            )
        self.hidden_dim = int(hidden_dim)
        self.recurrent_gate_initial_update = float(
            recurrent_gate_initial_update
        )
        if not 0.0 < self.recurrent_gate_initial_update < 1.0:
            raise ValueError(
                "recurrent gate initial update must lie strictly inside (0, 1)"
            )
        self.interaction_activation_floor = float(interaction_activation_floor)
        if not 0.0 <= self.interaction_activation_floor <= 1.0:
            raise ValueError("interaction activation floor must be in [0, 1]")
        if graph_mode not in {"graph", "temporal_only"}:
            raise ValueError(f"Unsupported GASF graph mode: {graph_mode}")
        self.graph_mode = graph_mode
        self.last_filter_diagnostics = []
        self.last_effective_filter_diagnostics = {}
        self.last_context_activation_features = None
        self.last_innovation_diagnostics = {}
        self.last_innovation_details = {}
        if self.filter_mode not in {"abc", "glp_wiener", "learned"}:
            raise ValueError(f"Unsupported GASF filter mode: {self.filter_mode}")
        self.frequency_filters = nn.ModuleList()
        self.shared_frequency_filter = None
        if self.filter_mode == "glp_wiener":
            # Full, time-shift-invariant spectral descriptor per scale:
            # per-axis normalized power, total normalized power, (for xy)
            # real/imaginary cross-spectral coherence, log RMS, and the
            # per-axis energy share. Every rFFT bin is retained.
            per_bin_features = self.signal_dim + 1
            if self.signal_dim == 2:
                per_bin_features += 2
            descriptor_dim = sum(
                per_bin_features * (self.fft_points // 2 + 1)
                + 1
                + self.signal_dim
                for _window in self.windows
            ) + 2
            shared_hidden = max(8, hidden_dim)
            self.shared_frequency_filter = nn.Sequential(
                nn.LayerNorm(descriptor_dim),
                nn.Linear(descriptor_dim, shared_hidden),
                nn.SiLU(),
                nn.Linear(shared_hidden, max(4, hidden_dim // 2)),
                nn.SiLU(),
                nn.Linear(max(4, hidden_dim // 2), 3),
            )
            final = self.shared_frequency_filter[-1]
            nn.init.xavier_uniform_(final.weight, gain=0.02)
            initial_lambda = 0.5
            inverse_softplus = math.log(math.expm1(initial_lambda))
            with torch.no_grad():
                final.bias.copy_(torch.tensor([
                    0.0, inverse_softplus, 0.0,
                ], dtype=final.bias.dtype))
        else:
            self.frequency_filters.extend(
                [
                    nn.Sequential(
                        nn.Linear(2, hidden_dim // 2),
                        nn.SiLU(),
                        nn.Linear(hidden_dim // 2, self.chebyshev_order + 1),
                    )
                    for _ in self.windows
                ]
            )
        recurrent_hidden = max(1, hidden_dim // 2)
        # The recurrent state is deliberately restricted to the preceding ABC
        # parameters.  One scalar update coefficient changes the complete
        # response coherently from the current spectral descriptor, previous
        # ABC values, and current history validity.  No previous spectral
        # latent is stored or used by this experiment.
        recurrent_gate_input = 2 + (self.chebyshev_order + 1) + 1
        self.parameter_update_gates = nn.ModuleList()
        if self.filter_mode == "glp_wiener":
            descriptor_dim = self.shared_frequency_filter[0].normalized_shape[0]
            shared_parameter_update_gate = nn.Sequential(
                nn.Linear(descriptor_dim + 3 + 1, recurrent_hidden),
                nn.SiLU(),
                nn.Linear(recurrent_hidden, 1),
            )
            self.parameter_update_gates.append(
                shared_parameter_update_gate
            )
        else:
            self.parameter_update_gates.extend(
                [
                    nn.Sequential(
                        nn.Linear(recurrent_gate_input, recurrent_hidden),
                        nn.SiLU(),
                        nn.Linear(recurrent_hidden, 1),
                    )
                    for _ in self.windows
                ]
            )
        gate_bias = torch.logit(torch.tensor(self.recurrent_gate_initial_update))
        for gate_network in self.parameter_update_gates:
            nn.init.xavier_uniform_(gate_network[-1].weight, gain=0.05)
            with torch.no_grad():
                gate_network[-1].bias.fill_(float(gate_bias))
        self.temporal_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(signal_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.AdaptiveAvgPool1d(1),
                )
                for _ in self.windows
            ]
        )
        self.scale_scorers = nn.ModuleList(
            [nn.Linear(hidden_dim + 2, 1) for _ in self.windows]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # These two encoders see statistically distinct signals: group-common
        # innovation and target-specific innovation.  They deliberately reuse
        # the same monotone (a,b,c) frequency response as the Temporal branch,
        # so the ablation changes information content rather than filter type.
        self.group_innovation_encoders = nn.ModuleList()
        self.individual_innovation_encoders = nn.ModuleList()
        if self.innovation_decomposition:
            self.group_innovation_encoders.extend(
                [
                    nn.Sequential(
                        nn.Conv1d(signal_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.AdaptiveAvgPool1d(1),
                    )
                    for _ in self.windows
                ]
            )
            self.individual_innovation_encoders.extend(
                [
                    nn.Sequential(
                        nn.Conv1d(signal_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.AdaptiveAvgPool1d(1),
                    )
                    for _ in self.windows
                ]
            )
        self.SEAN = (
            TemporalProbabilisticSEAN(hidden_dim, hidden_dim)
            if self.use_sean else None
        )
        self.interaction_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Preserve the physically equivariant direction of the neighbour
        # innovation. Encoding each node independently before graph
        # propagation can discard this low-dimensional signal through
        # nonlinear pooling and LayerNorm. This branch remains feature-level:
        # it maps the SEAN-weighted causal innovation into the GASF latent and
        # never writes a state/acceleration correction directly.
        self.interaction_innovation_projection = nn.Sequential(
            nn.Linear(signal_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # A compact post-graph selector for the physical interaction path.
        # Inputs are causal rotation-invariant statistics, including explicit
        # agreement between the node innovation and SEAN neighbour consensus.
        self.context_activation_head = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.xavier_uniform_(self.context_activation_head[-1].weight, gain=0.02)
        nn.init.zeros_(self.context_activation_head[-1].bias)
        # Preserve a near-temporal initialization without making the graph
        # increment exactly zero (which would also zero the activation gradient).
        # Keep the graph increment residual-sized, but large enough that the
        # expert and temporal counterfactuals are numerically distinguishable.
        nn.init.xavier_uniform_(self.interaction_projection[-1].weight, gain=0.2)
        nn.init.zeros_(self.interaction_projection[-1].bias)
        # Old checkpoints therefore retain their exact prediction until the
        # isolated graph-context adaptation stage trains this new branch.
        nn.init.zeros_(self.interaction_innovation_projection[-1].weight)
        nn.init.zeros_(self.interaction_innovation_projection[-1].bias)
        self.innovation_uncertainty_encoder = None
        self.innovation_decomposition_projection = None
        if self.innovation_decomposition:
            self.innovation_uncertainty_encoder = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.innovation_decomposition_projection = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Exact baseline preservation is essential for a causal
            # contribution test: before adaptation, enabling this path changes
            # no prediction.
            nn.init.zeros_(self.innovation_decomposition_projection[-1].weight)
            nn.init.zeros_(self.innovation_decomposition_projection[-1].bias)

    @staticmethod
    def _empty_structure(reference: torch.Tensor) -> StructureOutput:
        empty = reference.new_empty(0)
        empty_mask = torch.empty(0, dtype=torch.bool, device=reference.device)
        empty_edges = torch.empty(2, 0, dtype=torch.long, device=reference.device)
        node_count = reference.size(0)
        return StructureOutput(
            empty,
            torch.zeros_like(reference),
            empty_mask,
            0,
            empty_edges,
            empty,
            torch.ones(node_count, 1, device=reference.device, dtype=reference.dtype),
            torch.zeros(node_count, 1, device=reference.device, dtype=reference.dtype),
        )

    @staticmethod
    def _graph_propagate(
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(features)
        source, target = edge_index
        node_count = features.size(0)
        degree = torch.zeros(node_count, device=features.device, dtype=features.dtype)
        degree.index_add_(0, target, edge_weight.to(features.dtype))
        norm = edge_weight.to(features.dtype) * degree[source].clamp_min(1e-6).rsqrt() * degree[
            target
        ].clamp_min(1e-6).rsqrt()
        output = torch.zeros_like(features)
        view_shape = (norm.size(0),) + (1,) * (features.dim() - 1)
        output.index_add_(0, target, features[source] * norm.view(view_shape))
        return output

    @staticmethod
    def _scale_sequence_and_spectrum(
        residual: torch.Tensor,
        window: int,
        fft_points: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the right-aligned causal window and its complex spectrum."""
        sequence = residual[:, -window:, :]
        if sequence.size(1) < window:
            sequence = F.pad(sequence, (0, 0, window - sequence.size(1), 0))
        sequence = torch.nan_to_num(sequence)
        n_fft = window if fft_points is None else int(fft_points)
        spectrum = torch.fft.rfft(sequence.transpose(1, 2), n=n_fft, dim=-1)
        return sequence, spectrum

    def _glp_gain(
        self,
        parameters: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth low-pass response gamma * exp(-lambda * f**p)."""
        gamma, attenuation, exponent = torch.split(parameters, 1, dim=-1)
        return gamma * torch.exp(
            -attenuation * frequencies.clamp_min(0.0).pow(exponent)
        )

    def _glp_shared_parameters(
        self,
        residual: torch.Tensor,
        node_entropy: torch.Tensor,
        valid_ratio: torch.Tensor,
        previous_filter_parameters: torch.Tensor,
        continuing_state: torch.Tensor,
    ):
        """Infer one response from the complete spectra of all time scales."""
        descriptor_parts = []
        prepared_scales = []
        for window in self.windows:
            sequence, spectrum = self._scale_sequence_and_spectrum(
                residual, window, self.fft_points
            )
            # Keep every rFFT bin but describe it through power and relative
            # cross-channel phase. An absolute complex phase changes under a
            # harmless shift of the causal time origin, whereas a spectral
            # response should be shift invariant. The relative cross-spectrum
            # retains directional coupling between the two innovation axes.
            rms = sequence.square().mean(
                dim=(1, 2), keepdim=True
            ).sqrt().clamp_min(1e-6)
            power = spectrum.abs().square()
            channel_energy = power.sum(dim=-1).clamp_min(1e-8)
            channel_distribution = power / channel_energy.unsqueeze(-1)
            total_power = power.sum(dim=1)
            total_distribution = total_power / total_power.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            descriptor_parts.append(
                channel_distribution.flatten(start_dim=1)
            )
            descriptor_parts.append(total_distribution)
            if self.signal_dim == 2:
                cross = spectrum[:, 0] * spectrum[:, 1].conj()
                cross_scale = (
                    power[:, 0] * power[:, 1]
                ).sqrt().clamp_min(1e-8)
                descriptor_parts.append(
                    torch.stack(
                        (cross.real / cross_scale, cross.imag / cross_scale),
                        dim=1,
                    ).flatten(start_dim=1)
                )
            descriptor_parts.append(torch.log1p(rms.flatten(start_dim=1)))
            descriptor_parts.append(
                channel_energy / channel_energy.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
            )
            prepared_scales.append((sequence, spectrum))
        descriptor = torch.cat(
            descriptor_parts + [node_entropy, valid_ratio], dim=-1
        )
        descriptor = torch.nan_to_num(descriptor)
        raw_parameters = self.shared_frequency_filter(descriptor)
        candidate_parameters = torch.cat(
            (
                torch.sigmoid(raw_parameters[:, 0:1]),
                F.softplus(raw_parameters[:, 1:2]).clamp_max(20.0),
                self.glp_p_min
                + (self.glp_p_max - self.glp_p_min)
                * torch.sigmoid(raw_parameters[:, 2:3]),
            ),
            dim=-1,
        )

        # GLP uses one physical response for all scales. Existing recurrent
        # state keeps a per-scale layout for checkpoint/API compatibility.
        previous_parameters = previous_filter_parameters.mean(dim=1)
        previous_valid = continuing_state.all(dim=1)
        previous_mask = previous_valid.to(residual.dtype)
        if self.recurrent_filter:
            if self.parameter_update_gate_override is None:
                proposed_gate = torch.sigmoid(
                    self.parameter_update_gates[0](torch.cat(
                        (
                            descriptor,
                            previous_parameters,
                            valid_ratio,
                        ),
                        dim=-1,
                    ))
                )
            else:
                override = float(self.parameter_update_gate_override)
                if not 0.0 <= override <= 1.0:
                    raise ValueError(
                        "parameter_update_gate_override must be in [0, 1]"
                    )
                proposed_gate = torch.full_like(previous_mask, override)
            update_gate = torch.where(
                previous_valid, proposed_gate, torch.ones_like(proposed_gate)
            )
            parameters = previous_parameters + update_gate * (
                candidate_parameters - previous_parameters
            )
        else:
            update_gate = torch.ones_like(previous_mask)
            parameters = candidate_parameters
        with torch.no_grad():
            parameter_step = torch.where(
                previous_valid,
                torch.linalg.vector_norm(
                    parameters - previous_parameters,
                    dim=-1,
                    keepdim=True,
                ),
                torch.zeros_like(update_gate),
            )
        return (
            parameters,
            descriptor,
            prepared_scales,
            update_gate,
            parameter_step,
            previous_parameters,
            previous_valid,
        )

    def _filter_glp_scale(
        self,
        sequence: torch.Tensor,
        spectrum: torch.Tensor,
        window: int,
        temporal_encoder: nn.Module,
        parameters: torch.Tensor,
        previous_parameters: torch.Tensor,
        previous_valid: torch.Tensor,
        valid_ratio: torch.Tensor,
        update_gate: torch.Tensor,
        parameter_step: torch.Tensor,
    ):
        frequencies = torch.linspace(
            0,
            1,
            spectrum.size(-1),
            device=spectrum.device,
            dtype=sequence.dtype,
        ).unsqueeze(0)
        gain = self._glp_gain(parameters, frequencies).clamp(0.0, 1.0)
        filtered_full = torch.fft.irfft(
            spectrum * gain.unsqueeze(1), n=self.fft_points, dim=-1
        )
        filtered_sequence = filtered_full[..., :window]
        temporal_latent = temporal_encoder(filtered_sequence).squeeze(-1)

        # A causal, label-free Wiener proxy anchors the learned response. The
        # high-frequency median estimates the white innovation floor; excess
        # power is treated as predictable low-frequency motion. cummin makes
        # the target compatible with the monotone GLP response. Positive bins
        # receive more weight so rare maneuvers cannot be ignored by an
        # all-zero solution on predominantly steady trajectories.
        power = spectrum.abs().square().sum(dim=1) / float(window * window)
        high_start = max(1, power.size(-1) // 2)
        noise_floor = power[:, high_start:].median(
            dim=-1, keepdim=True
        ).values
        wiener_target = (
            (power - noise_floor).clamp_min(0.0) / power.clamp_min(1e-8)
        )
        wiener_target = torch.cummin(wiener_target, dim=-1).values.detach()
        calibration_weight = 1.0 + 4.0 * wiener_target
        per_node_calibration = (
            calibration_weight * (gain - wiener_target).square()
        ).sum(dim=-1) / calibration_weight.sum(dim=-1).clamp_min(1e-6)
        valid_weight = valid_ratio[:, 0].clamp(0.0, 1.0)
        spectral_response_loss = (
            per_node_calibration * valid_weight
        ).sum() / valid_weight.sum().clamp_min(1.0)
        # Pointwise regression alone is minimized quickly by the population
        # average response. Match the *relative* retention across nodes in the
        # current causal batch as well, so a constant filter cannot satisfy the
        # calibration objective. This remains label-free: both sides are
        # computed from the current innovation spectrum.
        predicted_retention = gain.mean(dim=-1)
        target_retention = wiener_target.mean(dim=-1)
        valid_sum = valid_weight.sum().clamp_min(1.0)
        predicted_mean = (
            predicted_retention * valid_weight
        ).sum() / valid_sum
        target_mean = (target_retention * valid_weight).sum() / valid_sum
        predicted_centered = predicted_retention - predicted_mean
        target_centered = target_retention - target_mean
        target_variance = (
            target_centered.square() * valid_weight
        ).sum() / valid_sum
        centered_error = (
            (predicted_centered - target_centered).square() * valid_weight
        ).sum() / valid_sum
        spectral_adaptation_loss = torch.where(
            target_variance > 1e-6,
            centered_error / target_variance.clamp_min(1e-6),
            centered_error.new_zeros(()),
        )
        spectral_calibration_loss = (
            spectral_response_loss + spectral_adaptation_loss
        )
        with torch.no_grad():
            covariance = (
                predicted_centered * target_centered * valid_weight
            ).sum() / valid_sum
            predicted_variance = (
                predicted_centered.square() * valid_weight
            ).sum() / valid_sum
            response_target_correlation = covariance / torch.sqrt(
                predicted_variance * target_variance + 1e-8
            )

        with torch.no_grad():
            previous_gain = self._glp_gain(
                previous_parameters, frequencies
            ).clamp(0.0, 1.0)
            response_step = torch.where(
                previous_valid,
                (gain - previous_gain).square().mean(
                    dim=-1, keepdim=True
                ).sqrt(),
                torch.zeros_like(update_gate),
            )
        diagnostic_valid = valid_ratio[:, 0] > 0
        if not diagnostic_valid.any():
            diagnostic_valid = torch.ones_like(diagnostic_valid)
        gamma, attenuation, exponent = torch.split(parameters, 1, dim=-1)
        split = max(1, power.size(-1) // 2)
        low_energy = power[:, :split].mean(dim=-1, keepdim=True)
        high_energy = power[:, split:].mean(dim=-1, keepdim=True)
        diagnostics = {
            'lowpass_dc_gain': gain[diagnostic_valid, 0].detach().mean(),
            'lowpass_nyquist_gain': gain[
                diagnostic_valid, -1
            ].detach().mean(),
            'lowpass_mean_gain': gain[diagnostic_valid].detach().mean(),
            'low_high_energy_ratio': (
                low_energy / (low_energy + high_energy + 1e-8)
            )[diagnostic_valid].detach().mean(),
            'gamma': gamma[diagnostic_valid].detach().mean(),
            'lambda': attenuation[diagnostic_valid].detach().mean(),
            'p': exponent[diagnostic_valid].detach().mean(),
            'wiener_target_mean': wiener_target[
                diagnostic_valid
            ].mean(),
            'spectral_response_loss': spectral_response_loss.detach(),
            'spectral_adaptation_loss': spectral_adaptation_loss.detach(),
            'response_target_correlation': (
                response_target_correlation.detach()
            ),
            'spectral_calibration_loss': spectral_calibration_loss.detach(),
            'allpass_fraction_099': (
                gain[diagnostic_valid, -1].detach() >= 0.99
            ).to(gain.dtype).mean(),
            'c_saturation_fraction': gain.new_tensor(float('nan')),
            '_nyquist_gain_per_node': gain[:, -1].detach(),
            'parameter_update_gate': update_gate.detach().mean(),
            'parameter_step_norm': parameter_step.detach().mean(),
            'filter_response_step_rms': response_step.detach().mean(),
        }
        return (
            temporal_latent,
            diagnostics,
            spectral_calibration_loss,
            response_step,
        )

    def _filter_scale(
        self,
        residual: torch.Tensor,
        window: int,
        filter_network: nn.Module,
        temporal_encoder: nn.Module,
        node_entropy: torch.Tensor,
        valid_ratio: torch.Tensor,
        update_gate_network: nn.Module,
        previous_parameters: torch.Tensor,
        previous_valid: torch.Tensor,
    ):
        sequence = residual[:, -window:, :]
        if sequence.size(1) < window:
            sequence = F.pad(sequence, (0, 0, window - sequence.size(1), 0))
        sequence = torch.nan_to_num(sequence)

        # Both parameterized MPSIF variants use the configured zero-padded
        # spectrum.  The unconstrained learned legacy branch retains its
        # historical window-length transform for ablation compatibility.
        if self.filter_mode == "learned":
            n_fft = window
        elif self.filter_mode == "abc" and not self.abc_zero_pad_fft:
            n_fft = window
        else:
            n_fft = self.fft_points
        spectrum = torch.fft.rfft(sequence.transpose(1, 2), n=n_fft, dim=-1)
        if self.filter_mode == "learned":
            spectral_features = torch.view_as_real(spectrum).permute(0, 2, 1, 3)
            filtered_complex = torch.view_as_complex(
                spectral_features.permute(0, 2, 1, 3).contiguous()
            )
            filtered_sequence = torch.fft.irfft(filtered_complex, n=window, dim=-1)
            temporal_latent = temporal_encoder(filtered_sequence).squeeze(-1)
            return temporal_latent, {
                'lowpass_dc_gain': residual.new_tensor(float('nan')),
                'lowpass_nyquist_gain': residual.new_tensor(float('nan')),
                'lowpass_mean_gain': residual.new_tensor(float('nan')),
                'low_high_energy_ratio': residual.new_tensor(float('nan')),
                'parameter_update_gate': residual.new_tensor(1.0),
                'parameter_step_norm': residual.new_tensor(0.0),
                'filter_response_step_rms': residual.new_tensor(float('nan')),
                '_nyquist_gain_per_node': residual.new_full(
                    (residual.size(0),), float('nan')
                ),
            }, residual.new_zeros(residual.size(0), 3), residual.new_ones(
                residual.size(0), 1
            ), residual.new_zeros(residual.size(0), 1), residual.new_zeros(
                residual.size(0), 1
            )

        magnitude = spectrum.abs().mean(dim=1)
        split = max(1, magnitude.size(-1) // 2)
        low_energy = magnitude[:, :split].square().mean(dim=-1, keepdim=True)
        high_energy = magnitude[:, split:].square().mean(dim=-1, keepdim=True)
        energy_ratio = low_energy / (low_energy + high_energy + 1e-8)

        # A compact, smooth and guaranteed monotone low-pass response.  The
        # existing three-output network is reinterpreted as (a, b, c), keeping
        # old checkpoint tensor shapes compatible while removing independent
        # per-frequency degrees of freedom.
        spectral_descriptor = torch.cat([energy_ratio, node_entropy], dim=-1)
        previous_mask = previous_valid.to(residual.dtype)
        candidate_raw_parameters = filter_network(spectral_descriptor)
        candidate_c = candidate_raw_parameters[:, 2:3]
        if self.abc_c_limit is not None:
            # A soft bound prevents c from driving both sigmoids into the
            # saturated all-pass regime while preserving unit slope around
            # zero.  None deliberately retains exact legacy behavior.
            limit = self.abc_c_limit
            candidate_c = limit * torch.tanh(candidate_c / limit)
        candidate_parameters = torch.cat(
            (
                -F.softplus(candidate_raw_parameters[:, 0:1]),
                -F.softplus(candidate_raw_parameters[:, 1:2]),
                candidate_c,
            ),
            dim=-1,
        )
        if self.recurrent_filter:
            if self.parameter_update_gate_override is None:
                proposed_gate = torch.sigmoid(update_gate_network(torch.cat(
                    (
                        spectral_descriptor,
                        previous_parameters,
                        valid_ratio,
                    ),
                    dim=-1,
                )))
            else:
                override = float(self.parameter_update_gate_override)
                if not 0.0 <= override <= 1.0:
                    raise ValueError(
                        "parameter_update_gate_override must be in [0, 1]"
                    )
                proposed_gate = torch.full_like(previous_mask, override)
            # With no causal predecessor, bypass the gate so the first valid
            # frame is identical to the original feed-forward ABC filter.
            parameter_update_gate = torch.where(
                previous_valid, proposed_gate, torch.ones_like(proposed_gate)
            )
            parameters = previous_parameters + parameter_update_gate * (
                candidate_parameters - previous_parameters
            )
        else:
            parameter_update_gate = torch.ones_like(previous_mask)
            parameters = candidate_parameters
        a, b, c = torch.split(parameters, 1, dim=-1)
        frequencies = torch.linspace(
            0, 1, spectrum.size(-1), device=residual.device, dtype=residual.dtype
        ).unsqueeze(0)
        gain = torch.sigmoid(a * frequencies.square() + b * frequencies + c)
        if self.abc_normalize_dc:
            gain = (gain / torch.sigmoid(c).clamp_min(1e-4)).clamp(0.0, 1.0)
        # One real-valued response multiplies the complete complex coefficient
        # for both measurement axes, preserving spectral phase.  The DC gain
        # remains adaptive through c instead of being normalized to one.
        filtered_full = torch.fft.irfft(
            spectrum * gain.unsqueeze(1), n=n_fft, dim=-1
        )
        filtered_sequence = filtered_full[..., :window]
        temporal_latent = temporal_encoder(filtered_sequence).squeeze(-1)
        with torch.no_grad():
            previous_a, previous_b, previous_c = torch.split(
                previous_parameters, 1, dim=-1
            )
            previous_gain = torch.sigmoid(
                previous_a * frequencies.square()
                + previous_b * frequencies
                + previous_c
            )
            if self.abc_normalize_dc:
                previous_gain = (
                    previous_gain / torch.sigmoid(previous_c).clamp_min(1e-4)
                ).clamp(0.0, 1.0)
            response_step = (gain - previous_gain).square().mean(
                dim=-1, keepdim=True
            ).sqrt()
            response_step = torch.where(
                previous_valid, response_step, torch.zeros_like(response_step)
            )
            parameter_step = torch.where(
                previous_valid,
                torch.linalg.vector_norm(
                    parameters - previous_parameters, dim=-1, keepdim=True
                ),
                torch.zeros_like(response_step),
            )
        diagnostic_valid = valid_ratio[:, 0] > 0
        if not diagnostic_valid.any():
            diagnostic_valid = torch.ones_like(diagnostic_valid)
        diagnostics = {
            'lowpass_dc_gain': gain[diagnostic_valid, 0].detach().mean(),
            'lowpass_nyquist_gain': gain[
                diagnostic_valid, -1
            ].detach().mean(),
            'lowpass_mean_gain': gain[diagnostic_valid].detach().mean(),
            'low_high_energy_ratio': energy_ratio[
                diagnostic_valid
            ].detach().mean(),
            'a': a[diagnostic_valid].detach().mean(),
            'b': b[diagnostic_valid].detach().mean(),
            'c': c[diagnostic_valid].detach().mean(),
            'allpass_fraction_099': (
                gain[diagnostic_valid, -1].detach() >= 0.99
            ).to(gain.dtype).mean(),
            'c_saturation_fraction': (
                c[diagnostic_valid].detach().abs() >= 4.0
            ).to(gain.dtype).mean(),
            '_nyquist_gain_per_node': gain[:, -1].detach(),
            'parameter_update_gate': parameter_update_gate.detach().mean(),
            'parameter_step_norm': parameter_step.detach().mean(),
            'filter_response_step_rms': response_step.detach().mean(),
        }
        return (
            temporal_latent,
            diagnostics,
            parameters,
            parameter_update_gate,
            parameter_step,
            response_step,
        )

    def _encode_with_filter_parameters(
        self,
        residual: torch.Tensor,
        window: int,
        temporal_encoder: nn.Module,
        parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Encode another signal with the already selected scale response."""
        sequence = residual[:, -window:, :]
        if sequence.size(1) < window:
            sequence = F.pad(sequence, (0, 0, window - sequence.size(1), 0))
        sequence = torch.nan_to_num(sequence)
        if self.filter_mode == "learned":
            n_fft = window
        elif self.filter_mode == "abc" and not self.abc_zero_pad_fft:
            n_fft = window
        else:
            n_fft = self.fft_points
        spectrum = torch.fft.rfft(sequence.transpose(1, 2), n=n_fft, dim=-1)
        if self.filter_mode == "abc":
            a, b, c = torch.split(parameters, 1, dim=-1)
            frequencies = torch.linspace(
                0,
                1,
                spectrum.size(-1),
                device=residual.device,
                dtype=residual.dtype,
            ).unsqueeze(0)
            gain = torch.sigmoid(
                a * frequencies.square() + b * frequencies + c
            )
            if self.abc_normalize_dc:
                gain = (
                    gain / torch.sigmoid(c).clamp_min(1e-4)
                ).clamp(0.0, 1.0)
            spectrum = spectrum * gain.unsqueeze(1)
        elif self.filter_mode == "glp_wiener":
            frequencies = torch.linspace(
                0,
                1,
                spectrum.size(-1),
                device=residual.device,
                dtype=residual.dtype,
            ).unsqueeze(0)
            gain = self._glp_gain(parameters, frequencies).clamp(0.0, 1.0)
            spectrum = spectrum * gain.unsqueeze(1)
        filtered_full = torch.fft.irfft(spectrum, n=n_fft, dim=-1)
        filtered_sequence = filtered_full[..., :window]
        return temporal_encoder(filtered_sequence).squeeze(-1)

    def _decompose_innovation(
        self,
        raw: torch.Tensor,
        covariance: torch.Tensor,
        valid: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        """Information-weighted soft-graph common/individual decomposition."""
        valid = valid.squeeze(-1).bool()
        covariance = covariance.detach()
        dimension = raw.size(-1)
        identity = torch.eye(
            dimension, dtype=raw.dtype, device=raw.device
        ).view(1, 1, dimension, dimension)
        symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        scale = eigenvalues.sum(dim=-1, keepdim=True).abs() / dimension
        eigenvalues = eigenvalues.clamp_min((1e-6 * scale).clamp_min(1e-6))
        safe_covariance = (
            eigenvectors @ torch.diag_embed(eigenvalues)
            @ eigenvectors.transpose(-1, -2)
        )
        precision = (
            eigenvectors @ torch.diag_embed(eigenvalues.reciprocal())
            @ eigenvectors.transpose(-1, -2)
        )
        precision = torch.where(
            valid[..., None, None], precision, torch.zeros_like(precision)
        )

        information = torch.zeros_like(precision)
        information_vector = torch.zeros_like(raw)
        neighbor_weight = raw.new_zeros(raw.shape[:-1] + (1,))
        if edge_index.numel() > 0:
            source, target = edge_index
            weight = edge_weight.to(raw.dtype).view(-1, 1, 1)
            edge_valid = valid[source] & valid[target]
            effective_weight = weight * edge_valid.unsqueeze(-1).to(raw.dtype)
            weighted_precision = precision[source] * effective_weight.unsqueeze(-1)
            weighted_vector = (
                precision[source] @ raw[source].unsqueeze(-1)
            ).squeeze(-1) * effective_weight
            information.index_add_(0, target, weighted_precision)
            information_vector.index_add_(0, target, weighted_vector)
            neighbor_weight.index_add_(0, target, effective_weight)

        has_neighbors = (neighbor_weight > 1e-8) & valid.unsqueeze(-1)
        # A common-mode estimate must include the target's own innovation as
        # well as its SEAN neighbours.  Using neighbours alone degenerates to
        # copying the other target in a two-node group and unnecessarily
        # doubles the individual-noise variance.  Isolated nodes still use the
        # explicit fallback below rather than re-labelling their own signal as
        # a group-common component.
        common_information = information + torch.where(
            has_neighbors.unsqueeze(-1), precision, torch.zeros_like(precision)
        )
        own_information_vector = (
            precision @ raw.unsqueeze(-1)
        ).squeeze(-1)
        common_information_vector = information_vector + torch.where(
            has_neighbors, own_information_vector, torch.zeros_like(own_information_vector)
        )
        regularized_information = common_information + 1e-6 * identity
        group_covariance = torch.linalg.inv(regularized_information)
        group_raw = (
            group_covariance @ common_information_vector.unsqueeze(-1)
        ).squeeze(-1)
        group_raw = torch.where(has_neighbors, group_raw, torch.zeros_like(group_raw))
        group_covariance = torch.where(
            has_neighbors.unsqueeze(-1), group_covariance, safe_covariance
        )
        individual_raw = torch.where(
            valid.unsqueeze(-1), raw - group_raw, torch.zeros_like(raw)
        )
        group_white, group_nis, _ = CovarianceUtils.whiten_innovation(
            group_raw,
            group_covariance,
            valid_mask=has_neighbors,
            max_norm=self.innovation_whiten_clip,
        )
        individual_white, individual_nis, _ = CovarianceUtils.whiten_innovation(
            individual_raw,
            safe_covariance + torch.where(
                has_neighbors.unsqueeze(-1), group_covariance, torch.zeros_like(group_covariance)
            ),
            valid_mask=valid.unsqueeze(-1),
            max_norm=self.innovation_whiten_clip,
        )
        return (
            group_raw,
            individual_raw,
            group_white,
            individual_white,
            group_nis,
            individual_nis,
            has_neighbors,
        )

    def _innovation_increment(
        self,
        bundle: dict,
        structure: StructureOutput,
        scale_weights: torch.Tensor,
        filter_parameters: torch.Tensor,
    ):
        raw = torch.nan_to_num(bundle['raw'])
        covariance = torch.nan_to_num(bundle['covariance'])
        valid = bundle['valid'].bool()
        valid_2d = valid.squeeze(-1)
        own_nis = torch.nan_to_num(bundle['nis']).clamp_min(0.0)
        (
            group_raw,
            individual_raw,
            group_white,
            individual_white,
            group_nis,
            individual_nis,
            has_neighbors,
        ) = self._decompose_innovation(
            raw,
            covariance,
            valid,
            structure.edge_index,
            structure.edge_weight,
        )

        group_latents = []
        individual_latents = []
        for index, window in enumerate(self.windows):
            group_latent = self._encode_with_filter_parameters(
                group_white,
                window,
                self.group_innovation_encoders[index],
                filter_parameters[:, index],
            )
            individual_latent = self._encode_with_filter_parameters(
                individual_white,
                window,
                self.individual_innovation_encoders[index],
                filter_parameters[:, index],
            )
            group_latents.append(group_latent)
            individual_latents.append(individual_latent)
        group_latent = (
            torch.stack(group_latents, dim=1) * scale_weights.unsqueeze(-1)
        ).sum(dim=1)
        individual_latent = (
            torch.stack(individual_latents, dim=1) * scale_weights.unsqueeze(-1)
        ).sum(dim=1)

        valid_float = valid_2d.to(raw.dtype)
        count = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        own_mean = (own_nis.squeeze(-1) * valid_float).sum(dim=1, keepdim=True) / count
        own_masked = own_nis.squeeze(-1).masked_fill(~valid_2d, 0.0)
        own_max = own_masked.amax(dim=1, keepdim=True)
        # Online buffers are right-aligned; the last slot is always the most
        # recent causal frame (and is zero when that frame is invalid).
        own_latest = own_nis[:, -1]
        group_mean = (group_nis.squeeze(-1) * valid_float).sum(dim=1, keepdim=True) / count
        individual_mean = (
            individual_nis.squeeze(-1) * valid_float
        ).sum(dim=1, keepdim=True) / count
        latest_covariance = covariance[:, -1]
        latest_eigenvalues = torch.linalg.eigvalsh(
            0.5 * (latest_covariance + latest_covariance.transpose(-1, -2))
        ).clamp_min(1e-6)
        logdet = torch.log(latest_eigenvalues).sum(dim=-1, keepdim=True)
        anisotropy = torch.log(
            latest_eigenvalues[:, -1:] / latest_eigenvalues[:, :1]
        )
        uncertainty_statistics = torch.cat(
            (
                torch.log1p(own_mean),
                torch.log1p(own_max),
                torch.log1p(own_latest),
                torch.log1p(group_mean),
                torch.log1p(individual_mean),
                logdet,
                anisotropy,
                valid_float.mean(dim=1, keepdim=True),
            ),
            dim=-1,
        )
        uncertainty_latent = self.innovation_uncertainty_encoder(
            torch.nan_to_num(uncertainty_statistics)
        )
        increment = self.innovation_decomposition_projection(
            torch.cat((group_latent, individual_latent, uncertainty_latent), dim=-1)
        )
        latest_valid = valid_2d[:, -1:].to(increment.dtype)
        increment = increment * latest_valid
        self.last_innovation_diagnostics = {
            'reconstruction_max_error': (
                group_raw + individual_raw - raw
            )[valid_2d].abs().max().detach() if valid_2d.any() else raw.new_tensor(0.0),
            'group_nis_mean': group_mean.detach().mean(),
            'individual_nis_mean': individual_mean.detach().mean(),
            'neighbor_fraction': has_neighbors.float().detach().mean(),
            'increment_norm_mean': increment.detach().norm(dim=-1).mean(),
        }
        self.last_innovation_details = {
            # These are diagnostics only. Detaching prevents an evaluator from
            # accidentally retaining a complete rollout autograd graph.
            'valid_latest': valid_2d[:, -1].detach(),
            'has_neighbors_latest': has_neighbors[:, -1, 0].detach(),
            'own_raw_latest': raw[:, -1].detach(),
            'group_raw_latest': group_raw[:, -1].detach(),
            'individual_raw_latest': individual_raw[:, -1].detach(),
            'own_white_latest': bundle['whitened'][:, -1].detach(),
            'group_white_latest': group_white[:, -1].detach(),
            'individual_white_latest': individual_white[:, -1].detach(),
            'own_nis_latest': own_nis[:, -1, 0].detach(),
            'group_nis_latest': group_nis[:, -1, 0].detach(),
            'individual_nis_latest': individual_nis[:, -1, 0].detach(),
            'edge_index': structure.edge_index.detach(),
            'edge_weight': structure.edge_weight.detach(),
        }
        return increment

    def forward(
        self,
        residual: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_entropy: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        innovation_bundle: dict | None = None,
        recurrent_state: SpectralFilterState | None = None,
        recurrent_valid_mask: torch.Tensor | None = None,
    ) -> SpectralOutput:
        self.last_context_activation_features = None
        self.last_innovation_details = {}
        if valid_mask is None:
            valid_ratio = torch.ones(residual.size(0), 1, device=residual.device)
        else:
            mask = valid_mask.reshape(valid_mask.size(0), -1).float()
            valid_ratio = mask.mean(dim=-1, keepdim=True)

        if node_entropy is None:
            node_entropy = torch.ones(
                residual.size(0), 1, device=residual.device, dtype=residual.dtype
            )

        node_count = residual.size(0)
        scale_count = len(self.windows)
        if recurrent_valid_mask is None:
            if valid_mask is None:
                current_recurrent_valid = torch.ones(
                    node_count, 1, dtype=torch.bool, device=residual.device
                )
            else:
                flattened_valid = valid_mask.reshape(node_count, -1).bool()
                current_recurrent_valid = flattened_valid[:, -1:]
        else:
            current_recurrent_valid = recurrent_valid_mask.reshape(
                node_count, -1
            )[:, -1:].bool().to(residual.device)

        if recurrent_state is None:
            previous_filter_parameters = residual.new_zeros(
                node_count, scale_count, self.chebyshev_order + 1
            )
            previous_state_valid = torch.zeros(
                node_count,
                scale_count,
                1,
                dtype=torch.bool,
                device=residual.device,
            )
        else:
            expected_parameter_shape = (
                node_count, scale_count, self.chebyshev_order + 1
            )
            expected_valid_shape = (node_count, scale_count, 1)
            if tuple(recurrent_state.filter_parameters.shape) != expected_parameter_shape:
                raise ValueError(
                    "GASF recurrent parameter shape mismatch: "
                    f"{tuple(recurrent_state.filter_parameters.shape)} != "
                    f"{expected_parameter_shape}"
                )
            if tuple(recurrent_state.valid.shape) != expected_valid_shape:
                raise ValueError(
                    "GASF recurrent validity shape mismatch: "
                    f"{tuple(recurrent_state.valid.shape)} != "
                    f"{expected_valid_shape}"
                )
            previous_filter_parameters = recurrent_state.filter_parameters.to(
                residual
            )
            previous_state_valid = recurrent_state.valid.to(
                device=residual.device, dtype=torch.bool
            )
        continuing_state = previous_state_valid & current_recurrent_valid.unsqueeze(1)

        latents = []
        scores = []
        confidence_features = torch.cat([node_entropy, valid_ratio], dim=-1)
        filter_diagnostics = []
        filter_parameters = []
        parameter_update_gates = []
        parameter_step_norms = []
        filter_response_steps = []
        if self.filter_mode == "glp_wiener":
            (
                shared_parameters,
                _,
                prepared_scales,
                shared_update_gate,
                shared_parameter_step,
                previous_shared_parameters,
                previous_shared_valid,
            ) = self._glp_shared_parameters(
                residual,
                node_entropy,
                valid_ratio,
                previous_filter_parameters,
                continuing_state,
            )
            spectral_calibration_losses = []
            for index, (
                window,
                prepared,
                temporal_encoder,
                scorer,
            ) in enumerate(zip(
                self.windows,
                prepared_scales,
                self.temporal_encoders,
                self.scale_scorers,
            )):
                sequence, spectrum = prepared
                (
                    latent,
                    diagnostics,
                    calibration_loss,
                    response_step,
                ) = self._filter_glp_scale(
                    sequence,
                    spectrum,
                    window,
                    temporal_encoder,
                    shared_parameters,
                    previous_shared_parameters,
                    previous_shared_valid,
                    valid_ratio,
                    shared_update_gate,
                    shared_parameter_step,
                )
                filter_diagnostics.append(diagnostics)
                latents.append(latent)
                filter_parameters.append(shared_parameters)
                parameter_update_gates.append(shared_update_gate)
                parameter_step_norms.append(shared_parameter_step)
                filter_response_steps.append(response_step)
                spectral_calibration_losses.append(calibration_loss)
                scores.append(scorer(torch.cat(
                    [latent, confidence_features], dim=-1
                )))
            spectral_calibration_loss = torch.stack(
                spectral_calibration_losses
            ).mean()
        else:
            spectral_calibration_loss = residual.new_tensor(0.0)
            for index, (
                window,
                filter_network,
                temporal_encoder,
                scorer,
                update_gate_network,
            ) in enumerate(zip(
                self.windows,
                self.frequency_filters,
                self.temporal_encoders,
                self.scale_scorers,
                self.parameter_update_gates,
            )):
                (
                    latent,
                    diagnostics,
                    parameters,
                    update_gate,
                    parameter_step,
                    response_step,
                ) = self._filter_scale(
                    residual,
                    window,
                    filter_network,
                    temporal_encoder,
                    node_entropy,
                    valid_ratio,
                    update_gate_network,
                    previous_filter_parameters[:, index],
                    continuing_state[:, index],
                )
                filter_diagnostics.append(diagnostics)
                latents.append(latent)
                filter_parameters.append(parameters)
                parameter_update_gates.append(update_gate)
                parameter_step_norms.append(parameter_step)
                filter_response_steps.append(response_step)
                scores.append(scorer(torch.cat(
                    [latent, confidence_features], dim=-1
                )))

        stacked_latents = torch.stack(latents, dim=1)
        stacked_filter_parameters = torch.stack(filter_parameters, dim=1)
        stacked_update_gates = torch.cat(parameter_update_gates, dim=-1)
        stacked_parameter_steps = torch.cat(parameter_step_norms, dim=-1)
        stacked_response_steps = torch.cat(filter_response_steps, dim=-1)
        recurrent_valid = current_recurrent_valid.unsqueeze(1).expand(
            -1, scale_count, -1
        )
        recurrent_state = SpectralFilterState(
            torch.where(
                recurrent_valid,
                stacked_filter_parameters,
                torch.zeros_like(stacked_filter_parameters),
            ),
            recurrent_valid,
        )
        scale_weights = torch.softmax(torch.cat(scores, dim=-1), dim=-1)
        nyquist_per_scale = torch.stack(
            [
                diagnostic['_nyquist_gain_per_node']
                for diagnostic in filter_diagnostics
            ],
            dim=-1,
        )
        diagnostic_current_valid = current_recurrent_valid[:, 0]
        if not diagnostic_current_valid.any():
            diagnostic_current_valid = torch.ones_like(diagnostic_current_valid)
        for index, diagnostic in enumerate(filter_diagnostics):
            weight = scale_weights[:, index].detach()
            valid_weight = weight[diagnostic_current_valid]
            diagnostic['scale_weight_mean'] = valid_weight.mean()
            diagnostic['scale_weight_std'] = valid_weight.std(unbiased=False)
            diagnostic['effective_nyquist_contribution'] = (
                weight[diagnostic_current_valid]
                * nyquist_per_scale[diagnostic_current_valid, index]
            ).mean()
            diagnostic.pop('_nyquist_gain_per_node')
        self.last_effective_filter_diagnostics = {
            'effective_nyquist_gain': (
                scale_weights.detach()[diagnostic_current_valid]
                * nyquist_per_scale[diagnostic_current_valid]
            ).sum(dim=-1).mean(),
            'nyquist_scale_diversity': nyquist_per_scale[
                diagnostic_current_valid
            ].std(
                dim=-1, unbiased=False
            ).mean(),
        }
        temporal_fused = (stacked_latents * scale_weights.unsqueeze(-1)).sum(dim=1)
        temporal_fused = self.output_projection(temporal_fused)

        structure = self._empty_structure(temporal_fused)
        interaction_delta = torch.zeros_like(temporal_fused)
        innovation_context = temporal_fused.new_zeros(
            temporal_fused.size(0), self.signal_dim + 1
        )
        fused = temporal_fused
        if self.graph_mode == "graph" and self.use_sean:
            if batch is None or states is None:
                raise ValueError("SEAN-enabled GASF requires batch and causal states")
            # A node is eligible as soon as it exists at the latest causal
            # input frame. Requiring an entirely valid history unnecessarily
            # suppresses newly born but predictable targets.
            latest_valid = (
                valid_mask[:, -1:, :]
                if valid_mask is not None else None
            )
            residual_norm = torch.linalg.vector_norm(residual, dim=-1)
            residual_delta = residual[:, 1:] - residual[:, :-1]
            delta_norm = torch.linalg.vector_norm(residual_delta, dim=-1)
            # Explicit causal maneuver evidence. The normalized spectral
            # latent passes through LayerNorm, so it cannot reliably retain
            # absolute innovation magnitude on its own.
            selector_context = torch.stack(
                (
                    torch.log1p(residual_norm.square().mean(dim=-1).sqrt()),
                    torch.log1p(residual_norm.amax(dim=-1)),
                    torch.log1p(delta_norm.square().mean(dim=-1).sqrt()),
                    torch.log1p(residual_norm[:, -1]),
                ),
                dim=-1,
            )
            structure = self.SEAN(
                batch,
                states,
                temporal_fused,
                latest_valid,
                edge_index,
                edge_weight,
                selector_context,
            )
            graph_latent = self._graph_propagate(
                temporal_fused,
                structure.edge_index,
                structure.edge_weight,
            )
            contrast = graph_latent - temporal_fused
            latest_innovation = residual[:, -1]
            graph_innovation = self._graph_propagate(
                latest_innovation,
                structure.edge_index,
                structure.edge_weight,
            )
            innovation_context = torch.cat(
                (
                    graph_innovation,
                    torch.linalg.vector_norm(
                        graph_innovation, dim=-1, keepdim=True
                    ),
                ),
                dim=-1,
            )
            if self.innovation_decomposition:
                if innovation_bundle is None:
                    raise ValueError(
                        "innovation_decomposition requires online innovation histories"
                    )
                interaction_delta = self._innovation_increment(
                    innovation_bundle,
                    structure,
                    scale_weights,
                    stacked_filter_parameters,
                )
                # There is no learned confidence gate in this formulation.
                # SEAN determines the soft graph; its evidence is fused as a
                # zero-initialized residual feature once that graph exists.
                structure = structure._replace(
                    activation=torch.ones_like(structure.activation)
                )
            else:
                interaction_delta = self.interaction_projection(torch.cat(
                    (temporal_fused, graph_latent, contrast), dim=-1
                ))
            if self.use_context_activation and not self.innovation_decomposition:
                own_norm = torch.linalg.vector_norm(
                    latest_innovation, dim=-1, keepdim=True
                )
                graph_norm = torch.linalg.vector_norm(
                    graph_innovation, dim=-1, keepdim=True
                )
                agreement = F.cosine_similarity(
                    latest_innovation,
                    graph_innovation,
                    dim=-1,
                    eps=1e-8,
                ).unsqueeze(-1)
                contrast_norm = torch.linalg.vector_norm(
                    graph_innovation - latest_innovation,
                    dim=-1,
                    keepdim=True,
                )
                activation_features = torch.cat(
                    (
                        selector_context,
                        torch.log1p(own_norm),
                        torch.log1p(graph_norm),
                        agreement,
                        torch.log1p(contrast_norm),
                    ),
                    dim=-1,
                )
                self.last_context_activation_features = (
                    activation_features.detach()
                )
                structure = structure._replace(
                    activation=torch.sigmoid(
                        self.context_activation_head(activation_features)
                    )
                )
            if not self.innovation_decomposition:
                interaction_delta = interaction_delta + (
                    self.interaction_innovation_projection(innovation_context)
                )
            activation = structure.activation
            if self.interaction_activation_override is not None:
                override = float(self.interaction_activation_override)
                if not 0.0 <= override <= 1.0:
                    raise ValueError(
                        "interaction_activation_override must be in [0, 1]"
                    )
                activation = torch.full_like(activation, override)
            floor = (
                float(self.interaction_activation_floor_override)
                if self.interaction_activation_floor_override is not None
                else (
                    0.0
                    if self.interaction_activation_override is not None
                    else self.interaction_activation_floor
                )
            )
            if floor != 0.0:
                if not 0.0 <= floor <= 1.0:
                    raise ValueError(
                        "interaction_activation_floor_override must be in [0, 1]"
                    )
                activation = floor + (1.0 - floor) * activation
            gain = 1.0
            if self.interaction_gain_override is not None:
                gain = float(self.interaction_gain_override)
                if not torch.isfinite(torch.tensor(gain)) or gain < 0.0:
                    raise ValueError(
                        "interaction_gain_override must be finite and non-negative"
                    )
            effective_activation = activation * gain
            structure = structure._replace(activation=effective_activation)
            fused = temporal_fused + effective_activation * interaction_delta
        elif self.graph_mode == "graph":
            # Compatibility path for the existing no-SEAN graph ablation.
            graph_latent = self._graph_propagate(
                temporal_fused, edge_index, edge_weight
            )
            fused = 0.5 * (temporal_fused + graph_latent)

        # In direct-concatenation mode this value is diagnostic only.  Use the
        # actual interaction activation so its meaning remains auditable.
        confidence = structure.activation
        self.last_filter_diagnostics = filter_diagnostics
        return SpectralOutput(
            fused,
            confidence,
            scale_weights,
            temporal_fused,
            interaction_delta,
            innovation_context,
            structure,
            recurrent_state,
            stacked_update_gates,
            stacked_parameter_steps,
            stacked_response_steps,
            continuing_state.squeeze(-1),
            spectral_calibration_loss,
        )

