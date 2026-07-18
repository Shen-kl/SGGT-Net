from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureOutput(NamedTuple):
    adjacency_vector: torch.Tensor
    structural_features: torch.Tensor
    vector_mask: torch.Tensor
    group_count: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    node_entropy: torch.Tensor


class SpectralOutput(NamedTuple):
    latent: torch.Tensor
    confidence: torch.Tensor
    scale_weights: torch.Tensor


def _upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    row, col = torch.triu_indices(
        matrix.size(0), matrix.size(1), offset=1, device=matrix.device
    )
    return matrix[row, col]


class TemporalProbabilisticSEAN(nn.Module):
    """Infer a symmetric soft interaction graph from temporal node features."""

    def __init__(self, hidden_dim: int, struct_dim: int):
        super().__init__()
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
    ) -> StructureOutput:
        if states.dim() == 3:
            states = states.mean(dim=1)
        if mask is None:
            valid_nodes = torch.ones(hidden.size(0), dtype=torch.bool, device=hidden.device)
        else:
            valid_nodes = mask.reshape(mask.size(0), -1).all(dim=-1).bool()

        projected = self.node_projection(hidden)
        structural_features = torch.zeros_like(projected)
        node_entropy = torch.ones(hidden.size(0), 1, device=hidden.device)
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

            vector_parts.append(_upper_triangle(adjacency))
            vector_mask_parts.append(_upper_triangle(pair_valid).bool())

            source, target = torch.where(pair_valid)
            if source.numel() > 0:
                edge_indices.append(
                    torch.stack([node_indices[source], node_indices[target]], dim=0)
                )
                edge_weights.append(adjacency[source, target])
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
        )


class JointGraphTemporalSpectralFilter(nn.Module):
    """Multi-scale temporal FFT followed by differentiable graph filtering."""

    def __init__(
        self,
        hidden_dim: int,
        windows: tuple[int, ...] = (8, 16),
        signal_dim: int = 2,
        chebyshev_order: int = 2,
    ):
        super().__init__()
        self.windows = tuple(int(window) for window in windows)
        self.chebyshev_order = int(chebyshev_order)
        self.signal_dim = signal_dim
        self.frequency_filters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, self.chebyshev_order + 1),
                )
                for _ in self.windows
            ]
        )
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
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
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

    def _filter_scale(
        self,
        residual: torch.Tensor,
        window: int,
        filter_network: nn.Module,
        temporal_encoder: nn.Module,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_entropy: torch.Tensor,
    ) -> torch.Tensor:
        sequence = residual[:, -window:, :]
        if sequence.size(1) < window:
            sequence = F.pad(sequence, (0, 0, window - sequence.size(1), 0))
        sequence = torch.nan_to_num(sequence)

        spectrum = torch.fft.rfft(sequence.transpose(1, 2), n=window, dim=-1)
        spectral_features = torch.view_as_real(spectrum).permute(0, 2, 1, 3)
        frequency_count = spectral_features.size(1)

        terms = [spectral_features]
        if self.chebyshev_order >= 1:
            terms.append(self._graph_propagate(terms[0], edge_index, edge_weight))
        for _ in range(2, self.chebyshev_order + 1):
            propagated = self._graph_propagate(terms[-1], edge_index, edge_weight)
            terms.append(2 * propagated - terms[-2])

        frequencies = torch.linspace(0, 1, frequency_count, device=residual.device)
        frequency_input = torch.stack(
            [
                frequencies.unsqueeze(0).expand(residual.size(0), -1),
                node_entropy.expand(-1, frequency_count),
            ],
            dim=-1,
        )
        coefficients = torch.softmax(filter_network(frequency_input), dim=-1)
        filtered = sum(
            coefficients[..., order, None, None] * term
            for order, term in enumerate(terms)
        )

        filtered_complex = torch.view_as_complex(
            filtered.permute(0, 2, 1, 3).contiguous()
        )
        filtered_sequence = torch.fft.irfft(filtered_complex, n=window, dim=-1)
        return temporal_encoder(filtered_sequence).squeeze(-1)

    def forward(
        self,
        residual: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_entropy: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> SpectralOutput:
        if valid_mask is None:
            valid_ratio = torch.ones(residual.size(0), 1, device=residual.device)
        else:
            mask = valid_mask.reshape(valid_mask.size(0), -1).float()
            valid_ratio = mask.mean(dim=-1, keepdim=True)

        latents = []
        scores = []
        confidence_features = torch.cat([node_entropy, valid_ratio], dim=-1)
        for window, filter_network, temporal_encoder, scorer in zip(
            self.windows,
            self.frequency_filters,
            self.temporal_encoders,
            self.scale_scorers,
        ):
            latent = self._filter_scale(
                residual,
                window,
                filter_network,
                temporal_encoder,
                edge_index,
                edge_weight,
                node_entropy,
            )
            latents.append(latent)
            scores.append(scorer(torch.cat([latent, confidence_features], dim=-1)))

        stacked_latents = torch.stack(latents, dim=1)
        scale_weights = torch.softmax(torch.cat(scores, dim=-1), dim=-1)
        fused = (stacked_latents * scale_weights.unsqueeze(-1)).sum(dim=1)
        fused = self.output_projection(fused)
        confidence = self.confidence_head(torch.cat([fused, confidence_features], dim=-1))
        return SpectralOutput(fused, confidence, scale_weights)

