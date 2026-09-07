"""Causal own/peer innovation correction for the one-step tracking prior."""

import math

import torch
import torch.nn as nn


class CausalGroupInnovationCorrector(nn.Module):
    """Predict a bounded prior-state correction from strictly past innovations.

    The target's own innovation and the leave-one-out peer innovation are kept
    as separate channels.  Peer histories are formed online with the graph from
    the same historical frame; this module never sees the current measurement.
    """

    def __init__(
        self,
        history_length=8,
        hidden_size=32,
        raw_scale=110.0,
        whiten_clip=20.0,
        max_position_m=50.0,
        max_velocity_mps=5.0,
        position_scale=1500.0,
        velocity_scale=300.0,
        activation_floor=0.5,
        activation_nis_low=1.38629436112,
        activation_nis_high=5.99146454711,
        minimum_valid_history=None,
        recent_valid_window=3,
        association_context=False,
        use_peer_innovation=True,
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.raw_scale = float(raw_scale)
        self.whiten_clip = float(whiten_clip)
        self.activation_floor = float(activation_floor)
        self.activation_nis_low = float(activation_nis_low)
        self.activation_nis_high = float(activation_nis_high)
        self.minimum_valid_history = (
            None if minimum_valid_history is None
            else int(minimum_valid_history)
        )
        self.recent_valid_window = int(recent_valid_window)
        self.association_context = bool(association_context)
        self.use_peer_innovation = bool(use_peer_innovation)
        if not 0.0 <= self.activation_floor <= 1.0:
            raise ValueError("causal activation floor must lie in [0, 1]")
        if self.activation_nis_high <= self.activation_nis_low:
            raise ValueError("causal activation NIS high must exceed NIS low")
        if (
            self.minimum_valid_history is not None
            and not 1 <= self.minimum_valid_history <= self.history_length
        ):
            raise ValueError("minimum valid history must lie in [1, history_length]")
        if not 1 <= self.recent_valid_window <= self.history_length:
            raise ValueError("recent valid window must lie in [1, history_length]")
        self.register_buffer(
            "maximum_normalized_correction",
            torch.tensor(
                [
                    max_position_m / position_scale,
                    max_position_m / position_scale,
                    max_velocity_mps / velocity_scale,
                    max_velocity_mps / velocity_scale,
                ],
                dtype=torch.float32,
            ),
        )

        raw_dimension = self.history_length * 2
        # own/peer whitened histories plus per-frame NIS, log diagonal S and
        # correlation.  These features condition reliability, but do not mix
        # the two raw motion signals before their independent encoders.
        uncertainty_per_frame = 2 + 2 + 1 + 2 + 1
        if self.association_context:
            # own/peer validity, detection mass, association entropy, and two
            # rotation-invariant eigenvalue summaries of candidate spread.
            uncertainty_per_frame += 6
        uncertainty_dimension = self.history_length * uncertainty_per_frame
        self.own_encoder = self._encoder(raw_dimension, hidden_size)
        self.peer_encoder = self._encoder(raw_dimension, hidden_size)
        self.uncertainty_encoder = self._encoder(
            uncertainty_dimension, hidden_size
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, 4),
        )
        self.selector = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        # Exact Temporal degeneration at initialization.  The first optimizer
        # step learns only the output map; upstream features become trainable
        # as soon as that map is non-zero.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        # Start from an auditable non-collapsed 0.5 policy.  Balanced
        # counterfactual supervision, rather than the class prior, decides
        # whether the selector subsequently moves toward either branch.
        nn.init.xavier_uniform_(self.selector[-1].weight, gain=0.02)
        nn.init.zeros_(self.selector[-1].bias)

    @staticmethod
    def _encoder(input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )

    @staticmethod
    def _tail(value, length):
        if value.size(1) < length:
            padding = value.new_zeros(
                value.size(0), length - value.size(1), *value.shape[2:]
            )
            value = torch.cat((padding, value), dim=1)
        return value[:, -length:]

    def _features(self, innovation):
        length = self.history_length
        own_raw = self._tail(innovation["raw"], length)
        peer_raw = self._tail(innovation["peer_raw"], length)
        own_white = self._tail(innovation["whitened"], length)
        peer_white = self._tail(innovation["peer_whitened"], length)
        covariance = self._tail(innovation["covariance"], length)
        nis = self._tail(innovation["nis"], length)
        own_valid = self._tail(innovation["valid"].bool(), length)
        peer_valid = self._tail(innovation["peer_valid"].bool(), length)
        if not self.use_peer_innovation:
            peer_raw = torch.zeros_like(peer_raw)
            peer_white = torch.zeros_like(peer_white)
            peer_valid = own_valid

        paired_valid = own_valid & peer_valid
        if self.minimum_valid_history is None:
            eligible = paired_valid.all(dim=1).squeeze(-1)
        else:
            enough_history = (
                paired_valid.sum(dim=1).squeeze(-1)
                >= self.minimum_valid_history
            )
            recent = min(self.recent_valid_window, length)
            has_recent = paired_valid[:, -recent:].any(dim=1).squeeze(-1)
            eligible = enough_history & has_recent
        raw_scale = max(self.raw_scale, 1e-6)
        white_scale = max(self.whiten_clip, 1e-6)
        own_feature = torch.nan_to_num(own_raw / raw_scale).reshape(
            own_raw.size(0), -1
        )
        peer_feature = torch.nan_to_num(peer_raw / raw_scale).reshape(
            peer_raw.size(0), -1
        )

        safe_covariance = torch.nan_to_num(
            covariance, nan=0.0, posinf=raw_scale ** 2, neginf=0.0
        )
        diagonal = safe_covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        log_diagonal = torch.log1p(diagonal / (raw_scale ** 2))
        denominator = torch.sqrt(
            diagonal[..., 0] * diagonal[..., 1]
        ).clamp_min(1e-6)
        correlation = (
            safe_covariance[..., 0, 1] / denominator
        ).clamp(-1.0, 1.0).unsqueeze(-1)
        nis_scale = math.log1p(white_scale ** 2)
        uncertainty_parts = [
            own_white.clamp(-white_scale, white_scale) / white_scale,
            peer_white.clamp(-white_scale, white_scale) / white_scale,
            torch.log1p(nis.clamp_min(0.0)) / max(nis_scale, 1e-6),
            log_diagonal,
            correlation,
        ]
        if self.association_context:
            detection_mass = self._tail(
                innovation["association_detection_mass"], length
            ).clamp(0.0, 1.0)
            entropy = self._tail(
                innovation["association_entropy"], length
            ).clamp(0.0, 1.0)
            spread = self._tail(innovation["association_spread"], length)
            spread = torch.nan_to_num(spread, nan=0.0, posinf=raw_scale ** 2)
            eigenvalues = torch.linalg.eigvalsh(
                0.5 * (spread + spread.transpose(-1, -2))
            ).clamp_min(0.0)
            spread_scale = math.log1p(raw_scale ** 2)
            spread_summary = torch.log1p(eigenvalues) / max(spread_scale, 1e-6)
            uncertainty_parts.extend((
                own_valid.to(own_raw.dtype),
                peer_valid.to(own_raw.dtype),
                detection_mass,
                entropy,
                spread_summary,
            ))
        uncertainty = torch.nan_to_num(
            torch.cat(uncertainty_parts, dim=-1)
        ).reshape(own_raw.size(0), -1)

        feature = torch.cat(
            (
                self.own_encoder(own_feature),
                self.peer_encoder(peer_feature),
                self.uncertainty_encoder(uncertainty),
            ),
            dim=-1,
        )
        return feature, eligible

    def _causal_demand(self, innovation, eligible):
        """Build a truth-free continuous demand target from past innovations.

        NIS supplies the statistically calibrated magnitude signal.  Positive
        own/peer directional agreement raises demand only when both channels
        are simultaneously surprising, so an isolated noisy target cannot
        masquerade as a common maneuver.
        """
        length = self.history_length
        own = self._tail(innovation["whitened"], length)
        peer = self._tail(innovation["peer_whitened"], length)
        nis = self._tail(innovation["nis"], length).squeeze(-1)
        own_valid = self._tail(innovation["valid"].bool(), length)
        peer_valid = self._tail(innovation["peer_valid"].bool(), length)
        if not self.use_peer_innovation:
            peer = torch.zeros_like(peer)
            peer_valid = own_valid
        recent = min(3, length)
        own_recent = own[:, -recent:]
        peer_recent = peer[:, -recent:]
        recent_valid = (own_valid & peer_valid)[:, -recent:].to(own.dtype)
        denominator_valid = recent_valid.sum(dim=1).squeeze(-1).clamp_min(1.0)
        own_recent = own_recent * recent_valid
        peer_recent = peer_recent * recent_valid
        own_nis = (
            nis[:, -recent:] * recent_valid.squeeze(-1)
        ).sum(dim=1) / denominator_valid
        peer_nis = (
            peer_recent.square().sum(dim=-1)
        ).sum(dim=1) / denominator_valid
        denominator = self.activation_nis_high - self.activation_nis_low

        def excess(value):
            return ((value - self.activation_nis_low) / denominator).clamp(0.0, 1.0)

        own_excess = excess(own_nis)
        peer_excess = excess(peer_nis)
        own_direction = own_recent.sum(dim=1) / denominator_valid.unsqueeze(-1)
        peer_direction = peer_recent.sum(dim=1) / denominator_valid.unsqueeze(-1)
        alignment = torch.nn.functional.cosine_similarity(
            own_direction, peer_direction, dim=-1, eps=1e-6
        ).clamp_min(0.0)
        common_demand = torch.sqrt(own_excess * peer_excess) * alignment
        demand = 0.6 * own_excess + 0.4 * common_demand
        target = self.activation_floor + (1.0 - self.activation_floor) * demand
        return torch.where(eligible, target, torch.zeros_like(target)).unsqueeze(-1)

    def predict(self, innovation):
        """Return the frozen-expert correction and its causal activation."""
        feature, eligible = self._features(innovation)
        correction = torch.tanh(self.fusion(feature))
        correction = correction * self.maximum_normalized_correction.to(correction)
        correction = torch.where(
            eligible.unsqueeze(-1), correction, torch.zeros_like(correction)
        )
        activation = torch.sigmoid(self.selector(feature))
        activation = torch.where(
            eligible.unsqueeze(-1), activation, torch.zeros_like(activation)
        )
        demand = self._causal_demand(innovation, eligible)
        return correction, eligible, activation, demand

    def forward(self, innovation):
        # Keep the original two-value API for diagnostics and unit tests.
        correction, eligible, _, _ = self.predict(innovation)
        return correction, eligible

    @staticmethod
    def apply_correction(base_state, correction, alpha=1.0):
        """Mix a correction while making alpha=0 an exact counterfactual."""
        if torch.is_tensor(alpha):
            scale = alpha.to(correction)
            if scale.ndim == 1:
                scale = scale.unsqueeze(-1)
            return base_state + scale.unsqueeze(-1) * correction.unsqueeze(1)
        return base_state + float(alpha) * correction.unsqueeze(1)
