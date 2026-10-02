"""Causal own/peer innovation acceleration correction for SGGT-V2."""

import torch
import torch.nn as nn


class CausalGroupInnovationCorrector(nn.Module):
    """Predict a bounded acceleration increment from strictly past innovations.

    The corrector intentionally consumes only raw own/leave-one-out peer
    innovation histories plus the base acceleration at the beginning of the
    prediction interval.  It does not consume whitened innovations, NIS,
    innovation covariance, association confidence, or a learned activation
    gate.  Its two-dimensional output is expressed in normalized acceleration
    units and is added inside the same ODE that is integrated by RK4.
    """

    def __init__(
        self,
        history_length=8,
        hidden_size=32,
        raw_scale=110.0,
        max_acceleration_mps2=5.0,
        velocity_scale=300.0,
        minimum_valid_history=None,
        recent_valid_window=3,
        use_peer_innovation=True,
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.raw_scale = float(raw_scale)
        self.minimum_valid_history = (
            None if minimum_valid_history is None
            else int(minimum_valid_history)
        )
        self.recent_valid_window = int(recent_valid_window)
        self.use_peer_innovation = bool(use_peer_innovation)
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.raw_scale <= 0.0:
            raise ValueError("raw_scale must be positive")
        if max_acceleration_mps2 <= 0.0:
            raise ValueError("max_acceleration_mps2 must be positive")
        if velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        if (
            self.minimum_valid_history is not None
            and not 1 <= self.minimum_valid_history <= self.history_length
        ):
            raise ValueError(
                "minimum_valid_history must lie in [1, history_length]"
            )
        if not 1 <= self.recent_valid_window <= self.history_length:
            raise ValueError(
                "recent_valid_window must lie in [1, history_length]"
            )

        maximum_normalized_acceleration = (
            float(max_acceleration_mps2) / float(velocity_scale)
        )
        self.register_buffer(
            "maximum_normalized_acceleration",
            torch.tensor(maximum_normalized_acceleration, dtype=torch.float32),
        )

        raw_dimension = self.history_length * 2
        self.own_encoder = self._encoder(raw_dimension, hidden_size)
        self.peer_encoder = self._encoder(raw_dimension, hidden_size)
        # The interval-start base acceleration is a compact description of
        # what the frozen MPSIF/ODE predictor already intends to do.
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_size + 2, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, 2),
        )
        # Exact degeneration to the temporal MPSIF predictor at initialization.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

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

    def _features(self, innovation, base_acceleration):
        length = self.history_length
        own_raw = self._tail(innovation["raw"], length)
        peer_raw = self._tail(innovation["peer_raw"], length)
        own_valid = self._tail(innovation["valid"].bool(), length)
        peer_valid = self._tail(innovation["peer_valid"].bool(), length)
        if not self.use_peer_innovation:
            peer_raw = torch.zeros_like(peer_raw)
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

        own_raw = torch.where(own_valid, own_raw, torch.zeros_like(own_raw))
        peer_raw = torch.where(
            peer_valid, peer_raw, torch.zeros_like(peer_raw)
        )
        own_feature = torch.nan_to_num(own_raw / self.raw_scale).reshape(
            own_raw.size(0), -1
        )
        peer_feature = torch.nan_to_num(peer_raw / self.raw_scale).reshape(
            peer_raw.size(0), -1
        )

        if base_acceleration.ndim == 3:
            base_acceleration = base_acceleration.mean(dim=1)
        if base_acceleration.ndim != 2 or base_acceleration.size(-1) != 2:
            raise ValueError(
                "base_acceleration must have shape [N, 2] or [N, M, 2]"
            )
        if base_acceleration.size(0) != own_raw.size(0):
            raise ValueError(
                "base_acceleration and innovation node counts must match"
            )
        acceleration_scale = self.maximum_normalized_acceleration.to(
            base_acceleration
        ).clamp_min(1e-8)
        base_feature = torch.nan_to_num(
            base_acceleration / acceleration_scale
        ).clamp(-4.0, 4.0)

        feature = torch.cat(
            (
                self.own_encoder(own_feature),
                self.peer_encoder(peer_feature),
                base_feature,
            ),
            dim=-1,
        )
        return feature, eligible

    def predict(self, innovation, base_acceleration):
        """Return a normalized acceleration increment and diagnostics."""
        feature, eligible = self._features(innovation, base_acceleration)
        correction = torch.tanh(self.fusion(feature))
        correction = correction * self.maximum_normalized_acceleration.to(
            correction
        )
        correction = torch.where(
            eligible.unsqueeze(-1), correction, torch.zeros_like(correction)
        )
        # Retain the diagnostic fields expected by the rollout interface.  The
        # formal model is always-on when eligible and has no confidence gate.
        activation = eligible.to(correction.dtype).unsqueeze(-1)
        demand = torch.zeros_like(activation)
        return correction, eligible, activation, demand

    def forward(self, innovation, base_acceleration):
        correction, eligible, _, _ = self.predict(
            innovation, base_acceleration
        )
        return correction, eligible
