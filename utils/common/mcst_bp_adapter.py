"""Adapter between a causally trained MCST model and the audited BP tracker."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from utils.common.coordinate_transformation import enu2aer


DEFAULT_MCST_REPOSITORY = Path(r"E:\PycharmProjects\Neural Network\MCST_2D_update")


class MCSTBPAdapter:
    """Keep independent recurrent histories while sharing one MCST network."""

    model_name = "MCST"

    def __init__(self, checkpoint, repository=DEFAULT_MCST_REPOSITORY, device="cpu"):
        self.repository = Path(repository).resolve()
        self.device = torch.device(device)
        repository_parent = str(self.repository.parent)
        if repository_parent not in sys.path:
            sys.path.insert(0, repository_parent)
        from MCST_2D_update.models.MCST import ManeuverCompensationStrongTracker

        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        normalization = payload.get("normalization", {})
        self.position_scale = float(normalization.get("position_scale", 1500.0))
        self.velocity_scale = float(normalization.get("velocity_scale", 300.0))
        self.residual_scale = float(normalization.get("residual_scale", 110.0))
        protocol = payload.get("protocol", {})
        self.warmup_frames = int(protocol.get("warmup_frames", 5))
        self.residual_history_frames = int(protocol.get("residual_history_frames", 16))
        dt = float(protocol.get("dt_seconds", 2.0))
        self.model = ManeuverCompensationStrongTracker(
            4, 64, 4, 0.1, 2, 5, 3, 64,
            2, 128, 4, 0.1, 5, self.device,
            dt=dt, num_rk4_steps=1,
        ).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()

    @property
    def config(self):
        return {
            "S_POS": self.position_scale,
            "S_VEL": self.velocity_scale,
            "S_RES": self.residual_scale,
        }

    def initialize_track(self, track):
        layers = self.model.predictor.num_layers
        hidden = self.model.predictor.hidden_features
        track._mcst_hidden = torch.zeros(layers, 1, hidden, device=self.device)
        track._mcst_cell = torch.zeros_like(track._mcst_hidden)
        track._mcst_sigma_history = [
            torch.full((1, 1, 4), -18.0, device=self.device)
        ]

    def _ensure_track_runtime(self, track):
        if not hasattr(track, "_mcst_hidden"):
            self.initialize_track(track)
        missing = len(track.X) - len(track._mcst_sigma_history)
        if missing > 0:
            track._mcst_sigma_history.extend(
                torch.full((1, 1, 4), -18.0, device=self.device)
                for _ in range(missing)
            )

    @staticmethod
    def _states(values):
        return torch.as_tensor(
            np.stack([np.asarray(value).reshape(4) for value in values]),
            dtype=torch.float32,
        ).unsqueeze(0)

    @staticmethod
    def _measurements(values):
        return torch.as_tensor(
            np.stack([np.asarray(value).reshape(2) for value in values]),
            dtype=torch.float32,
        ).unsqueeze(0)

    def _normalize_states(self, states, center):
        result = states.clone()
        result[..., 0] = (result[..., 0] - center[..., 0]) / self.position_scale
        result[..., 2] = (result[..., 2] - center[..., 1]) / self.position_scale
        result[..., 1] = result[..., 1] / self.velocity_scale
        result[..., 3] = result[..., 3] / self.velocity_scale
        return result

    def _normalize_measurements(self, measurements, center):
        return (measurements - center) / self.position_scale

    def _physical_state(self, state, center):
        result = state.float().clone()
        result[..., 0] = result[..., 0] * self.position_scale + center[..., 0]
        result[..., 2] = result[..., 2] * self.position_scale + center[..., 1]
        result[..., 1] = result[..., 1] * self.velocity_scale
        result[..., 3] = result[..., 3] * self.velocity_scale
        return result

    @staticmethod
    def _left_pad(sequence, length):
        if sequence.shape[1] >= length:
            return sequence[:, -length:]
        padding = torch.zeros(
            sequence.shape[0], length - sequence.shape[1], sequence.shape[2],
            device=sequence.device, dtype=sequence.dtype,
        )
        return torch.cat([padding, sequence], dim=1)

    @torch.inference_mode()
    def predict_into_radar(self, radar, track):
        self._ensure_track_runtime(track)
        states = self._states(track.X).to(self.device)
        detections = self._measurements(track.associate_plot_track).to(self.device)
        history = self.warmup_frames
        if states.shape[1] < history or detections.shape[1] < history:
            raise ValueError("MCST prediction requested before causal warm-up is complete")

        state_window = states[:, -history:]
        detection_window = detections[:, -history:]
        center = detection_window.mean(dim=1, keepdim=True)
        state_norm = self._normalize_states(state_window, center)

        residual_length = min(
            states.shape[1], detections.shape[1], self.residual_history_frames
        )
        residual_states = states[:, -residual_length:]
        residual_detections = detections[:, -residual_length:]
        residual_center = residual_detections.mean(dim=1, keepdim=True)
        residual_state_norm = self._left_pad(
            self._normalize_states(residual_states, residual_center),
            self.residual_history_frames,
        )
        residual_detection_norm = self._left_pad(
            self._normalize_measurements(residual_detections, residual_center),
            self.residual_history_frames,
        )
        sigma = torch.cat(track._mcst_sigma_history[-history:], dim=1)
        sigma = self._left_pad(sigma, history)

        prior_norm, prior_logvar, hidden = self.model.predictor(
            sigma,
            state_norm,
            residual_detection_norm,
            residual_state_norm,
            (track._mcst_hidden, track._mcst_cell),
        )
        track._mcst_hidden, track._mcst_cell = hidden[0].detach(), hidden[1].detach()
        track._mcst_prior_norm = prior_norm.detach()
        track._mcst_prior_logvar = prior_logvar.detach()
        track._mcst_center = center.detach()
        track._mcst_sigma_history.append(prior_logvar.detach())
        prior = self._physical_state(prior_norm, center)[0, 0].cpu().numpy().reshape(4, 1)

        _, predicted_covariance = radar.tracker.KalmanPredict(
            np.zeros((4, 1)), track.gmComponents[0].P, 1.0
        )
        component = track.gmComponents[0]
        component.X = prior.copy()
        component.P = predicted_covariance.copy()
        track_position = next(
            index for index, candidate in enumerate(radar.legacy_PT_set)
            if candidate is track
        )
        cluster_index = int(radar.group_manager["IDX"][track_position]) - 1
        partition_count = len(
            radar.group_manager["group_partition"][cluster_index]
        )
        component.X_group_partition = np.repeat(prior, partition_count, axis=1)
        component.P_group_partition = np.repeat(
            predicted_covariance[:, :, None], partition_count, axis=2
        )
        track.numberOfGmComponents = 1
        track.gmComponents = track.gmComponents[:1]
        track.gmWeight = track.gmWeight[:1]
        track.X.append(prior)
        track.P = predicted_covariance
        track.X_predict = prior
        track.P_predict = predicted_covariance

        azimuth, elevation, distance = enu2aer(prior[0], prior[2], 0)
        if azimuth > 180:
            azimuth -= 360
        track.Polar_X = np.array([distance, azimuth, elevation]).T
        in_range = radar.detection_distance_range[0] <= distance <= radar.detection_distance_range[1]
        in_azimuth = radar.detection_azimuth_range[0] <= azimuth <= radar.detection_azimuth_range[1]
        in_elevation = radar.detection_elevation_range[0] <= elevation <= radar.detection_elevation_range[1]
        if not (in_range and in_azimuth and in_elevation):
            track.track_property = 2

    @torch.inference_mode()
    def update(self, track, detection):
        history = self.warmup_frames
        if len(track.X) < history + 1:
            raise ValueError("MCST update requested without a stored MCST prior")
        past_states = self._states(track.X[-history - 1 : -1]).to(self.device)
        past_detections = self._measurements(
            track.associate_plot_track[-history:]
        ).to(self.device)
        center = track._mcst_center
        past_state_norm = self._normalize_states(past_states, center)
        update_input = torch.cat(
            [past_state_norm[:, 1:], track._mcst_prior_norm], dim=1
        )
        current_detection = torch.as_tensor(
            np.asarray(detection).reshape(1, 1, 2),
            dtype=torch.float32, device=self.device,
        )
        past_detection_norm = self._normalize_measurements(past_detections, center)
        current_detection_norm = self._normalize_measurements(current_detection, center)
        detection_input = torch.cat(
            [past_detection_norm[:, 1:], current_detection_norm], dim=1
        )
        posterior_norm, _, _ = self.model.updater(
            update_input, track._mcst_prior_logvar, detection_input
        )
        posterior = self._physical_state(posterior_norm, center)
        return posterior[0, 0].cpu().numpy().reshape(4, 1)
