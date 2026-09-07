"""Adapter from BP association messages to SGGT causal innovation features."""

import numpy as np

from utils.common.caculate_measurement_noise_cov import (
    calculate_measurement_noise_cov,
)
from utils.common.coordinate_transformation import enu2aer


def bp_association_marginals(association_info, message_aq, eps=1e-12):
    """Return row-normalized ``P(a_i=j | Z)`` including missed detection.

    The BP implementation returns measurement-to-target messages.  Multiplying
    those messages by the original target likelihood potentials and
    normalizing produces the target association marginal.
    """
    if message_aq is None:
        return None
    potential = np.asarray(association_info["beta_message"], dtype=np.float64)[
        1:, :
    ]
    message = np.asarray(message_aq, dtype=np.float64)
    if potential.shape != message.shape:
        raise ValueError(
            f"BP potential/message shapes differ: {potential.shape} vs {message.shape}"
        )
    weight = np.maximum(potential, 0.0) * np.maximum(message, 0.0)
    denominator = weight.sum(axis=1, keepdims=True)
    invalid = denominator[:, 0] <= eps
    marginal = weight / np.maximum(denominator, eps)
    marginal[invalid] = 0.0
    marginal[invalid, 0] = 1.0
    return marginal


def association_moments(prior_position, measurements, marginal, eps=1e-12):
    """Summarize one BP target marginal without selecting a hard detection."""
    prior_position = np.asarray(prior_position, dtype=np.float64).reshape(2)
    measurements = np.asarray(measurements, dtype=np.float64).reshape(-1, 2)
    marginal = np.asarray(marginal, dtype=np.float64).reshape(-1)
    if marginal.size != measurements.shape[0] + 1:
        raise ValueError("marginal must contain miss plus every measurement")
    marginal = np.maximum(marginal, 0.0)
    marginal /= max(float(marginal.sum()), eps)
    detection_mass = float(marginal[1:].sum())
    if detection_mass > eps:
        conditional = marginal[1:] / detection_mass
        measurement = np.sum(conditional[:, None] * measurements, axis=0)
        difference = measurements - measurement
        spread = np.einsum(
            "j,ji,jk->ik", conditional, difference, difference
        )
        innovation = measurement - prior_position
    else:
        measurement = prior_position.copy()
        spread = np.zeros((2, 2), dtype=np.float64)
        innovation = np.zeros(2, dtype=np.float64)
    nonzero = marginal[marginal > eps]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    support = max(int(np.count_nonzero(marginal)), 1)
    entropy /= max(float(np.log(support)), eps) if support > 1 else 1.0
    return {
        "measurement": measurement,
        "innovation": innovation,
        "detection_mass": detection_mass,
        "entropy": entropy,
        "spread": 0.5 * (spread + spread.T),
        "valid": detection_mass > 0.05,
    }


def _innovation_covariance(radar, track):
    prior = np.asarray(track.X_predict, dtype=np.float64).reshape(-1, 1)
    covariance = np.asarray(track.P_predict, dtype=np.float64)
    predicted_measurement = radar.tracker.H @ prior
    azimuth, elevation, distance = enu2aer(
        predicted_measurement[0, 0], predicted_measurement[1, 0], 0
    )
    _, measurement_covariance = calculate_measurement_noise_cov(
        R=distance,
        A=azimuth,
        E=elevation,
        rangeNoise=radar.distance_measurement_error,
        aziNoise=radar.azimuth_measurement_error * np.pi / 180.0,
        eleNoise=0,
    )
    # measurement_evaluation uses the same factor in its BP likelihood/update.
    measurement_covariance = 10.0 * np.asarray(measurement_covariance[:2, :2])
    value = (
        radar.tracker.H @ covariance @ radar.tracker.H.T
        + measurement_covariance
    )
    return 0.5 * (value + value.T)


def attach_bp_association_context(radar, association_info, message_aq):
    """Attach causal soft-association moments to legacy tracks for next frame."""
    marginal = bp_association_marginals(association_info, message_aq)
    if marginal is None:
        return None
    measurements = np.asarray([
        np.asarray(plot.X, dtype=np.float64).reshape(2)
        for plot in radar.plot_track_set
    ])
    for target_index, track in enumerate(radar.legacy_PT_set):
        prior_position = np.asarray(track.X_predict, dtype=np.float64)[
            [0, 2], 0
        ]
        moments = association_moments(
            prior_position, measurements, marginal[target_index]
        )
        covariance = _innovation_covariance(radar, track)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        inverse_sqrt = (
            eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
        )
        whitened = inverse_sqrt @ moments["innovation"]
        track.sggt_association_measurement = moments["measurement"].reshape(2, 1)
        track.sggt_innovation_raw = moments["innovation"].copy()
        track.sggt_innovation_covariance = covariance
        track.sggt_innovation_whitened = whitened
        track.sggt_nis = float(whitened @ whitened)
        track.sggt_association_detection_mass = moments["detection_mass"]
        track.sggt_association_entropy = moments["entropy"]
        track.sggt_association_spread = moments["spread"]
        track.sggt_association_valid = bool(moments["valid"])
    return marginal


def attach_missed_detection_context(radar):
    """Write a fresh invalid innovation when the current scan has no plots."""
    for track in radar.legacy_PT_set:
        prior_position = np.asarray(track.X_predict, dtype=np.float64)[
            [0, 2], 0
        ]
        track.sggt_association_measurement = prior_position.reshape(2, 1)
        track.sggt_innovation_raw = np.zeros(2, dtype=np.float64)
        track.sggt_innovation_whitened = np.zeros(2, dtype=np.float64)
        track.sggt_innovation_covariance = _innovation_covariance(radar, track)
        track.sggt_nis = 0.0
        track.sggt_association_detection_mass = 0.0
        track.sggt_association_entropy = 0.0
        track.sggt_association_spread = np.zeros((2, 2), dtype=np.float64)
        track.sggt_association_valid = False
