"""Utilities shared by SGGT-Net training and validation."""

import numpy as np
import torch


class MeasurementNoiseCalculator:
    @staticmethod
    def enu2aer(enu_x, enu_y, enu_z):
        azi = torch.atan2(enu_x, enu_y) * 180 / np.pi
        elev = torch.atan2(enu_z, (enu_x ** 2 + enu_y ** 2).sqrt()) * 180 / np.pi
        range_val = (enu_x ** 2 + enu_y ** 2 + enu_z ** 2).sqrt()
        return azi, elev, range_val


class CovarianceNormalizer:
    @staticmethod
    def normalize_measurement_cov(covariance, config):
        return covariance / config['S_POS'] ** 2

    @staticmethod
    def normalize_state_cov(covariance, config):
        normalized = covariance.clone()
        normalized[..., [0, 1], :] /= config['S_POS']
        normalized[..., :, [0, 1]] /= config['S_POS']
        normalized[..., [2, 3], :] /= config['S_VEL']
        normalized[..., :, [2, 3]] /= config['S_VEL']
        return normalized


class GraphDataProcessor:
    @staticmethod
    def leave_one_out_peer_mean(features, edge_index, valid):
        """Mean valid source-neighbour features, excluding the target itself."""
        output = torch.zeros_like(features)
        count = features.new_zeros(features.size(0), 1)
        if edge_index is None or edge_index.numel() == 0:
            return output, count.bool()
        source, target = edge_index.long()
        keep = (
            (source != target)
            & valid[source]
            & valid[target]
        )
        source = source[keep]
        target = target[keep]
        if source.numel():
            output.index_add_(0, target, features[source])
            count.index_add_(
                0, target, torch.ones(source.numel(), 1, device=features.device,
                                      dtype=features.dtype)
            )
            output = output / count.clamp_min(1.0)
        return output, count > 0

    @staticmethod
    def init_input_window(graph_data, config, args, start_bias=0):
        """Create the first causal input/target window."""
        batch = graph_data.x.shape[0]
        input_len = config['input_graph_window_len']
        output_len = config['output_graph_window_len']
        target_start = start_bias + input_len
        window = graph_data.clone()

        window.x = graph_data.x[:, start_bias:target_start, :].clone()
        window.y = graph_data.y[:, target_start:target_start + output_len, :]
        previous_truth_velocity = graph_data.y[
            :, target_start - 1:target_start, [1, 3]
        ]
        truth_velocity = graph_data.y[
            :, target_start:target_start + output_len, [1, 3]
        ]
        window.maneuver_delta_v = truth_velocity - torch.cat(
            (previous_truth_velocity, truth_velocity[:, :-1]), dim=1
        )
        window.measurement = graph_data.measurement[:, start_bias:target_start, :]
        window.measurement_future = graph_data.measurement[
            :, target_start:target_start + output_len, :
        ]
        # The online network has produced no priors at initialization.
        window.residual = torch.zeros(
            batch,
            int(np.max(args.decoder_residual_length)),
            config['z_dimension'],
            dtype=graph_data.x.dtype,
            device=graph_data.x.device,
        )
        innovation_length = int(np.max(args.decoder_residual_length))
        innovation_shape = (
            batch, innovation_length, config['z_dimension']
        )
        # ``residual`` is retained for exact compatibility with the original
        # Temporal GASF.  The following buffers carry the EKF-consistent
        # physical innovation, its whitened counterpart and S covariance.
        window.innovation_raw = torch.zeros(
            innovation_shape,
            dtype=graph_data.x.dtype,
            device=graph_data.x.device,
        )
        window.innovation_whitened = torch.zeros_like(window.innovation_raw)
        window.innovation_peer_raw = torch.zeros_like(window.innovation_raw)
        window.innovation_peer_whitened = torch.zeros_like(window.innovation_raw)
        identity = torch.eye(
            config['z_dimension'],
            dtype=graph_data.x.dtype,
            device=graph_data.x.device,
        )
        window.innovation_covariance = identity.view(
            1, 1, config['z_dimension'], config['z_dimension']
        ).expand(batch, innovation_length, -1, -1).clone()
        window.nis_history = torch.zeros(
            batch,
            innovation_length,
            1,
            dtype=graph_data.x.dtype,
            device=graph_data.x.device,
        )
        window.innovation_valid = torch.zeros(
            batch,
            innovation_length,
            1,
            dtype=torch.bool,
            device=graph_data.x.device,
        )
        window.innovation_peer_valid = torch.zeros_like(window.innovation_valid)
        window.association_detection_mass = torch.zeros(
            batch, innovation_length, 1,
            dtype=graph_data.x.dtype, device=graph_data.x.device,
        )
        window.association_entropy = torch.zeros_like(
            window.association_detection_mass
        )
        window.association_spread = torch.zeros(
            batch, innovation_length, config['z_dimension'], config['z_dimension'],
            dtype=graph_data.x.dtype, device=graph_data.x.device,
        )
        window.edge_index = graph_data.edge_index[start_bias:target_start]
        window.edge_features = graph_data.edge_features[start_bias:target_start]
        window.nan_mask = graph_data.nan_mask[:, start_bias:target_start, :]
        current_mask = graph_data.real_mask[
            :, target_start:target_start + output_len, :
        ]
        previous_mask = graph_data.real_mask[
            :, target_start - 1:target_start, :
        ]
        # A target is predictable only after it existed in the preceding
        # frame.  Its birth frame initializes the filter and is not a forecast.
        window.current_real_mask = current_mask
        window.real_mask = current_mask & previous_mask
        if hasattr(graph_data, 'struct_feat') and graph_data.struct_feat:
            window.struct_feat = graph_data.struct_feat[
                target_start:target_start + output_len
            ]
        return window

    @staticmethod
    def update_input_window(
        window,
        graph_data,
        config,
        x_update_renorm,
        prior_state_renorm,
        bias,
        args,
        measurement_innovation=None,
        innovation_covariance=None,
        whitened_innovation=None,
        nis=None,
        association_detection_mass=None,
        association_entropy=None,
        association_spread=None,
        association_valid=None,
    ):
        """Slide one frame and append z_k - x_{k|k-1} to the residual buffer."""
        del args  # The residual buffer length is already fixed at initialization.
        input_len = config['input_graph_window_len']
        output_len = config['output_graph_window_len']
        appended_frame = bias + input_len - 1
        target_start = bias + input_len

        current_valid = graph_data.real_mask[:, appended_frame, 0].bool()
        previous_valid = graph_data.real_mask[:, appended_frame - 1, 0].bool()
        continuing = current_valid & previous_valid
        posterior = x_update_renorm.squeeze(1).detach()
        # Preserve the packed initial state on a birth frame.  Before birth or
        # after death, keep the dataset padding rather than writing a fictitious
        # Kalman posterior into the causal history.
        stored_state = graph_data.x[:, appended_frame, :].clone()
        stored_state[continuing] = posterior[continuing]
        graph_data.x[:, appended_frame, :] = stored_state
        window.x[:, :-1, :] = window.x[:, 1:, :].clone()
        window.x[:, -1, :] = stored_state
        window.y = graph_data.y[:, target_start:target_start + output_len, :]
        previous_truth_velocity = graph_data.y[
            :, target_start - 1:target_start, [1, 3]
        ]
        truth_velocity = graph_data.y[
            :, target_start:target_start + output_len, [1, 3]
        ]
        # Keep the maneuver-demand target independent of the online posterior.
        # `graph_data.x` is intentionally updated in-place below/above as the
        # filter rolls forward, whereas `graph_data.y` remains immutable truth.
        window.maneuver_delta_v = truth_velocity - torch.cat(
            (previous_truth_velocity, truth_velocity[:, :-1]), dim=1
        )
        window.measurement = graph_data.measurement[:, bias:bias + input_len, :]
        window.measurement_future = graph_data.measurement[
            :, target_start:target_start + output_len, :
        ]

        prior = prior_state_renorm.squeeze(1).detach()
        innovation = graph_data.measurement[:, appended_frame, :] - prior[:, [0, 2]]
        if measurement_innovation is not None:
            innovation = measurement_innovation.detach()
            if innovation.ndim == 3 and innovation.size(1) == 1:
                innovation = innovation[:, 0]
        valid = graph_data.real_mask[:, appended_frame, 0].unsqueeze(-1)
        if association_valid is not None:
            association_feature_valid = association_valid.detach().bool()
            while association_feature_valid.ndim > valid.ndim:
                association_feature_valid = association_feature_valid.squeeze(1)
            valid = valid & association_feature_valid
        innovation = torch.where(valid, innovation, torch.zeros_like(innovation))
        window.residual[:, :-1, :] = window.residual[:, 1:, :].clone()
        window.residual[:, -1, :] = innovation

        if hasattr(window, 'innovation_raw'):
            # Only a continuing track has a meaningful one-step prior and
            # therefore a valid innovation. Birth frames initialize the
            # filter; absent/dead targets remain padding throughout.
            feature_valid = continuing.unsqueeze(-1)
            if association_valid is not None:
                feature_valid = feature_valid & association_feature_valid
            raw = (
                measurement_innovation.detach()
                if measurement_innovation is not None else innovation
            )
            if raw.ndim == 3 and raw.size(1) == 1:
                raw = raw[:, 0]
            white = (
                whitened_innovation.detach()
                if whitened_innovation is not None else torch.zeros_like(raw)
            )
            if white.ndim == 3 and white.size(1) == 1:
                white = white[:, 0]
            nis_value = (
                nis.detach() if nis is not None
                else white.square().sum(dim=-1, keepdim=True)
            )
            if nis_value.ndim == 3 and nis_value.size(1) == 1:
                nis_value = nis_value[:, 0]
            covariance_value = (
                innovation_covariance.detach()
                if innovation_covariance is not None
                else window.innovation_covariance[:, -1]
            )
            raw = torch.where(feature_valid, raw, torch.zeros_like(raw))
            white = torch.where(feature_valid, white, torch.zeros_like(white))
            nis_value = torch.where(
                feature_valid, nis_value, torch.zeros_like(nis_value)
            )
            identity = torch.eye(
                covariance_value.size(-1),
                dtype=covariance_value.dtype,
                device=covariance_value.device,
            ).unsqueeze(0).expand_as(covariance_value)
            covariance_value = torch.where(
                feature_valid.unsqueeze(-1), covariance_value, identity
            )
            detection_mass = (
                association_detection_mass.detach()
                if association_detection_mass is not None
                else feature_valid.to(raw.dtype)
            )
            entropy_value = (
                association_entropy.detach()
                if association_entropy is not None
                else torch.zeros_like(detection_mass)
            )
            spread_value = (
                association_spread.detach()
                if association_spread is not None
                else torch.zeros_like(covariance_value)
            )
            while detection_mass.ndim > 2:
                detection_mass = detection_mass.squeeze(1)
            while entropy_value.ndim > 2:
                entropy_value = entropy_value.squeeze(1)
            detection_mass = torch.where(
                feature_valid, detection_mass, torch.zeros_like(detection_mass)
            )
            entropy_value = torch.where(
                feature_valid, entropy_value, torch.zeros_like(entropy_value)
            )
            spread_value = torch.where(
                feature_valid.unsqueeze(-1),
                spread_value,
                torch.zeros_like(spread_value),
            )
            peer_feature, peer_valid = GraphDataProcessor.leave_one_out_peer_mean(
                torch.cat((raw, white), dim=-1),
                graph_data.edge_index[appended_frame],
                feature_valid[:, 0],
            )
            peer_raw = peer_feature[:, :raw.size(-1)]
            peer_white = peer_feature[:, raw.size(-1):]
            for name, value in (
                ('innovation_raw', raw),
                ('innovation_whitened', white),
                ('innovation_peer_raw', peer_raw),
                ('innovation_peer_whitened', peer_white),
                ('innovation_covariance', covariance_value),
                ('nis_history', nis_value),
                ('innovation_valid', feature_valid),
                ('innovation_peer_valid', peer_valid),
                ('association_detection_mass', detection_mass),
                ('association_entropy', entropy_value),
                ('association_spread', spread_value),
            ):
                buffer = getattr(window, name)
                buffer[:, :-1] = buffer[:, 1:].clone()
                buffer[:, -1] = value

        window.edge_index = graph_data.edge_index[bias:bias + input_len]
        window.edge_features = graph_data.edge_features[bias:bias + input_len]
        window.nan_mask = graph_data.nan_mask[:, bias:bias + input_len, :]
        current_mask = graph_data.real_mask[
            :, target_start:target_start + output_len, :
        ]
        previous_mask = graph_data.real_mask[
            :, target_start - 1:target_start, :
        ]
        window.current_real_mask = current_mask
        window.real_mask = current_mask & previous_mask
        if hasattr(graph_data, 'struct_feat') and graph_data.struct_feat:
            window.struct_feat = graph_data.struct_feat[
                target_start:target_start + output_len
            ]
        return window

    @staticmethod
    def init_covariance_from_measurement(measurement_covariance, delta_t, config,
                                         device=None):
        """Initialize [x, vx, y, vy] covariance from Cartesian measurement R."""
        batch = measurement_covariance.shape[0]
        covariance = torch.zeros(batch, 4, 4, device=device,
                                 dtype=measurement_covariance.dtype)
        r = measurement_covariance
        covariance[..., 0, 0] = r[..., 0, 0]
        covariance[..., 0, 1] = r[..., 0, 0] / delta_t
        covariance[..., 0, 2] = r[..., 0, 1]
        covariance[..., 0, 3] = r[..., 0, 1] / delta_t
        covariance[..., 1, 0] = r[..., 0, 0] / delta_t
        covariance[..., 1, 1] = 2 * r[..., 0, 0] / delta_t ** 2
        covariance[..., 1, 2] = r[..., 0, 1] / delta_t
        covariance[..., 1, 3] = 2 * r[..., 0, 1] / delta_t ** 2
        covariance[..., 2, 0] = r[..., 0, 1]
        covariance[..., 2, 1] = r[..., 0, 1] / delta_t
        covariance[..., 2, 2] = r[..., 1, 1]
        covariance[..., 2, 3] = r[..., 1, 1] / delta_t
        covariance[..., 3, 0] = r[..., 0, 1] / delta_t
        covariance[..., 3, 1] = 2 * r[..., 0, 1] / delta_t ** 2
        covariance[..., 3, 2] = r[..., 1, 1] / delta_t
        covariance[..., 3, 3] = 2 * r[..., 1, 1] / delta_t ** 2

        transform = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0],
             [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=covariance.dtype, device=device,
        )
        covariance = transform @ covariance @ transform.transpose(-1, -2)
        return CovarianceNormalizer.normalize_state_cov(covariance, config)


class TrainingMetrics:
    def __init__(self):
        self.metrics = {
            'loss': [],
            'location_rmse_prediction': [],
            'velocity_rmse_prediction': [],
            'location_rmse_update': [],
            'velocity_rmse_update': [],
            'location_rmse_measurement': [],
            'f1': [],
            'loss_q_rank': [],
            'loss_q_surprise': [],
            'loss_q_smooth': [],
            'loss_q_baseline': [],
            'q_surprise_active': [],
            'q_acceleration_multiplier_mean': [],
            'q_acceleration_multiplier_std': [],
            'q_position_variance_mean_m2': [],
            'q_position_variance_std_m2': [],
            'q_causal_surprise_mean': [],
            'q_causal_surprise_std': [],
            'gasf_parameter_update_gate_mean': [],
            'gasf_parameter_update_gate_std': [],
            'loss_interaction_base': [],
            'loss_interaction_expert': [],
            'loss_interaction_strong_base': [],
            'loss_interaction_strong_expert': [],
            'interaction_strong_advantage': [],
            'loss_interaction_gate': [],
            'loss_interaction_regret': [],
            'loss_interaction_aux_expert': [],
            'loss_interaction_aux_rank': [],
            'loss_interaction_state_alignment': [],
            'loss_interaction_state_improvement': [],
            'loss_causal_velocity_guard': [],
            'loss_interaction_demand': [],
            'interaction_demand_balanced_active': [],
            'interaction_demand_positive_fraction': [],
            'interaction_demand_accuracy': [],
            'interaction_demand_balanced_accuracy': [],
            'interaction_demand_positive_score_mean': [],
            'interaction_demand_negative_score_mean': [],
            'interaction_demand_score_separation': [],
            'interaction_demand_tp_count': [],
            'interaction_demand_tn_count': [],
            'interaction_demand_positive_count': [],
            'interaction_demand_negative_count': [],
            'interaction_demand_positive_score_sum': [],
            'interaction_demand_negative_score_sum': [],
            'interaction_gate_balanced_active': [],
            'interaction_supervision_active': [],
            'interaction_advantage_mean': [],
            'interaction_advantage_abs_mean': [],
            'interaction_advantage_std': [],
            'interaction_expert_better_fraction': [],
            'interaction_gate_target_mean': [],
            'interaction_gate_target_std': [],
            'interaction_gate_tp_count': [],
            'interaction_gate_tn_count': [],
            'interaction_gate_positive_count': [],
            'interaction_gate_negative_count': [],
            'interaction_gate_positive_score_sum': [],
            'interaction_gate_negative_score_sum': [],
            'causal_gate_mae': [],
            'causal_gate_correlation': [],
            'interaction_gate_balanced_accuracy': [],
            'interaction_gate_positive_score_mean': [],
            'interaction_gate_negative_score_mean': [],
            'interaction_gate_score_separation': [],
            'interaction_activation_mean': [],
            'interaction_activation_std': [],
            'interaction_activation_below_005': [],
            'interaction_activation_above_095': [],
            'causal_innovation_eligible_fraction': [],
            'causal_innovation_acceleration_rms_mps2': [],
            'causal_innovation_acceleration_max_mps2': [],
            'association_detection_mass_mean': [],
            'association_entropy_mean': [],
            'association_missed_fraction': [],
            'association_spread_trace_mean': [],
            'location_rmse_prediction_sse': [],
            'velocity_rmse_prediction_sse': [],
            'location_rmse_update_sse': [],
            'velocity_rmse_update_sse': [],
            'location_rmse_measurement_sse': [],
            'prediction_count': [],
            'update_count': [],
        }

    _POOLED_RMSE = {
        'location_rmse_prediction': ('location_rmse_prediction_sse', 'prediction_count'),
        'velocity_rmse_prediction': ('velocity_rmse_prediction_sse', 'prediction_count'),
        'location_rmse_update': ('location_rmse_update_sse', 'update_count'),
        'velocity_rmse_update': ('velocity_rmse_update_sse', 'update_count'),
        'location_rmse_measurement': ('location_rmse_measurement_sse', 'prediction_count'),
    }
    _DEMAND_COUNTS = {
        'interaction_demand_tp_count',
        'interaction_demand_tn_count',
        'interaction_demand_positive_count',
        'interaction_demand_negative_count',
        'interaction_demand_positive_score_sum',
        'interaction_demand_negative_score_sum',
    }
    _GATE_COUNTS = {
        'interaction_gate_tp_count',
        'interaction_gate_tn_count',
        'interaction_gate_positive_count',
        'interaction_gate_negative_count',
        'interaction_gate_positive_score_sum',
        'interaction_gate_negative_score_sum',
    }

    def add_batch_metrics(self, batch_metrics):
        for key, value in batch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_batch_mean(self):
        return self.get_epoch_mean()

    def reset(self):
        for key in self.metrics:
            self.metrics[key] = []

    def get_epoch_mean(self):
        result = {}
        internal = {
            name for pair in self._POOLED_RMSE.values() for name in pair
        } | self._DEMAND_COUNTS | self._GATE_COUNTS
        for key, values in self.metrics.items():
            if key in internal:
                continue
            if key in self._POOLED_RMSE:
                sse_key, count_key = self._POOLED_RMSE[key]
                total_sse = float(np.sum(self.metrics[sse_key]))
                total_count = float(np.sum(self.metrics[count_key]))
                result[key] = float(np.sqrt(total_sse / max(total_count, 1.0)))
            else:
                result[key] = float(np.mean(values)) if values else 0.0
        result['evaluated_prediction_points'] = float(
            np.sum(self.metrics['prediction_count'])
        )
        result['evaluated_update_points'] = float(
            np.sum(self.metrics['update_count'])
        )
        demand_positive = float(np.sum(
            self.metrics['interaction_demand_positive_count']
        ))
        demand_negative = float(np.sum(
            self.metrics['interaction_demand_negative_count']
        ))
        demand_tp = float(np.sum(self.metrics['interaction_demand_tp_count']))
        demand_tn = float(np.sum(self.metrics['interaction_demand_tn_count']))
        positive_score_sum = float(np.sum(
            self.metrics['interaction_demand_positive_score_sum']
        ))
        negative_score_sum = float(np.sum(
            self.metrics['interaction_demand_negative_score_sum']
        ))
        demand_total = demand_positive + demand_negative
        if demand_total > 0:
            result['interaction_demand_positive_fraction'] = (
                demand_positive / demand_total
            )
            result['interaction_demand_accuracy'] = (
                demand_tp + demand_tn
            ) / demand_total
            positive_recall = demand_tp / max(demand_positive, 1.0)
            negative_recall = demand_tn / max(demand_negative, 1.0)
            result['interaction_demand_balanced_accuracy'] = 0.5 * (
                positive_recall + negative_recall
            )
            positive_score_mean = positive_score_sum / max(demand_positive, 1.0)
            negative_score_mean = negative_score_sum / max(demand_negative, 1.0)
            result['interaction_demand_positive_score_mean'] = positive_score_mean
            result['interaction_demand_negative_score_mean'] = negative_score_mean
            result['interaction_demand_score_separation'] = (
                positive_score_mean - negative_score_mean
            )
        gate_positive = float(np.sum(
            self.metrics['interaction_gate_positive_count']
        ))
        gate_negative = float(np.sum(
            self.metrics['interaction_gate_negative_count']
        ))
        gate_total = gate_positive + gate_negative
        if gate_total > 0:
            gate_tp = float(np.sum(self.metrics['interaction_gate_tp_count']))
            gate_tn = float(np.sum(self.metrics['interaction_gate_tn_count']))
            gate_positive_score = float(np.sum(
                self.metrics['interaction_gate_positive_score_sum']
            )) / max(gate_positive, 1.0)
            gate_negative_score = float(np.sum(
                self.metrics['interaction_gate_negative_score_sum']
            )) / max(gate_negative, 1.0)
            result['interaction_gate_balanced_accuracy'] = 0.5 * (
                gate_tp / max(gate_positive, 1.0)
                + gate_tn / max(gate_negative, 1.0)
            )
            result['interaction_gate_positive_score_mean'] = gate_positive_score
            result['interaction_gate_negative_score_mean'] = gate_negative_score
            result['interaction_gate_score_separation'] = (
                gate_positive_score - gate_negative_score
            )
        return result

    def get_all_values(self, key):
        return self.metrics.get(key, [])
