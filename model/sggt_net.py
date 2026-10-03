from typing import NamedTuple

import torch
import torch.nn as nn

from config import CONFIG
from model.gru_gnn_residual import (
    DecoderV2Output,
    GRUGNNDecoder,
    GRUGNNDecoderV2,
    GRUGNNEncoder,
)
from model.kalman_utils import (
    CovarianceUtils,
    ExtendedKalmanFilter,
    gmm_to_single_gaussian,
)
from model.causal_innovation import CausalGroupInnovationCorrector
from model.association_uncertainty import CausalAssociationAugmentor
from model.sggt_v2_modules import SpectralFilterState


class SGGTOutput(NamedTuple):
    """Model output with the first eleven entries kept V1-index compatible."""

    states: torch.Tensor
    covariances: torch.Tensor
    mixture_coeffs: torch.Tensor
    real_mask: torch.Tensor
    target: torch.Tensor
    updated_state: torch.Tensor
    updated_covariance: torch.Tensor
    encoder_hidden: torch.Tensor
    adjacency: torch.Tensor
    adjacency_mask: torch.Tensor
    gate: torch.Tensor
    spectral_weights: torch.Tensor
    node_entropy: torch.Tensor
    auxiliary_delta_v: torch.Tensor
    film_gate: torch.Tensor
    group_count: torch.Tensor
    counterfactual_base: torch.Tensor
    counterfactual_expert: torch.Tensor
    encoder_output_cache: torch.Tensor
    decoder_hidden_cache: torch.Tensor
    mixture_coeffs_cache: torch.Tensor
    policy_gate: torch.Tensor
    gasf_confidence: torch.Tensor
    q_acceleration_multiplier: torch.Tensor
    q_position_noise: torch.Tensor
    q_innovation_score: torch.Tensor
    auxiliary_delta_v_temporal: torch.Tensor
    auxiliary_delta_v_expert: torch.Tensor
    measurement_innovation: torch.Tensor
    innovation_covariance: torch.Tensor
    whitened_innovation: torch.Tensor
    nis: torch.Tensor
    causal_innovation_correction: torch.Tensor
    causal_innovation_eligible: torch.Tensor
    causal_innovation_activation: torch.Tensor
    causal_innovation_demand: torch.Tensor
    association_detection_mass: torch.Tensor
    association_entropy: torch.Tensor
    association_spread: torch.Tensor
    association_valid: torch.Tensor
    gasf_recurrent_state: SpectralFilterState | None
    gasf_parameter_update_gates: torch.Tensor


class SGGT_Net(nn.Module):
    def __init__(
        self,
        encoder_input_size,
        encoder_hidden_size,
        encoder_n_heads,
        encoder_n_layers,
        encoder_n_mixtures,
        encoder_dropout,
        encoder_gnn_layer,
        encoder_use_edge_features,
        decoder_motion_model,
        decoder_max_length,
        decoder_hidden_size,
        decoder_n_heads,
        decoder_n_layers,
        decoder_alpha,
        decoder_dropout,
        decoder_residual_length,
        decoder_z_dimension,
        decoder_gnn_layer,
        decoder_use_GASF,
        decoder_use_SEAN,
        delta_T,
        model_variant="sggt_v2",
        covariance_transition="ode_jacobian",
        gasf_filter_mode="abc",
        gasf_graph_mode="graph",
        counterfactual_training=False,
        adaptive_q_mode="legacy",
        adaptive_q_max_multiplier=5.0,
        adaptive_q_max_position_variance=400.0,
        adaptive_q_initial_multiplier=1.0,
        adaptive_q_initial_position_variance=0.01,
        adaptive_q_acceleration=True,
        adaptive_q_tail_boost=0.0,
        adaptive_q_tail_threshold=0.8,
        innovation_decomposition=False,
        innovation_whiten_clip=20.0,
        gasf_recurrent_filter=False,
        gasf_recurrent_gate_initial_update=0.1,
        causal_innovation_correction=False,
        causal_innovation_history=8,
        causal_innovation_hidden_size=32,
        causal_innovation_output_mode="acceleration",
        causal_innovation_max_acceleration_mps2=5.0,
        causal_innovation_max_position_m=50.0,
        causal_innovation_max_velocity_mps=5.0,
        causal_innovation_dynamic_activation=False,
        causal_innovation_activation_floor=0.5,
        causal_innovation_activation_nis_low=1.38629436112,
        causal_innovation_activation_nis_high=5.99146454711,
        causal_innovation_minimum_valid_history=None,
        causal_innovation_recent_valid_window=3,
        causal_innovation_association_context=False,
        causal_innovation_use_peer=True,
        association_augmentation=False,
        association_pd_min=0.75,
        association_pd_max=0.98,
        association_clutter_mu_min=0.0,
        association_clutter_mu_max=0.0,
        association_surveillance_side_m=3000.0,
        association_max_clutter=32,
        association_local_clutter_rate=0.25,
        association_max_local_clutter=2,
        association_local_clutter_scale=2.5,
        association_gate_threshold=9.21034037198,
        association_minimum_detection_mass=0.05,
        association_include_target_competitors=False,
        association_clutter_requires_detection=True,
    ):
        super().__init__()
        self.model_variant = model_variant
        self.covariance_transition = covariance_transition
        self.counterfactual_training = bool(counterfactual_training)
        self.counterfactual_evaluation = False
        self.encoder = GRUGNNEncoder(
            int(encoder_input_size),
            int(encoder_hidden_size),
            int(encoder_n_heads),
            int(encoder_n_layers),
            int(encoder_n_mixtures),
            encoder_dropout,
            encoder_gnn_layer,
            encoder_use_edge_features,
        )
        decoder_type = GRUGNNDecoderV2 if model_variant == "sggt_v2" else GRUGNNDecoder
        decoder_args = (
            decoder_motion_model,
            delta_T,
            int(decoder_max_length),
            int(decoder_hidden_size),
            int(decoder_n_heads),
            int(decoder_n_layers),
            decoder_alpha,
            decoder_dropout,
            decoder_residual_length,
            int(decoder_z_dimension),
            decoder_gnn_layer,
            decoder_use_GASF,
            decoder_use_SEAN,
        )
        if model_variant == "sggt_v2":
            self.decoder = decoder_type(
                *decoder_args,
                gasf_filter_mode=gasf_filter_mode,
                gasf_graph_mode=gasf_graph_mode,
                adaptive_q_mode=adaptive_q_mode,
                adaptive_q_max_multiplier=adaptive_q_max_multiplier,
                adaptive_q_max_position_variance=adaptive_q_max_position_variance,
                adaptive_q_initial_multiplier=adaptive_q_initial_multiplier,
                adaptive_q_initial_position_variance=(
                    adaptive_q_initial_position_variance
                ),
                adaptive_q_acceleration=adaptive_q_acceleration,
                adaptive_q_tail_boost=adaptive_q_tail_boost,
                adaptive_q_tail_threshold=adaptive_q_tail_threshold,
                innovation_decomposition=innovation_decomposition,
                innovation_whiten_clip=innovation_whiten_clip,
                gasf_recurrent_filter=gasf_recurrent_filter,
                gasf_recurrent_gate_initial_update=(
                    gasf_recurrent_gate_initial_update
                ),
            )
        else:
            self.decoder = decoder_type(*decoder_args)
        self.ekf_filter = ExtendedKalmanFilter(delta_T)
        self.initial_uncertainty = 1e-3
        self.initial_uncertainty_measurement = 1e-4
        # Evaluation-only counterfactual controls. The default preserves the
        # learned process noise and therefore does not alter training.
        self.process_noise_mode = "learned"
        self.process_noise_scale = 1.0
        self.fixed_process_noise = None
        self.last_process_noise_raw = None
        self.last_process_noise_used = None
        self.last_state_process_noise = None
        self.innovation_whiten_clip = float(innovation_whiten_clip)
        self.association_augmentation_enabled = bool(association_augmentation)
        self.association_augmentation_active = False
        self.association_augmentor = CausalAssociationAugmentor(
            pd_min=association_pd_min,
            pd_max=association_pd_max,
            clutter_mu_min=association_clutter_mu_min,
            clutter_mu_max=association_clutter_mu_max,
            surveillance_side_m=association_surveillance_side_m,
            max_clutter=association_max_clutter,
            local_clutter_rate=association_local_clutter_rate,
            max_local_clutter=association_max_local_clutter,
            local_clutter_scale=association_local_clutter_scale,
            gate_threshold=association_gate_threshold,
            minimum_detection_mass=association_minimum_detection_mass,
            include_target_competitors=association_include_target_competitors,
            clutter_requires_detection=association_clutter_requires_detection,
        )
        self.causal_innovation_correction = bool(causal_innovation_correction)
        self.causal_innovation_output_mode = str(
            causal_innovation_output_mode
        )
        self.causal_innovation_dynamic_activation = bool(
            causal_innovation_dynamic_activation
        )
        if self.causal_innovation_correction:
            if model_variant != "sggt_v2":
                raise ValueError("causal innovation correction requires SGGT-V2")
            if self.causal_innovation_dynamic_activation:
                raise ValueError(
                    "CIPIC correction has no confidence gate"
                )
            if causal_innovation_association_context:
                raise ValueError(
                    "CIPIC correction uses raw own/peer histories only"
                )
            self.decoder.motion_model.causal_innovation_corrector = (
                CausalGroupInnovationCorrector(
                history_length=causal_innovation_history,
                hidden_size=causal_innovation_hidden_size,
                raw_scale=CONFIG['S_RES'],
                output_mode=self.causal_innovation_output_mode,
                max_acceleration_mps2=(
                    causal_innovation_max_acceleration_mps2
                ),
                max_position_m=causal_innovation_max_position_m,
                max_velocity_mps=causal_innovation_max_velocity_mps,
                position_scale=CONFIG['S_POS'],
                velocity_scale=CONFIG['S_VEL'],
                minimum_valid_history=causal_innovation_minimum_valid_history,
                recent_valid_window=causal_innovation_recent_valid_window,
                use_peer_innovation=causal_innovation_use_peer,
                )
            )

    @property
    def causal_innovation_corrector(self):
        """Expose the motion-model-owned corrector to staged trainers."""
        return self.decoder.motion_model.causal_innovation_corrector

    @staticmethod
    def _innovation_bundle(graph_data, residual):
        """Return optional online innovation histories with safe defaults."""
        node_count, length, dimension = residual.shape
        reference = residual
        identity = torch.eye(
            dimension, dtype=reference.dtype, device=reference.device
        ).view(1, 1, dimension, dimension)
        return {
            'raw': getattr(
                graph_data,
                'innovation_raw',
                reference.new_zeros(node_count, length, dimension),
            ),
            'whitened': getattr(
                graph_data,
                'innovation_whitened',
                reference.new_zeros(node_count, length, dimension),
            ),
            'peer_raw': getattr(
                graph_data,
                'innovation_peer_raw',
                reference.new_zeros(node_count, length, dimension),
            ),
            'peer_whitened': getattr(
                graph_data,
                'innovation_peer_whitened',
                reference.new_zeros(node_count, length, dimension),
            ),
            'covariance': getattr(
                graph_data,
                'innovation_covariance',
                identity.expand(node_count, length, -1, -1).clone(),
            ),
            'nis': getattr(
                graph_data,
                'nis_history',
                reference.new_zeros(node_count, length, 1),
            ),
            'valid': getattr(
                graph_data,
                'innovation_valid',
                torch.zeros(
                    node_count,
                    length,
                    1,
                    dtype=torch.bool,
                    device=reference.device,
                ),
            ),
            'peer_valid': getattr(
                graph_data,
                'innovation_peer_valid',
                torch.zeros(
                    node_count,
                    length,
                    1,
                    dtype=torch.bool,
                    device=reference.device,
                ),
            ),
            'association_detection_mass': getattr(
                graph_data,
                'association_detection_mass',
                reference.new_zeros(node_count, length, 1),
            ),
            'association_entropy': getattr(
                graph_data,
                'association_entropy',
                reference.new_zeros(node_count, length, 1),
            ),
            'association_spread': getattr(
                graph_data,
                'association_spread',
                reference.new_zeros(node_count, length, dimension, dimension),
            ),
        }

    def _select_process_noise(self, learned_noise):
        mode = self.process_noise_mode
        if mode == "learned":
            return learned_noise
        if mode == "scaled":
            return learned_noise * float(self.process_noise_scale)
        if mode == "zero":
            return torch.zeros_like(learned_noise)
        if mode == "fixed":
            if self.fixed_process_noise is None:
                raise ValueError("fixed_process_noise is required in fixed mode")
            fixed = self.fixed_process_noise.to(learned_noise)
            while fixed.ndim < learned_noise.ndim:
                fixed = fixed.unsqueeze(0)
            return fixed.expand_as(learned_noise)
        raise ValueError(f"Unsupported process_noise_mode: {mode}")

    def _flow_jacobian(self, past_state, model_input):
        if self.model_variant != "sggt_v2" or self.covariance_transition != "ode_jacobian":
            return None
        F_t, _ = self.decoder.motion_model.state_transition_matrix(
            past_state, model_input, static_f=None
        )
        # Training through F would require costly second derivatives.  The state
        # prediction and process-noise paths remain fully differentiable.
        return F_t.detach()

    def _decode(self, decoder_input):
        decoded = self.decoder(decoder_input)
        if isinstance(decoded, DecoderV2Output):
            return decoded
        next_state, model_input, process_noise, hidden, adjacency, mask, groups, gate = decoded
        node_count = next_state.size(0)
        device = next_state.device
        return DecoderV2Output(
            next_state,
            model_input,
            process_noise,
            hidden,
            adjacency,
            mask,
            groups,
            next_state.new_zeros(node_count, 0),
            next_state.new_ones(node_count, 1),
            next_state.new_zeros(node_count, 2),
            gate if torch.is_tensor(gate) else next_state.new_zeros(node_count, 1),
            next_state.new_zeros(node_count, 1),
            next_state.new_ones(node_count, 1),
            next_state.new_zeros(node_count, 1, 4, 4),
            next_state.new_zeros(node_count, 1),
            next_state.new_zeros(node_count, 1),
            next_state.new_zeros(node_count, 2),
            next_state.new_zeros(node_count, 2),
            None,
            next_state.new_zeros(node_count, 0),
        )

    def forward(
        self, graph_data, tf_prob=0.0, P_t=None, encoder_hidden=None, R=None,
        encoder_output_cache=None, decoder_hidden_cache=None,
        mixture_coeffs_cache=None,
        gasf_recurrent_state=None,
    ):
        target_tensor = graph_data.y
        input_tensor = graph_data.x
        target_length = target_tensor.size(1)
        node_count = input_tensor.size(0)
        mixtures = self.decoder.mixtures
        n_states = self.decoder.motion_model.n_states
        target = target_tensor[:, :, :n_states]
        real_mask = graph_data.real_mask[..., 0].float()

        if P_t is None:
            P_t = torch.diag_embed(
                torch.ones(node_count, mixtures, n_states, device=input_tensor.device)
            ) * self.initial_uncertainty
        elif P_t.ndim == 3:
            P_t = P_t.unsqueeze(1).repeat(1, mixtures, 1, 1).detach()

        if encoder_output_cache is None:
            encoder_output, mixture_coeffs = self.encoder(graph_data, encoder_hidden)
        else:
            encoder_output = encoder_output_cache
            mixture_coeffs = mixture_coeffs_cache
            if mixture_coeffs is None:
                raise ValueError("mixture_coeffs_cache is required with encoder_output_cache")
        encoder_hidden_output = encoder_output[:, -1].detach()
        if decoder_hidden_cache is None:
            decoder_hidden = self.decoder.get_initial_state(
                encoder_output[:, -1], graph_data
            )
        else:
            decoder_hidden = decoder_hidden_cache
        decoder_input = torch.zeros(node_count, mixtures * n_states, device=input_tensor.device)
        past_state = input_tensor[:, -1:, :n_states].expand(-1, mixtures, -1)
        use_teacher_forcing = tf_prob > 0.0 and torch.rand((), device=input_tensor.device) < tf_prob

        states, covariances, adjacency, adjacency_mask = [], [], [], []
        counterfactual_base, counterfactual_expert = [], []
        spectral_weights, node_entropy, auxiliary_delta_v, film_gate, group_count = [], [], [], [], []
        policy_gate, gasf_confidence = [], []
        gasf_parameter_update_gates = []
        q_acceleration_multiplier, q_position_noise, q_innovation_score = [], [], []
        auxiliary_delta_v_temporal, auxiliary_delta_v_expert = [], []
        causal_corrections, causal_eligibility = [], []
        causal_activations = []
        causal_demands = []
        residual = graph_data.residual.clone()
        innovation_bundle = self._innovation_bundle(graph_data, residual)
        self.decoder.motion_model.causal_counterfactual_enabled = bool(
            self.causal_innovation_correction
            and (self.counterfactual_training or self.counterfactual_evaluation)
        )
        x_update = input_tensor[:, -1:, :n_states]
        P_update = P_t[:, 0]
        measurement_innovation = input_tensor.new_zeros(node_count, 1, 2)
        innovation_covariance = torch.eye(
            2, dtype=input_tensor.dtype, device=input_tensor.device
        ).unsqueeze(0).expand(node_count, -1, -1).clone()
        whitened_innovation = input_tensor.new_zeros(node_count, 1, 2)
        nis = input_tensor.new_zeros(node_count, 1, 1)
        association_detection_mass = input_tensor.new_ones(node_count, 1, 1)
        association_entropy = input_tensor.new_zeros(node_count, 1, 1)
        association_spread = input_tensor.new_zeros(node_count, 2, 2)
        association_valid = torch.zeros(
            node_count, 1, 1, dtype=torch.bool, device=input_tensor.device
        )
        # Multi-step decoding must only use information available at the
        # prediction origin.  Reuse the latest observed graph for every
        # decoder step instead of leaking target/future graph topology.
        observed_edge_index = graph_data.edge_index[-1]
        observed_edge_features = graph_data.edge_features[-1]
        recurrent_valid_mask = graph_data.nan_mask[:, -1].reshape(
            node_count, -1
        ).any(dim=-1, keepdim=True)

        for step in range(target_length):
            decoder_data = (
                decoder_input,
                decoder_hidden,
                encoder_output,
                observed_edge_index,
                observed_edge_features,
                past_state,
                graph_data.batch,
                real_mask.unsqueeze(-1).bool(),
                residual,
            )
            if self.model_variant == "sggt_v2":
                decoder_data = decoder_data + (
                    innovation_bundle,
                    gasf_recurrent_state,
                    recurrent_valid_mask,
                )
            decoded = self._decode(decoder_data)
            gasf_recurrent_state = decoded.gasf_recurrent_state
            if self.causal_innovation_correction:
                correction = decoded.model_input[
                    'causal_innovation_correction'
                ]
                eligible = decoded.model_input['causal_innovation_eligible']
                activation = decoded.model_input[
                    'causal_innovation_activation'
                ]
                demand = decoded.model_input['causal_innovation_demand']
                base_state = decoded.model_input.get('causal_base_state')
                if base_state is not None:
                    counterfactual_base.append(base_state)
                    counterfactual_expert.append(decoded.next_state)
                causal_corrections.append(correction)
                causal_eligibility.append(eligible)
                causal_activations.append(activation)
                causal_demands.append(demand)
            elif (
                self.counterfactual_training
                and (self.training or self.counterfactual_evaluation)
                and self.decoder.use_GASF
                and self.decoder.use_SEAN
            ):
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(input_tensor.device) if input_tensor.is_cuda else None
                activation_override = (
                    self.decoder.GASF.interaction_activation_override
                )
                gain_override = self.decoder.GASF.interaction_gain_override
                deterministic_motion = not self.decoder.motion_model.training
                # In staged expert/adaptation runs the online branch is fixed
                # exactly to one counterfactual and all frozen dropout is in
                # eval mode. Reuse that state instead of executing an identical
                # ODE call. Joint stochastic training keeps the fully paired
                # calls and RNG reset below.
                reuse_base = deterministic_motion and activation_override == 0.0
                reuse_expert = (
                    deterministic_motion
                    and activation_override == 1.0
                    and gain_override in (None, 1.0)
                )
                if reuse_base:
                    base_state = decoded.next_state
                else:
                    base_inputs = dict(decoded.model_input)
                    base_inputs['GASF_output_fusion'] = decoded.model_input[
                        'GASF_output_temporal'
                    ]
                    base_inputs['GASF_interaction_context'] = decoded.model_input[
                        'GASF_interaction_context_temporal'
                    ]
                    base_state, _ = self.decoder.motion_model(
                        past_state, base_inputs, static_f=None
                    )
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, input_tensor.device)
                if reuse_expert:
                    expert_state = decoded.next_state
                else:
                    expert_inputs = dict(decoded.model_input)
                    expert_inputs['GASF_output_fusion'] = decoded.model_input[
                        'GASF_output_interaction_expert'
                    ]
                    expert_inputs['GASF_interaction_context'] = decoded.model_input[
                        'GASF_interaction_context_expert'
                    ]
                    expert_state, _ = self.decoder.motion_model(
                        past_state, expert_inputs, static_f=None
                    )
                counterfactual_base.append(base_state)
                counterfactual_expert.append(expert_state)
            decoder_hidden = decoded.hidden
            F_t = self._flow_jacobian(past_state, decoded.model_input)
            G_t, _ = self.decoder.motion_model.input_transition_matrix(
                past_state, decoded.model_input
            )
            process_noise = self._select_process_noise(decoded.process_noise)
            P_t, state_process_noise = self.ekf_filter.time_update(
                P_t, G_t, process_noise, input_tensor.device, F_t=F_t
            )
            P_t = P_t + decoded.q_position_noise
            P_t = 0.5 * (P_t + P_t.transpose(-1, -2))
            state_process_noise = state_process_noise + decoded.q_position_noise
            self.last_process_noise_raw = decoded.process_noise.detach()
            self.last_process_noise_used = process_noise.detach()
            self.last_state_process_noise = state_process_noise.detach()

            states.append(decoded.next_state)
            covariances.append(P_t)
            adjacency.append(decoded.adjacency_vector)
            adjacency_mask.append(decoded.adjacency_mask)
            spectral_weights.append(decoded.spectral_weights)
            node_entropy.append(decoded.node_entropy)
            auxiliary_delta_v.append(decoded.auxiliary_delta_v)
            film_gate.append(decoded.film_gate)
            policy_gate.append(decoded.policy_gate)
            gasf_confidence.append(decoded.gasf_confidence)
            gasf_parameter_update_gates.append(
                decoded.gasf_parameter_update_gates
            )
            q_acceleration_multiplier.append(decoded.q_acceleration_multiplier)
            q_position_noise.append(decoded.q_position_noise)
            q_innovation_score.append(decoded.q_innovation_score)
            auxiliary_delta_v_temporal.append(decoded.auxiliary_delta_v_temporal)
            auxiliary_delta_v_expert.append(decoded.auxiliary_delta_v_expert)
            group_count.append(torch.as_tensor(decoded.group_count, device=input_tensor.device))

            prediction_detached = decoded.next_state.detach()
            if use_teacher_forcing:
                valid = real_mask[:, step].view(-1, 1, 1).bool()
                teacher = target[:, step].unsqueeze(1).expand(-1, mixtures, -1)
                past_state = torch.where(valid, teacher, prediction_detached)
            else:
                past_state = prediction_detached

            residual[:, :-1] = residual[:, 1:].clone()
            residual[:, -1] = (
                graph_data.measurement_future[:, step]
                - prediction_detached[:, 0, [0, 1]]
            ) / CONFIG['S_RES']

            if step == 0:
                pi = torch.full(
                    (node_count, mixtures), 1.0 / mixtures, device=input_tensor.device
                )
                x_bar, P_bar = gmm_to_single_gaussian(decoded.next_state, P_t, pi)
                if R is None:
                    measurement_dim = graph_data.measurement_future.size(-1)
                    R = torch.diag_embed(
                        torch.ones(node_count, measurement_dim, device=input_tensor.device)
                    ) * self.initial_uncertainty_measurement
                if self.association_augmentation_active:
                    association = self.association_augmentor.update(
                        x_bar.unsqueeze(1),
                        P_bar,
                        graph_data.measurement_future[:, step].unsqueeze(1),
                        R,
                        graph_data.batch,
                        real_mask[:, step].bool(),
                        getattr(graph_data, 'position_centroid_normalized', None),
                    )
                    x_update = association.state
                    P_update = association.covariance
                    measurement_innovation = association.innovation
                    innovation_covariance = association.innovation_covariance
                    association_detection_mass = association.detection_mass
                    association_entropy = association.entropy
                    association_spread = association.spread
                    association_valid = association.valid
                else:
                    x_update, P_update, measurement_innovation, innovation_covariance = (
                        self.ekf_filter.measurement_update(
                            x_bar.unsqueeze(1),
                            P_bar,
                            graph_data.measurement_future[:, step].unsqueeze(1),
                            R,
                        )
                    )
                # Never consume padded measurements for unborn/dead targets.
                # Birth initialization is handled by the online trajectory
                # runner; here invalid nodes retain their prior unchanged.
                update_valid = real_mask[:, step].view(-1, 1, 1).bool()
                if self.association_augmentation_active:
                    update_valid = update_valid & association_valid
                else:
                    association_valid = update_valid
                x_update = torch.where(update_valid, x_update, x_bar.unsqueeze(1))
                covariance_valid = update_valid.squeeze(-1).unsqueeze(-1)
                P_update = torch.where(covariance_valid, P_update, P_bar)
                white, nis_value, safe_s = CovarianceUtils.whiten_innovation(
                    measurement_innovation.squeeze(1),
                    innovation_covariance,
                    valid_mask=update_valid.squeeze(1),
                    max_norm=self.innovation_whiten_clip,
                )
                whitened_innovation = white.unsqueeze(1)
                nis = nis_value.unsqueeze(1)
                measurement_innovation = torch.where(
                    update_valid,
                    measurement_innovation,
                    torch.zeros_like(measurement_innovation),
                )
                covariance_valid = update_valid.squeeze(-1).unsqueeze(-1)
                identity = torch.eye(
                    safe_s.size(-1), dtype=safe_s.dtype, device=safe_s.device
                ).unsqueeze(0).expand_as(safe_s)
                innovation_covariance = torch.where(
                    covariance_valid, safe_s, identity
                )

        all_adjacency = torch.stack(adjacency, dim=0).transpose(-1, -2)
        all_adjacency_mask = torch.stack(adjacency_mask, dim=0).transpose(-1, -2)
        all_film_gate = torch.stack(film_gate, dim=1)
        return SGGTOutput(
            torch.stack(states, dim=1),
            torch.stack(covariances, dim=1),
            mixture_coeffs,
            real_mask,
            target,
            x_update,
            P_update,
            encoder_hidden_output,
            all_adjacency,
            all_adjacency_mask,
            all_film_gate,
            torch.stack(spectral_weights, dim=1),
            torch.stack(node_entropy, dim=1),
            torch.stack(auxiliary_delta_v, dim=1),
            all_film_gate,
            torch.stack(group_count),
            (torch.stack(counterfactual_base, dim=1)
             if counterfactual_base else input_tensor.new_empty(0)),
            (torch.stack(counterfactual_expert, dim=1)
             if counterfactual_expert else input_tensor.new_empty(0)),
            encoder_output,
            decoder_hidden,
            mixture_coeffs,
            torch.stack(policy_gate, dim=1),
            torch.stack(gasf_confidence, dim=1),
            torch.stack(q_acceleration_multiplier, dim=1),
            torch.stack(q_position_noise, dim=1),
            torch.stack(q_innovation_score, dim=1),
            torch.stack(auxiliary_delta_v_temporal, dim=1),
            torch.stack(auxiliary_delta_v_expert, dim=1),
            measurement_innovation,
            innovation_covariance,
            whitened_innovation,
            nis,
            (torch.stack(causal_corrections, dim=1)
             if causal_corrections else input_tensor.new_empty(0)),
            (torch.stack(causal_eligibility, dim=1)
             if causal_eligibility else torch.empty(
                 0, dtype=torch.bool, device=input_tensor.device
             )),
            (torch.stack(causal_activations, dim=1)
             if causal_activations else input_tensor.new_empty(0)),
            (torch.stack(causal_demands, dim=1)
             if causal_demands else input_tensor.new_empty(0)),
            association_detection_mass,
            association_entropy,
            association_spread,
            association_valid,
            gasf_recurrent_state,
            torch.stack(gasf_parameter_update_gates, dim=1),
        )

    def predict(
        self,
        graph_data,
        gasf_recurrent_state=None,
        return_recurrent_state=False,
    ):
        input_tensor = graph_data.x
        mixtures = self.decoder.mixtures
        n_states = self.decoder.motion_model.n_states
        batch = self.connected_components(graph_data.edge_index[-1], input_tensor.size(0)).to(input_tensor.device)
        P_t = graph_data.P_scaler
        if P_t.ndim == 3:
            P_t = P_t.unsqueeze(1).repeat(1, mixtures, 1, 1)
        encoder_output, _ = self.encoder(graph_data)
        hidden = self.decoder.get_initial_state(encoder_output[:, -1], graph_data)
        past_state = input_tensor[:, -1:, :n_states].expand(-1, mixtures, -1)
        decoder_data = (
            input_tensor[:, -1].clone().repeat(1, mixtures),
            hidden,
            encoder_output,
            graph_data.edge_index[-1],
            graph_data.edge_features[-1],
            past_state,
            batch,
            graph_data.nan_mask,
            graph_data.residual,
        )
        if self.model_variant == "sggt_v2":
            innovation_bundle = self._innovation_bundle(
                graph_data, graph_data.residual
            )
            # Legacy BP graphs do not explicitly store leave-one-out peer
            # histories. Build them causally from the graph at every historical
            # frame. Training windows already provide these tensors directly.
            if not hasattr(graph_data, 'innovation_peer_raw'):
                from train.training_utils_v3 import GraphDataProcessor

                peer_raw, peer_white, peer_valid = [], [], []
                innovation_length = graph_data.residual.size(1)
                edge_length = len(graph_data.edge_index)
                edge_offset = innovation_length - edge_length
                empty_edge = torch.empty(
                    2, 0, dtype=torch.long, device=graph_data.x.device
                )
                for frame in range(innovation_length):
                    valid = innovation_bundle['valid'][:, frame, 0].bool()
                    features = torch.cat((
                        innovation_bundle['raw'][:, frame],
                        innovation_bundle['whitened'][:, frame],
                    ), dim=-1)
                    peers, has_peer = GraphDataProcessor.leave_one_out_peer_mean(
                        features,
                        (
                            graph_data.edge_index[frame - edge_offset]
                            if frame >= edge_offset else empty_edge
                        ),
                        valid,
                    )
                    peer_raw.append(peers[:, :2])
                    peer_white.append(peers[:, 2:])
                    peer_valid.append(has_peer)
                innovation_bundle['peer_raw'] = torch.stack(peer_raw, dim=1)
                innovation_bundle['peer_whitened'] = torch.stack(
                    peer_white, dim=1
                )
                innovation_bundle['peer_valid'] = torch.stack(
                    peer_valid, dim=1
                )
            decoder_data = decoder_data + (
                innovation_bundle,
                gasf_recurrent_state,
                graph_data.nan_mask[:, -1].reshape(
                    input_tensor.size(0), -1
                ).any(dim=-1, keepdim=True),
            )
        self.decoder.motion_model.causal_counterfactual_enabled = False
        decoded = self._decode(decoder_data)
        G_t, _ = self.decoder.motion_model.input_transition_matrix(past_state, decoded.model_input)
        F_t = self._flow_jacobian(past_state, decoded.model_input)
        # Keep BP/inference covariance propagation identical to ``forward``.
        # The former implementation bypassed evaluation Q controls and omitted
        # the learned direct-position covariance branch entirely.
        process_noise = self._select_process_noise(decoded.process_noise)
        P_t, state_process_noise = self.ekf_filter.time_update(
            P_t, G_t, process_noise, input_tensor.device, F_t=F_t
        )
        P_t = P_t + decoded.q_position_noise
        P_t = 0.5 * (P_t + P_t.transpose(-1, -2))
        state_process_noise = state_process_noise + decoded.q_position_noise
        self.last_process_noise_raw = decoded.process_noise.detach()
        self.last_process_noise_used = process_noise.detach()
        self.last_state_process_noise = state_process_noise.detach()
        result = (decoded.next_state.unsqueeze(1), P_t, decoded.group_count)
        if return_recurrent_state:
            return result + (decoded.gasf_recurrent_state,)
        return result

    def update(self, state_input, P_input, measurement_input, R=None):
        if R is None:
            R = torch.diag_embed(
                torch.ones(
                    measurement_input.size(0),
                    measurement_input.size(-1),
                    device=state_input.device,
                )
            ) * self.initial_uncertainty_measurement
        state, covariance, _, _ = self.ekf_filter.measurement_update(
            state_input, P_input, measurement_input, R
        )
        return state, covariance

    @staticmethod
    def connected_components(edge_index, num_nodes):
        parent = list(range(num_nodes))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for source, target in zip(*edge_index.tolist()):
            union(source, target)
        roots = {}
        result = torch.zeros(num_nodes, dtype=torch.long)
        for node in range(num_nodes):
            root = find(node)
            roots.setdefault(root, len(roots))
            result[node] = roots[root]
        return result
