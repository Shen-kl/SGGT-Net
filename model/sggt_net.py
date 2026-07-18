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
from model.kalman_utils import ExtendedKalmanFilter, gmm_to_single_gaussian


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
    ):
        super().__init__()
        self.model_variant = model_variant
        self.covariance_transition = covariance_transition
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
        self.decoder = decoder_type(
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
        self.ekf_filter = ExtendedKalmanFilter(delta_T)
        self.initial_uncertainty = 1e-3
        self.initial_uncertainty_measurement = 1e-4

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
        )

    def forward(self, graph_data, tf_prob=0.0, P_t=None, encoder_hidden=None, R=None):
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

        encoder_output, mixture_coeffs = self.encoder(graph_data, encoder_hidden)
        encoder_hidden_output = encoder_output[:, -1].detach()
        decoder_hidden = self.decoder.get_initial_state(encoder_output[:, -1], graph_data)
        decoder_input = torch.zeros(node_count, mixtures * n_states, device=input_tensor.device)
        past_state = input_tensor[:, -1:, :n_states].expand(-1, mixtures, -1)
        use_teacher_forcing = tf_prob > 0.0 and torch.rand((), device=input_tensor.device) < tf_prob

        states, covariances, adjacency, adjacency_mask = [], [], [], []
        spectral_weights, node_entropy, auxiliary_delta_v, film_gate, group_count = [], [], [], [], []
        residual = graph_data.residual.clone()
        x_update = input_tensor[:, -1:, :n_states]
        P_update = P_t[:, 0]

        for step in range(target_length):
            decoded = self._decode((
                decoder_input,
                decoder_hidden,
                encoder_output,
                graph_data.tar_edge_index[step],
                graph_data.tar_edge_features[step],
                past_state,
                graph_data.batch,
                real_mask.unsqueeze(-1).bool(),
                residual,
            ))
            decoder_hidden = decoded.hidden
            F_t = self._flow_jacobian(past_state, decoded.model_input)
            G_t, _ = self.decoder.motion_model.input_transition_matrix(
                past_state, decoded.model_input
            )
            P_t, _ = self.ekf_filter.time_update(
                P_t, G_t, decoded.process_noise, input_tensor.device, F_t=F_t
            )

            states.append(decoded.next_state)
            covariances.append(P_t)
            adjacency.append(decoded.adjacency_vector)
            adjacency_mask.append(decoded.adjacency_mask)
            spectral_weights.append(decoded.spectral_weights)
            node_entropy.append(decoded.node_entropy)
            auxiliary_delta_v.append(decoded.auxiliary_delta_v)
            film_gate.append(decoded.film_gate)
            group_count.append(torch.as_tensor(decoded.group_count, device=input_tensor.device))

            prediction_detached = decoded.next_state.detach()
            if use_teacher_forcing:
                valid = real_mask[:, step].view(-1, 1, 1)
                teacher = target[:, step].unsqueeze(1).expand(-1, mixtures, -1)
                past_state = valid * teacher + (1.0 - valid) * prediction_detached
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
                x_update, P_update, _, _ = self.ekf_filter.measurement_update(
                    x_bar.unsqueeze(1),
                    P_bar,
                    graph_data.measurement_future[:, step].unsqueeze(1),
                    R,
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
        )

    def predict(self, graph_data):
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
        decoded = self._decode((
            input_tensor[:, -1].clone().repeat(1, mixtures),
            hidden,
            encoder_output,
            graph_data.edge_index[-1],
            graph_data.edge_features[-1],
            past_state,
            batch,
            graph_data.nan_mask,
            graph_data.residual,
        ))
        G_t, _ = self.decoder.motion_model.input_transition_matrix(past_state, decoded.model_input)
        F_t = self._flow_jacobian(past_state, decoded.model_input)
        P_t, _ = self.ekf_filter.time_update(P_t, G_t, decoded.process_noise, input_tensor.device, F_t=F_t)
        return decoded.next_state.unsqueeze(1), P_t, decoded.group_count

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
