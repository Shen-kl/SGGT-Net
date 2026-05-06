import torch
import torch.nn as nn
from model.gru_gnn_residual import *

'''
 模型信息：使用残差对预测结果进行修正 
'''


class SGGT_Net(nn.Module):
    def __init__(self, encoder_input_size, encoder_hidden_size, encoder_n_heads, encoder_n_layers,
                 encoder_n_mixtures, encoder_dropout, encoder_gnn_layer,
                 encoder_use_edge_features, decoder_motion_model, decoder_max_length, decoder_hidden_size,
                 decoder_n_heads, decoder_n_layers, decoder_alpha, decoder_dropout, decoder_residual_length,
                 decoder_z_dimension,
                 decoder_gnn_layer, decoder_use_MCU, decoder_use_struct, delta_T):
        super().__init__()
        self.encoder = GRUGNNEncoder(encoder_input_size, encoder_hidden_size, encoder_n_heads, encoder_n_layers,
                                     encoder_n_mixtures, encoder_dropout, encoder_gnn_layer,
                                     encoder_use_edge_features)
        self.decoder = GRUGNNDecoder(decoder_motion_model, delta_T, decoder_max_length, decoder_hidden_size,
                                     decoder_n_heads, decoder_n_layers, decoder_alpha, decoder_dropout,
                                     decoder_residual_length, decoder_z_dimension,
                                     decoder_gnn_layer, decoder_use_MCU, decoder_use_struct)

        

        self.initial_uncertainty = 1e-3
        self.initial_uncertainty_measurement = 1e-4
        self.time_update = self.ekf
        self.register_buffer("H", (torch.zeros(2, 4)))
        self.H[0, 0] = self.H[1, 1] = 1
        self.delta_T = delta_T

    def ekf(self, P_t, q_t, next_states, model_input, T, static_f):
        """Performs the time update of the state covariance estimate
        using standard EKF approach"""
        batch_size = model_input.size(0)


        # F_t, F_t_transpose = self.decoder.motion_model.state_transition_matrix(
        #     next_states, model_input, static_f)
        G_t, G_t_transpose = self.decoder.motion_model.input_transition_matrix(
            next_states, model_input)
        D = torch.diag_embed(torch.tensor([CONFIG['S_POS'],CONFIG['S_POS'],CONFIG['S_VEL'],CONFIG['S_VEL']]))
        D_inv = torch.inverse(D)
        D_inv_transpose = torch.transpose(D_inv, dim0=-2, dim1=-1)


        F_cv_t = torch.tensor(
            [[1, 0, T , 0], [0, 1, 0, T ],
             [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat([batch_size, 1, 1, 1])
        F_cv_t_bar = D_inv @ F_cv_t @ D
        F_t = F_cv_t_bar
        F_t_transpose = torch.transpose(F_cv_t_bar, dim0=-2, dim1=-1)

        Q_t = D_inv @ G_t @ q_t @ G_t_transpose @ D_inv_transpose

        # print("F_t max",F_t.abs().max().item())

        Q_t =  D_inv @ G_t @ q_t @ G_t_transpose @ D_inv_transpose
        Q_t = (Q_t + Q_t.transpose(-2,-1)) / 2
        # with torch.no_grad():
        #     Q_t *= 0.1

        P_t_next = F_t @ P_t @ F_t_transpose + Q_t
        P_t_next = (P_t_next + P_t_next.transpose(-2,-1)) / 2
        P_t_next = P_t_next + 1e-7 * torch.eye(P_t_next.shape[-1], device=P_t_next.device)

        return P_t_next, Q_t

    def forward(self, graph_data, tf_prob=0.0, P_t=None, encoder_hidden=None, R=None):
        # 取出数据
        target_tensor = graph_data.y
        input_tensor = graph_data.x
        edge_index = graph_data.edge_index
        tar_edge_index = graph_data.tar_edge_index
        tar_edge_features = graph_data.tar_edge_features
        target_length = target_tensor.size(1)
        batch_size = input_tensor.size(0)
        batch = graph_data.batch
        residual = graph_data.residual
        measurement_future = graph_data.measurement_future

        # 初始化值
        mixtures = self.decoder.mixtures  # 生成mixtures个 高斯混合分量
        n_states = self.decoder.motion_model.n_states
        target = target_tensor[:, :, :n_states]
        n_measurements = measurement_future.shape[-1]

        if P_t is None:
            P_t = torch.diag_embed(torch.ones(batch_size, mixtures, n_states,
                                              device=input_tensor.device)
                                   ) * self.initial_uncertainty  # * torch.tensor([1,1,1e1*(CONFIG['S_POS']/CONFIG['S_VEL'])**2,1e1*(CONFIG['S_POS']/CONFIG['S_VEL'])**2]) # (batch_size, mixtures, n_states, n_states)
        else:
            P_t = P_t.unsqueeze(dim=1).repeat(1, mixtures, 1, 1).detach()


        # 输入编码器
        encoder_output, mixture_coeffs = self.encoder(graph_data, encoder_hidden)
        encoder_hidden_output = encoder_output[:, -1].detach()

        decoder_hidden = self.decoder.get_initial_state(encoder_output[:, -1], graph_data)

        decoder_input = torch.zeros(batch_size, mixtures * n_states,
                                    device=input_tensor.device)

        # 调整  decoder_input 的输入
        # decoder_input = input_tensor[:,-1,:].clone().repeat(1, mixtures)

        past_state = input_tensor[:, -1:, :n_states]
        past_state = past_state.expand(-1, mixtures, -1)

        real_mask = graph_data.real_mask[..., 0].to(torch.float32)  # 1 where obs exists
        use_teacher_forcing = (tf_prob > 0.) and (torch.FloatTensor(
            1).uniform_(0, 1) < tf_prob)

        # Roll out decoding
        pred_states = []
        Ps = []
        A_vec_set = []
        A_vec_mask_output_set = []
        for di in range(target_length):
            dec_in = (decoder_input, decoder_hidden, encoder_output,
                      tar_edge_index[di], tar_edge_features[di], past_state, batch,
                      real_mask.unsqueeze(-1).to(torch.bool), residual)
            next_states, model_input, q_t, decoder_hidden, A_vec, A_vec_mask_output, total_group_num, gate_final\
                = self.decoder(dec_in)
            A_vec_set.append(A_vec)
            A_vec_mask_output_set.append(A_vec_mask_output)
            #  Update estimated state covariance
            P_t, Q_t = self.time_update(P_t, q_t, past_state, model_input, self.delta_T,
                                        static_f=torch.zeros(batch_size, mixtures, n_states))

            pred_states.append(next_states)
            Ps.append(P_t)

            prediction_det = next_states.detach()
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                ri_mask = real_mask[:, di].view(-1, 1, 1)  # (B,)
                teacher_pred = target[:, di, :].unsqueeze(dim=1).expand(
                    -1, mixtures, -1)  # (B, N_mixture, d)

                # Keep prediction when no teacher prediction exists
                past_state = ri_mask * teacher_pred + (1. - ri_mask) * prediction_det
            else:
                # 更新
                past_state = prediction_det

            residual[:, :-1, :] = residual[:, 1:, :]
            residual[:, -1, :] = CONFIG['S_POS'] / CONFIG['S_RES'] * (
                    measurement_future[:, di, :] - prediction_det[:, 0, [0, 1]]
            )

            if di == 0:
                # 混合更新
                pi = torch.ones(next_states.shape[0], next_states.shape[1]) / next_states.shape[1]
                x_bar, P_bar = gmm_to_single_gaussian(next_states, P_t, pi, eps=1e-9)
                # 执行更新步骤
                measurement_input = torch.cat([graph_data.measurement, measurement_future[:, di, :].unsqueeze(1)],
                                              dim=1)
                x_input = torch.cat([input_tensor, x_bar.unsqueeze(dim=1)], dim=1)

                # x_input = torch.cat([input_tensor, target_tensor[:,0,:].unsqueeze(dim=1)], dim=1) # 测试用 将输入改为真值

                # 卡尔曼更新
                if R is None:
                    R = torch.diag_embed(torch.ones(batch_size, n_measurements,
                                                    device=input_tensor.device)
                                         ) * self.initial_uncertainty_measurement  # (batch_size, mixtures, n_states, n_states)
                x_update, P_update, innovation, S = kalman_update(x_bar.unsqueeze(dim=1), P_bar,
                                                   measurement_future[:, di, :].unsqueeze(1), self.H, R)

        all_states = torch.stack(pred_states, dim=1)  # (B, N_t, N_mix, d)
        all_Ps = torch.stack(Ps, dim=1)  # (B, N_t, N_mix, d, d)
        all_A_vec = torch.stack(A_vec_set, dim=0).transpose(-1,-2)
        all_A_vec_mask = torch.stack(A_vec_mask_output_set,dim=0).transpose(-1,-2)
        return (all_states, all_Ps, mixture_coeffs, real_mask, target,
                x_update, P_update, encoder_hidden, Q_t, innovation, S, all_A_vec, all_A_vec_mask, gate_final)

    def predict(self, graph_data):
        '''
        实际跟踪中的预测
        :param graph_data:
        :return:
        '''
        input_tensor = graph_data.x
        batch_size = input_tensor.size(0)
        batch = self.connected_components(graph_data.edge_index[-1], num_nodes=graph_data.x.shape[0])
        mixtures = self.decoder.mixtures  # 生成mixtures个 高斯混合分量
        n_states = self.decoder.motion_model.n_states
        residual = graph_data.residual
        P_t = graph_data.P_scaler

        # 输入编码器
        encoder_output, mixture_coeffs = self.encoder(graph_data)

        decoder_hidden = self.decoder.get_initial_state(encoder_output[:, -1], graph_data)

        # decoder_input = torch.zeros(batch_size, mixtures * n_states,
        #                             device=input_tensor.device)

        # 调整  decoder_input 的输入
        decoder_input = input_tensor[:, -1, :].clone().repeat(1, mixtures)

        past_state = input_tensor[:, -1:, :n_states]
        past_state = past_state.expand(-1, mixtures, -1)

        # Roll out decoding
        pred_states = []

        tar_edge_index = graph_data.edge_index[-1]
        tar_edge_features = graph_data.edge_features[-1]
        nan_mask = graph_data.nan_mask
        dec_in = (decoder_input, decoder_hidden, encoder_output,
                  tar_edge_index, tar_edge_features, past_state, batch, nan_mask, residual)
        next_states, model_input, q_t, decoder_hidden, A_vec, A_vec_mask_output, total_group_num, _\
            = self.decoder(dec_in)

        P_t, Q_t = self.time_update(P_t, q_t, past_state, model_input, self.delta_T,
                               static_f=torch.zeros(batch_size, mixtures, n_states))

        pred_states.append(next_states)

        prediction_det = next_states.detach()

        all_states = torch.stack(pred_states, dim=1)  # (B, N_t, N_mix, d)

        return all_states, P_t, total_group_num

    def update(self, state_input, P_input, measurement_input, R=None):
        '''
        实际跟踪中的更新
        :param state_input:
        :param measurement_input:
        :return:
        '''
        # 卡尔曼更新
        n_measurements = measurement_input.shape[-1]
        batch_size = measurement_input.shape[0]
        if R is None:
            R = torch.diag_embed(torch.ones(batch_size, n_measurements,
                                        device=state_input.device)
                             ) * self.initial_uncertainty_measurement  # (batch_size, mixtures, n_states, n_states)
        x_update, P_update, _, _ = kalman_update(state_input, P_input, measurement_input,
                                           self.H, R)
        return x_update, P_update

    def connected_components(self, edge_index, num_nodes):
        """
        edge_index: [2, E]
        num_nodes: int
        return:
            group_id: [1, num_nodes], 群编号从 0 开始
        """

        parent = list(range(num_nodes))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # union all edges
        src, dst = edge_index
        for u, v in zip(src.tolist(), dst.tolist()):
            union(u, v)

        # compress & re-index
        root_to_group = {}
        group_id = torch.zeros(num_nodes, dtype=torch.long)

        gid = 0
        for i in range(num_nodes):
            r = find(i)
            if r not in root_to_group:
                root_to_group[r] = gid
                gid += 1
            group_id[i] = root_to_group[r]

        return group_id  # [1, num_nodes]


def gmm_to_single_gaussian(x, P, pi, eps=1e-9):
    """
    x:  (B, M, dx)
    P:  (B, M, dx, dx)
    pi: (B, M)  (logits or probs)
    return:
        x_bar: (B, dx)
        P_bar: (B, dx, dx)
    """
    # normalize pi
    pi = torch.softmax(pi, dim=-1) if pi.min() < 0 or pi.max() > 1 else pi
    pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)

    # mean
    x_bar = torch.sum(pi.unsqueeze(-1) * x, dim=1)  # (B, dx)

    # covariance
    dx = x.shape[-1]
    diff = x - x_bar.unsqueeze(1)  # (B, M, dx)
    outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)  # (B, M, dx, dx)

    P_bar = torch.sum(pi.unsqueeze(-1).unsqueeze(-1) * (P + outer), dim=1)  # (B, dx, dx)

    return x_bar, P_bar


def kalman_update(X_pred, P_pred, Z, H, R):
    """
    P_pred: (B, dx, dx)  or (dx, dx)
    K:      (B, dx, dz)  or (dx, dz)
    H:      (B, dz, dx)  or (dz, dx)
    R:      (B, dz, dz)  or (dz, dz)
    """
    dx = P_pred.shape[-1]
    batch = P_pred.shape[0]
    I = torch.eye(dx, device=P_pred.device, dtype=P_pred.dtype)

    D = torch.diag_embed(torch.tensor([CONFIG['S_POS'], CONFIG['S_POS'], CONFIG['S_VEL'], CONFIG['S_VEL']]))
    H_bar = H @ D

    R_renorm = R * (CONFIG['S_POS'])**2


    innovation = CONFIG['S_POS'] * (Z -  X_pred @ H.T)

    H_batch = H.unsqueeze(0).expand(batch, -1, -1)  # (B,dz,dx)
    H_bar_batch = H_bar.unsqueeze(0).expand(batch, -1, -1)
    # S = HPH^T + R
    HP = torch.bmm(H_bar_batch, P_pred)  # (B,dz,dx)
    S = torch.bmm(HP, H_bar_batch.transpose(1, 2)) + R_renorm  # (B,dz,dz)

    # K = P H^T S^{-1}
    PHt = torch.bmm(P_pred, H_bar_batch.transpose(1, 2))  # (B,dx,dz)

    # Solve S * X = PHt^T  => X = S^{-1} PHt^T
    # then K = X^T
    X = torch.linalg.solve(S, PHt.transpose(1, 2))  # (B,dz,dx)
    K = X.transpose(1, 2).clone()
    # K[:, [2, 3], :] = K[:, [2, 3], :] * CONFIG['S_POS'] / CONFIG['S_VEL']

    if P_pred.dim() == 3:
        I = I.unsqueeze(0).expand(P_pred.shape[0], -1, -1)

    dx = torch.bmm(K, innovation.permute(0, 2, 1)).squeeze(-1)
    #dx = torch.clamp(dx,-1,1)
    X_update = X_pred + dx.unsqueeze(1)
    A = I - K @ H_bar
    P_upd = A @ P_pred @ A.transpose(-1, -2) + K @ R_renorm @ K.transpose(-1, -2)
    P_upd = (P_upd + P_upd.transpose(-2, -1)) / 2

    return X_update, P_upd, innovation, S