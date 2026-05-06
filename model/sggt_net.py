"""
优化后的 SGGT_Net 模型

主要改进:
1. 移除了大量注释掉的调试代码
2. 简化了 EKF 方法
3. 使用工具模块提取重复逻辑
4. 改进了代码可读性和可维护性
"""

import torch
import torch.nn as nn
from model.gru_gnn_residual import *
from model.kalman_utils import ExtendedKalmanFilter, MatrixBuilder, CovarianceUtils, gmm_to_single_gaussian

class SGGT_Net(nn.Module):
    """
    使用残差对预测结果进行修正的群目标跟踪网络
    """

    def __init__(self, encoder_input_size, encoder_hidden_size, encoder_n_heads, encoder_n_layers,
                 encoder_n_mixtures, encoder_dropout, encoder_gnn_layer,
                 encoder_use_edge_features, decoder_motion_model, decoder_max_length, decoder_hidden_size,
                 decoder_n_heads, decoder_n_layers, decoder_alpha, decoder_dropout, decoder_residual_length,
                 decoder_z_dimension,
                 decoder_gnn_layer, decoder_use_GASF, decoder_use_struct, delta_T):
        super().__init__()
        self.encoder = GRUGNNEncoder(encoder_input_size, encoder_hidden_size, encoder_n_heads, encoder_n_layers,
                                     encoder_n_mixtures, encoder_dropout, encoder_gnn_layer,
                                     encoder_use_edge_features)
        self.decoder = GRUGNNDecoder(decoder_motion_model, delta_T, decoder_max_length, decoder_hidden_size,
                                     decoder_n_heads, decoder_n_layers, decoder_alpha, decoder_dropout,
                                     decoder_residual_length, decoder_z_dimension,
                                     decoder_gnn_layer, decoder_use_GASF, decoder_use_struct)

        # 扩展卡尔曼滤波器
        self.ekf_filter = ExtendedKalmanFilter(delta_T)

        # 初始不确定性参数
        self.initial_uncertainty = 1e-3
        self.initial_uncertainty_measurement = 1e-4
        self.delta_T = delta_T

    def forward(self, graph_data, tf_prob=0.0, P_t=None, encoder_hidden=None, R=None):
        """
        前向传播 (训练阶段)

        Args:
            graph_data: 图数据
            tf_prob: Teacher forcing 概率
            P_t: 初始协方差
            encoder_hidden: 编码器隐藏状态
            R: 测量噪声协方差

        Returns:
            tuple: 包含预测状态、协方差、混合系数等的元组
        """
        # 提取数据
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

        # 初始化变量
        mixtures = self.decoder.mixtures
        n_states = self.decoder.motion_model.n_states
        target = target_tensor[:, :, :n_states]
        n_measurements = measurement_future.shape[-1]

        # 初始化协方差矩阵
        if P_t is None:
            P_t = torch.diag_embed(
                torch.ones(batch_size, mixtures, n_states, device=input_tensor.device)
            ) * self.initial_uncertainty
        else:
            P_t = P_t.unsqueeze(dim=1).repeat(1, mixtures, 1, 1).detach()

        # 编码器前向传播
        encoder_output, mixture_coeffs = self.encoder(graph_data, encoder_hidden)
        encoder_hidden_output = encoder_output[:, -1].detach()

        # 解码器初始化
        decoder_hidden = self.decoder.get_initial_state(encoder_output[:, -1], graph_data)
        decoder_input = torch.zeros(batch_size, mixtures * n_states, device=input_tensor.device)

        # 设置过去状态
        past_state = input_tensor[:, -1:, :n_states]
        past_state = past_state.expand(-1, mixtures, -1)

        # 真实掩码和教师强制
        real_mask = graph_data.real_mask[..., 0].to(torch.float32)
        use_teacher_forcing = (tf_prob > 0.) and (torch.FloatTensor(1).uniform_(0, 1) < tf_prob)

        # 解码循环
        pred_states = []
        Ps = []
        A_vec_set = []
        A_vec_mask_output_set = []

        for di in range(target_length):
            # 解码器输入
            dec_in = (
                decoder_input, decoder_hidden, encoder_output,
                tar_edge_index[di], tar_edge_features[di], past_state, batch,
                real_mask.unsqueeze(-1).to(torch.bool), residual
            )

            # 解码器前向传播
            (next_states, model_input, q_t, decoder_hidden, A_vec,
             A_vec_mask_output, total_group_num, gate_final) = self.decoder(dec_in)

            # 存储注意力向量
            A_vec_set.append(A_vec)
            A_vec_mask_output_set.append(A_vec_mask_output)

            # 更新协方差 (时间更新)
            G_t, _ = self.decoder.motion_model.input_transition_matrix(
                past_state, model_input
            )
            P_t, Q_t = self.ekf_filter.time_update(
                P_t, G_t, q_t, device=input_tensor.device
            )

            pred_states.append(next_states)
            Ps.append(P_t)

            # 更新过去状态
            prediction_det = next_states.detach()
            if use_teacher_forcing:
                ri_mask = real_mask[:, di].view(-1, 1, 1)
                teacher_pred = target[:, di, :].unsqueeze(dim=1).expand(-1, mixtures, -1)
                past_state = ri_mask * teacher_pred + (1. - ri_mask) * prediction_det
            else:
                past_state = prediction_det

            # 更新残差
            residual[:, :-1, :] = residual[:, 1:, :]
            residual[:, -1, :] = CONFIG['S_RES'] / CONFIG['S_RES'] * (
                measurement_future[:, di, :] - prediction_det[:, 0, [0, 1]]
            )

            # 在第一步执行卡尔曼测量更新
            if di == 0:
                # 混合高斯到单高斯
                pi = torch.ones(next_states.shape[0], next_states.shape[1]) / next_states.shape[1]
                x_bar, P_bar = gmm_to_single_gaussian(next_states, P_t, pi, eps=1e-9)

                # 测量输入
                measurement_input = torch.cat([
                    graph_data.measurement,
                    measurement_future[:, di, :].unsqueeze(1)
                ], dim=1)
                x_input = torch.cat([input_tensor, x_bar.unsqueeze(dim=1)], dim=1)

                # 卡尔曼测量更新
                if R is None:
                    R = torch.diag_embed(
                        torch.ones(batch_size, n_measurements, device=input_tensor.device)
                    ) * self.initial_uncertainty_measurement

                H = MatrixBuilder.build_observation_matrix(input_tensor.device)
                x_update, P_update, innovation, S = self.ekf_filter.measurement_update(
                    x_bar.unsqueeze(dim=1), P_bar,
                    measurement_future[:, di, :].unsqueeze(1), R
                )

        # 堆叠结果
        all_states = torch.stack(pred_states, dim=1)
        all_Ps = torch.stack(Ps, dim=1)
        all_A_vec = torch.stack(A_vec_set, dim=0).transpose(-1, -2)
        all_A_vec_mask = torch.stack(A_vec_mask_output_set, dim=0).transpose(-1, -2)

        return (
            all_states, all_Ps, mixture_coeffs, real_mask, target,
            x_update, P_update, encoder_hidden,
            all_A_vec, all_A_vec_mask, gate_final
        )

    def predict(self, graph_data):
        """
        实际跟踪中的预测

        Args:
            graph_data: 图数据

        Returns:
            tuple: (预测状态, 协方差, 群数量)
        """
        input_tensor = graph_data.x
        batch_size = input_tensor.size(0)
        batch = self.connected_components(graph_data.edge_index[-1], num_nodes=graph_data.x.shape[0])

        mixtures = self.decoder.mixtures
        n_states = self.decoder.motion_model.n_states
        residual = graph_data.residual
        P_t = graph_data.P_scaler

        # 编码器
        encoder_output, mixture_coeffs = self.encoder(graph_data)

        # 解码器初始化
        decoder_hidden = self.decoder.get_initial_state(encoder_output[:, -1], graph_data)
        decoder_input = input_tensor[:, -1, :].clone().repeat(1, mixtures)
        past_state = input_tensor[:, -1:, :n_states].expand(-1, mixtures, -1)

        # 解码器输入
        tar_edge_index = graph_data.edge_index[-1]
        tar_edge_features = graph_data.edge_features[-1]
        nan_mask = graph_data.nan_mask

        dec_in = (
            decoder_input, decoder_hidden, encoder_output,
            tar_edge_index, tar_edge_features, past_state, batch, nan_mask, residual
        )

        (next_states, model_input, q_t, decoder_hidden, A_vec,
         A_vec_mask_output, total_group_num, _) = self.decoder(dec_in)

        # 时间更新
        G_t, _ = self.decoder.motion_model.input_transition_matrix(past_state, model_input)
        P_t, Q_t = self.ekf_filter.time_update(P_t, G_t, q_t, device=input_tensor.device)

        pred_states = []
        pred_states.append(next_states)

        all_states = torch.stack(pred_states, dim=1)
        return all_states, P_t, total_group_num

    def update(self, state_input, P_input, measurement_input, R=None):
        """
        实际跟踪中的更新 (卡尔曼测量更新)

        Args:
            state_input: 状态输入
            P_input: 协方差输入
            measurement_input: 测量输入
            R: 测量噪声 (可选)

        Returns:
            tuple: (更新后的状态, 更新后的协方差)
        """
        n_measurements = measurement_input.shape[-1]
        batch_size = measurement_input.shape[0]

        if R is None:
            R = torch.diag_embed(
                torch.ones(batch_size, n_measurements, device=state_input.device)
            ) * self.initial_uncertainty_measurement

        H = MatrixBuilder.build_observation_matrix(state_input.device)
        x_update, P_update, _, _ = self.ekf_filter.measurement_update(
            state_input, P_input, measurement_input, R
        )

        return x_update, P_update

    def connected_components(self, edge_index, num_nodes):
        """
        计算连通分量 (用于群检测)

        Args:
            edge_index: [2, E] 边索引
            num_nodes: 节点数量

        Returns:
            group_id: [num_nodes] 群编号 (从 0 开始)
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

        # 合并所有边
        src, dst = edge_index
        for u, v in zip(src.tolist(), dst.tolist()):
            union(u, v)

        # 压缩并重新编号
        root_to_group = {}
        group_id = torch.zeros(num_nodes, dtype=torch.long)

        gid = 0
        for i in range(num_nodes):
            r = find(i)
            if r not in root_to_group:
                root_to_group[r] = gid
                gid += 1
            group_id[i] = root_to_group[r]

        return group_id
