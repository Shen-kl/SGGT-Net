import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnn_layers import *
from .motion_models import *
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import Batch
from torch.nn.functional import softmax
from model.MLP import ResMLP
import numpy as np
from config import CONFIG
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter
from torch_geometric.utils import unbatch, unbatch_edge_index
from model.AttentionMechanism import *
from matplotlib import pyplot as plt
import pickle
from typing import NamedTuple

from model.sggt_v2_modules import (
    JointGraphTemporalSpectralFilter,
    SpectralFilterState,
    TemporalProbabilisticSEAN,
)


class CausalNISProcessNoiseAdapter(nn.Module):
    """Small two-head adapter for strictly lagged innovation statistics."""

    def __init__(self, input_size, hidden_size, acceleration_probability,
                 position_probability):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.acceleration_head = nn.Linear(hidden_size, 1)
        self.position_head = nn.Linear(hidden_size, 2)
        for head in (self.acceleration_head, self.position_head):
            nn.init.zeros_(head.weight)
        with torch.no_grad():
            self.acceleration_head.bias.fill_(math.log(
                acceleration_probability / (1.0 - acceleration_probability)
            ))
            self.position_head.bias.fill_(math.log(
                position_probability / (1.0 - position_probability)
            ))

    def forward(self, features):
        latent = self.trunk(features)
        return torch.cat((
            self.acceleration_head(latent),
            self.position_head(latent),
        ), dim=-1)

'''
 改进encoder-decoder 增加残差补偿模块 改进SDE
'''

def extract_upper_triangle(A):
    N, _ = A.shape
    row, col = torch.triu_indices(N, N, offset=1, device=A.device)
    return A[ row, col]   # [B, E]

class SEAN(nn.Module):
    """
    Learn symmetric adjacency matrix and structural embeddings.
    """

    def __init__(self, in_dim, hidden_dim, struct_dim):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.struct_proj = nn.Linear(hidden_dim, struct_dim)

        self.embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),
        )
        self.embedding_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )


        self.temp = nn.Parameter(torch.tensor(1.0))  # 可学习的缩放参数 (Temperature)

    def forward(self, batch, x_states, x_hidden, mask=None):
        """
        x: [B, N, d]
        mask: [B, N]  (True for valid node)

        returns:
            A: [B, N, N]  symmetric adjacency probability
            struct_feat: [B, N, struct_dim]
        """
        # 确保 batch 是有序的，获取排序索引
        sorted_indices = torch.argsort(batch)
        # 用于最后恢复原始顺序：inverse_indices[排序后的位置] = 原始位置
        inverse_indices = torch.argsort(sorted_indices)

        # 按 batch 顺序重新排列所有输入
        sorted_batch = batch[sorted_indices]
        sorted_x_hidden = x_hidden[sorted_indices]
        sorted_x_states = x_states[sorted_indices]
        sorted_mask = mask[sorted_indices]
        # 使用unbatch拆分节点特征
        node_features_list = unbatch(sorted_x_hidden, sorted_batch)

        x_state_list = unbatch(sorted_x_states, batch)
        if mask is None:
            mask = torch.ones([x_hidden.shape[0],1])
        mask_list = unbatch(sorted_mask, batch)

        # 初始化空队列
        A_vec_list = []
        struct_feat_list = []
        node_mask_list = []

        total_group_num = 0 # 群的总数

        for x_state, node_features, mask_batch in zip(x_state_list, node_features_list,mask_list):
            x_state = x_state.squeeze(dim=1)
            N, _ = node_features.shape

            # h = self.node_mlp(node_features)  # [N,1,H]

            z = self.embedding_node(node_features)
            x_state_diff_embed = self.embedding(x_state.unsqueeze(dim=1) - x_state.unsqueeze(dim=0))
            logits = self.node_mlp(z.unsqueeze(dim = 1) + z.unsqueeze(dim=0) + x_state_diff_embed).squeeze()  # [N, E] 得到身份嵌入

            # logits = torch.matmul(z, z.transpose(-1, -2)) / self.temp

            A = torch.sigmoid(logits)
            A_tmp = A.clone() # 用于计算群个数


            # ---- 去掉对角线 ----
            diag_mask = torch.eye(N, device=x_hidden.device).bool()
            A = A.masked_fill(diag_mask, 0.)

            # ---- mask invalid node ----
            node_mask = mask_batch & mask_batch.transpose(-1, -2)
            A = A * node_mask.float()
            A_tmp = A_tmp * node_mask.float()
            A_tmp = (A_tmp > 0.5).float()


            # 计算群的个数
            groups_num, groups = self.count_groups_torch(A_tmp.detach())
            total_group_num += groups_num

            # ---- 结构 embedding ----
            struct_feat = torch.matmul(A, z)  # 图聚合
            struct_feat = self.struct_proj(struct_feat)

            node_mask_vec = extract_upper_triangle(node_mask)
            A_vec = extract_upper_triangle(A)

            A_vec_list.append(A_vec)
            struct_feat_list.append(struct_feat)
            node_mask_list.append(node_mask_vec)

        # 拼接结果
        A_vec_sorted = torch.cat(A_vec_list, dim=-1)
        struct_feat_sorted = torch.cat(struct_feat_list, dim=0)
        node_mask_sorted = torch.cat(node_mask_list, dim=-1)


        # 恢复原始顺序
        A_vec_output = A_vec_sorted
        struct_feat_output = struct_feat_sorted[inverse_indices]
        node_mask_output = node_mask_sorted

        return A_vec_output, struct_feat_output, node_mask_output, total_group_num

    def count_groups_torch(self, A):

        N = A.shape[0]

        valid = torch.diagonal(A) != 0

        B = (A != 0)

        B[~valid, :] = False
        B[:, ~valid] = False

        B.fill_diagonal_(True)

        for k in range(N):
            B = B | (B[:, k:k + 1] & B[k:k + 1, :])

        visited = torch.zeros(N, dtype=torch.bool)

        groups = []

        for i in range(N):

            if not valid[i] or visited[i]:
                continue

            group = torch.where(B[i])[0]

            visited[group] = True

            groups.append(group.tolist())

        return len(groups), groups

class EdgeProjector(nn.Module):
    def __init__(self):
        super(EdgeProjector, self).__init__()
        # 线性层将 2 维特征投影到 1 维
        self.proj = nn.Linear(in_features=2, out_features=1)
        # 可以用 Sigmoid 或 ReLU
        self.activation = nn.Sigmoid()

    def forward(self, edge_attr_2d):
        # edge_attr_2d 维度: [E_total, 2]
        edge_weight_1d = self.proj(edge_attr_2d) # [E_total, 1]
        return self.activation(edge_weight_1d).squeeze(dim=-1) # [E_total]
        # 使用 squeeze(dim=-1) 确保输出是严格的 1D 向量

class GRUGNNCell(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, n_heads=3,
                 n_layers=1, dropout=0.1, gnn_layer="graphconv", edge_dim=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 3 * hidden_size  # Need 3 hidden states for all GRU gates

        self.Wx = create_sequential_gnn(input_size=input_size,
                                        output_size=gate_size,
                                        hidden_size=hidden_size,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        layers=n_layers,
                                        activation='lrelu',
                                        gnn_layer=gnn_layer,
                                        edge_dim=edge_dim)

        self.Wh = create_sequential_gnn(input_size=hidden_size,
                                        output_size=gate_size,
                                        hidden_size=hidden_size,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        layers=n_layers,
                                        activation='lrelu',
                                        gnn_layer=gnn_layer,
                                        edge_dim=edge_dim)

        self.bias = nn.Parameter(torch.empty(gate_size).uniform_(-1e-2, 1e-2))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (math.sqrt(self.hidden_size))

        # Exclude edge bws from initialization
        init_params = (p for name, p in self.named_parameters() if
                       not str.endswith(name, "log_edge_bw"))
        for p in init_params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -std, std)

    def forward(self, x, edge_index, h=None, edge_attr=None):
        #  Implements GRUCell update:
        #  https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        #  Using GNNs as learnable functions
        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        wrx, wzx, wqx = torch.split(self.Wx(x, edge_index, edge_attr),
                                    self.hidden_size, dim=1)
        wrh, wzh, wqh = torch.split(self.Wh(h, edge_index, edge_attr),
                                    self.hidden_size, dim=1)
        br, bz, bq = torch.split(self.bias, self.hidden_size, dim=0)

        r = torch.sigmoid(wrx + wrh + br)
        z = torch.sigmoid(wzx + wzh + bz)
        q = torch.tanh(wqx + r * wqh + bq)
        h = (1 - z) * q + z * h
        return h


class GRUGNNEncoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, n_heads=3, n_layers=1,
                 n_mixtures=7, dropout=0.1, gnn_layer="graphconv",
                 use_edge_features=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        init_std = 1.0 / (math.sqrt(hidden_size))
        self.init_state_param = nn.Parameter(torch.empty(hidden_size).uniform_(
            -1e-2, 1e-2))

        edge_dim = 1 if use_edge_features else None
        self.gru_cell = GRUGNNCell(input_size, hidden_size, n_heads, n_layers, dropout,
                                   gnn_layer, edge_dim=edge_dim)

        self.mixture = nn.Linear(hidden_size, n_mixtures)  # <-- mixture weights
        self.edge_projector = EdgeProjector()
    def init_hidden(self, data, batch_size):
        return self.init_state_param.repeat(batch_size, 1)

    def forward(self, data, hidden=None):
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_features
        batch_size, _, _ = x.size()
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = self.init_hidden(data, batch_size)
        output = [hidden]

        for x_i, ei_i, ef_i in zip(x.transpose(0, 1), edge_index, edge_features):
            hidden = self.gru_cell(x_i, ei_i, hidden, edge_attr=self.edge_projector(ef_i.to(torch.float32)))
            output.append(hidden)
        output = torch.stack(output, dim=1)
        mixture_w = self.mixture(self.dropout(F.leaky_relu(hidden, negative_slope=0.2)))
        return output, mixture_w


class GRUGNNDecoder(nn.Module):
    def __init__(self, motion_model, delta_T, max_length=10, hidden_size=64,
                 n_heads=3, n_layers=1, alpha=0.2, dropout=0.1, residual_length=[8, 16, 32], z_dimension=2,
                 gnn_layer="graphconv", use_GASF=True, decoder_use_struct=True):
        super().__init__()
        self.gru_cell = GRUGNNCell(hidden_size, hidden_size, n_heads, n_layers, dropout,
                                   gnn_layer, edge_dim=1)
        self.alpha = alpha
        self.use_GASF = use_GASF
        self.decoder_use_struct = decoder_use_struct
        self.motion_model = motion_model
        self.input_size = motion_model.n_states
        self.output_size = z_dimension  # 由三部分构成
        self.mixtures = motion_model.mixtures
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=dropout)

        # Temporal attention weight calculations
        self.embedding = nn.Linear(int(self.input_size * self.mixtures), hidden_size)
        self.attn = nn.ModuleList([ nn.Linear(hidden_size * 2, hidden_size * 2) for _ in range(2)]
                                  + [nn.Linear(hidden_size * 2, max_length)])

        self.attn_combine = nn.Linear(hidden_size, hidden_size)

        # Scale GRU outputs
        self.generator = nn.Linear(hidden_size, hidden_size * 4)

        # Motion model inputs
        self.controller = nn.Linear(hidden_size, int(self.input_size * self.mixtures))

        # To generate process noise
        self.sig_1 = nn.Linear(hidden_size , int(self.mixtures))
        self.sig_2 = nn.Linear(hidden_size , int(self.mixtures))
        self.rho = nn.Linear(hidden_size , int(self.mixtures))

        self.pairwise_g = ResMLP(hidden_size, hidden_size, int(self.input_size * self.mixtures),2)
        # struct
        self.SEAN = SEAN(hidden_size,hidden_size,hidden_size)

        self.GASF = nn.ModuleList(GraphSpectralFilter(fft_point=64, n_heads=2, n_layers=2, dropout=0.1, gnn_layer="graphconv")
                                   for _ in range(len(residual_length)))

        self.residual_length = residual_length

        self.linear_GASF = nn.ModuleList( nn.Linear(hidden_size, 1) for _ in range(len(residual_length)))

        self.delta_T = delta_T

        self.edge_projector = EdgeProjector()

        self.linear_1 = nn.Linear(int(self.output_size * self.mixtures), hidden_size)

        self.Q_scaler = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.LayerNorm(hidden_size),
                                    nn.SiLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Sigmoid())

        self.attention = AdditiveAttention(int(hidden_size), int(hidden_size), int(hidden_size * 2),
                                           dropout=0.1)

        # 可学习的衰减率，初始值对应约10帧的半衰期
        self.time_decay = nn.Parameter(torch.tensor(0.1))

        self.linear_2 = nn.Linear(hidden_size * 2, hidden_size)
    def process_noise_matrix(self, x1, x2, x3, batch_size, struct_feats):

        sig1 = F.softplus(self.sig_1(x1))
        sig2 = F.softplus(self.sig_2(x2))
        rho = F.softsign(self.rho(x3))

        q_t = torch.zeros(batch_size, int(self.mixtures), self.output_size,
                          self.output_size, device=x1.device)

        q_t[..., 0, 0] = torch.pow(sig1, 2)
        q_t[..., 1, 1] = torch.pow(sig2, 2)
        q_t[..., 0, 1] = q_t[..., 1, 0] = sig1 * sig2 * rho

        noise_scale = self.Q_scaler(struct_feats).unsqueeze(1).unsqueeze(1)

        return noise_scale * q_t

    def forward(self, data):
        x, hidden, encoder_out, edge_index, edge_feature, past_state, batch, nan_mask, residual = data
        batch_size = x.size(0)

        GASF_output_list = []
        delta_comp_last_frame = []
        cnt = 0
        # 频域处理
        for GASF_single in self.GASF:
            delta_comp_single, gate_final = GASF_single(residual[:,-self.residual_length[cnt]:,:], nan_mask, edge_index, None, edge_feature)
            GASF_output_list.append(delta_comp_single)
            cnt+=1

        GASF_output_stack = torch.stack(GASF_output_list,dim=1)

        GASF_alpha = []
        cnt = 0
        for linear_GASF_single in self.linear_GASF:
            GASF_alpha.append(linear_GASF_single(GASF_output_list[cnt]))
            cnt+=1

        GASF_alpha = torch.sigmoid(torch.cat(GASF_alpha,dim=1))

        GASF_output_fusion = torch.sum(GASF_output_stack * GASF_alpha[...,None], dim=1)


        # attention 输出
        T = encoder_out.shape[1]
        time = torch.arange(T).float()
        decay = torch.sigmoid(self.time_decay)  # 或 F.softplus
        time_feat = torch.exp(-decay * (T - 1 - time))  # 最新帧=0指数=1
        time_feat = time_feat.view(1, T, 1)

        encoder_out_atten_input = torch.cat([encoder_out.clone(), time_feat.expand(encoder_out.size(0), -1, encoder_out.size(-1))], dim=-1)
        encoder_out_atten_input = self.linear_2(encoder_out_atten_input)
        # encoder_out_atten_input = encoder_out.clone() + time_feat
        attn_applied = self.attention(encoder_out_atten_input,encoder_out_atten_input,encoder_out_atten_input[:,-1,:].unsqueeze(1))

        # output = torch.cat((embedded, attn_applied[:, 0]), 1)
        output = self.attn_combine(attn_applied[:, 0])
        output = F.leaky_relu(output, self.alpha)
        hidden = self.gru_cell(output, edge_index, hidden, edge_attr=self.edge_projector(edge_feature.to(torch.float32)))

        A_vec, struct_feats, A_vec_mask_output, total_group_num = self.SEAN(batch, past_state, hidden, mask=torch.all(nan_mask[:,-2:,-1],dim=1).unsqueeze(1))

        output = F.leaky_relu(self.generator(F.leaky_relu(hidden, self.alpha)), self.alpha)
        output = self.dropout(output)

        x1, x2, x3, x4 = torch.split(output, self.hidden_size, dim=-1)
        # x_model_input = self.controller(x4).view(batch_size,
        #                                        self.mixtures,
        #                                        self.input_size)

        noise_feat1 = torch.cat([x1],dim=-1)
        noise_feat2 = torch.cat([x2], dim=-1)
        noise_feat3 = torch.cat([x3], dim=-1)
        process_noise = self.process_noise_matrix(noise_feat1, noise_feat2, noise_feat3, batch_size, struct_feats)

        model_input = {}
        model_input['gnn_model_output'] = x4
        if self.use_GASF:
            model_input['struct_feats'] = struct_feats
        if self.decoder_use_struct:
            model_input['GASF_output_fusion'] = GASF_output_fusion

        next_state, model_input = self.motion_model(past_state, model_input, static_f=None)

        return next_state, model_input, process_noise, hidden, A_vec, A_vec_mask_output, total_group_num, gate_final

    def get_initial_state(self, last_enc_state, data):
        return last_enc_state


class DecoderV2Output(NamedTuple):
    next_state: torch.Tensor
    model_input: dict
    process_noise: torch.Tensor
    hidden: torch.Tensor
    adjacency_vector: torch.Tensor
    adjacency_mask: torch.Tensor
    group_count: int
    spectral_weights: torch.Tensor
    node_entropy: torch.Tensor
    auxiliary_delta_v: torch.Tensor
    film_gate: torch.Tensor
    policy_gate: torch.Tensor
    gasf_confidence: torch.Tensor
    q_acceleration_multiplier: torch.Tensor
    q_position_noise: torch.Tensor
    q_innovation_score: torch.Tensor
    auxiliary_delta_v_temporal: torch.Tensor
    auxiliary_delta_v_expert: torch.Tensor
    gasf_recurrent_state: SpectralFilterState | None
    gasf_parameter_update_gates: torch.Tensor


class GRUGNNDecoderV2(nn.Module):
    """Decoder with temporal structure inference and joint graph-time GASF."""

    def __init__(
        self,
        motion_model,
        delta_T,
        max_length=10,
        hidden_size=64,
        n_heads=3,
        n_layers=1,
        alpha=0.2,
        dropout=0.1,
        residual_length=(8, 16),
        z_dimension=2,
        gnn_layer="graphconv",
        use_GASF=True,
        use_SEAN=True,
        gasf_filter_mode="abc",
        gasf_graph_mode="graph",
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
    ):
        super().__init__()
        self.gru_cell = GRUGNNCell(
            hidden_size,
            hidden_size,
            n_heads,
            n_layers,
            dropout,
            gnn_layer,
            edge_dim=1,
        )
        self.alpha = alpha
        self.use_GASF = bool(use_GASF)
        # V2 spectral SEAN is intentionally defined only inside GASF.  A
        # standalone SEAN would duplicate the encoder GNN without a residual-
        # spectrum signal and is no longer a supported formal variant.
        self.use_SEAN = bool(use_SEAN and use_GASF)
        self.motion_model = motion_model
        self.input_size = motion_model.n_states
        self.output_size = z_dimension
        self.mixtures = motion_model.mixtures
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.adaptive_q_mode = str(adaptive_q_mode)
        self.adaptive_q_max_multiplier = float(adaptive_q_max_multiplier)
        self.adaptive_q_max_position_variance = float(adaptive_q_max_position_variance)
        self.adaptive_q_initial_multiplier = float(adaptive_q_initial_multiplier)
        self.adaptive_q_initial_position_variance = float(
            adaptive_q_initial_position_variance
        )
        self.adaptive_q_acceleration = bool(adaptive_q_acceleration)
        # Evaluation/ablation control for an explicitly monotone causal tail
        # response. Zero preserves the learned adapter exactly.
        self.q_tail_boost = float(adaptive_q_tail_boost)
        self.q_tail_position_boost = 0.0
        self.q_tail_threshold = float(adaptive_q_tail_threshold)
        # Evaluation-only counterfactual control. ``None`` preserves the
        # learned adapter; a non-negative value replaces the two direct
        # position-Q diagonal entries with that physical per-axis variance.
        # Keeping this control here makes position-Q scans use the exact same
        # covariance propagation path as training and online BP inference.
        self.q_position_variance_override_physical = None
        # Evaluation-only centered response scan.  ``None`` preserves the
        # learned causal adapter.  Zero replaces its time-varying response by
        # the supplied fixed means; negative values invert the response and
        # values above one amplify it.  Training never sets this control.
        self.q_response_scale_override = None
        self.q_response_acceleration_center = self.adaptive_q_initial_multiplier
        self.q_response_position_center = self.adaptive_q_initial_position_variance

        self.attn_combine = nn.Linear(hidden_size, hidden_size)
        self.generator = nn.Linear(hidden_size, hidden_size * 4)
        self.sig_1 = nn.Linear(hidden_size, int(self.mixtures))
        self.sig_2 = nn.Linear(hidden_size, int(self.mixtures))
        self.rho = nn.Linear(hidden_size, int(self.mixtures))
        self.edge_projector = EdgeProjector()
        self.attention = AdditiveAttention(
            int(hidden_size), int(hidden_size), int(hidden_size * 2), dropout=dropout
        )
        self.time_decay = nn.Parameter(torch.tensor(0.1))
        self.linear_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.Q_scaler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        # Q-v2 is strictly causal: ``residual`` contains innovations only up to
        # the preceding observed frame.  The final layer is initialized so an
        # old checkpoint starts close to multiplier=1 and negligible direct
        # position-model noise.
        if self.adaptive_q_mode == "causal_nis":
            if self.adaptive_q_max_multiplier <= 0.25:
                raise ValueError("causal_nis Q multiplier ceiling must exceed 0.25")
            if self.adaptive_q_max_position_variance <= 0.0:
                raise ValueError("causal_nis position-Q ceiling must be positive")
            initial_multiplier = min(max(
                self.adaptive_q_initial_multiplier, 0.250001
            ), self.adaptive_q_max_multiplier - 1e-6)
            log_minimum = math.log(0.25)
            log_maximum = math.log(self.adaptive_q_max_multiplier)
            acceleration_probability = (
                math.log(initial_multiplier) - log_minimum
            ) / (log_maximum - log_minimum)
            acceleration_probability = min(max(
                acceleration_probability, 1e-4
            ), 1.0 - 1e-4)
            # A shifted geometric map reaches zero but keeps a useful slope
            # around the small (1 m^2) initialization selected by the
            # scene02 counterfactual scan.
            position_floor = 0.25
            initial_position = min(max(
                self.adaptive_q_initial_position_variance, 0.0
            ), self.adaptive_q_max_position_variance - 1e-6)
            position_probability = math.log1p(
                initial_position / position_floor
            ) / math.log1p(
                self.adaptive_q_max_position_variance / position_floor
            )
            position_probability = min(max(
                position_probability, 1e-4
            ), 1.0 - 1e-4)
            # hidden + temporal/GASF latent + ten explicit causal statistics
            self.Q_innovation_encoder = CausalNISProcessNoiseAdapter(
                hidden_size * 2 + 10,
                hidden_size,
                acceleration_probability,
                position_probability,
            )
        elif self.adaptive_q_mode == "innovation":
            q_feature_size = hidden_size * 2 + 4
            self.Q_innovation_encoder = nn.Sequential(
                nn.Linear(q_feature_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 3),
            )
            final = self.Q_innovation_encoder[-1]
            nn.init.zeros_(final.weight)
            initial_probability = (1.0 - 0.25) / max(
                self.adaptive_q_max_multiplier - 0.25, 1e-6
            )
            initial_probability = min(max(initial_probability, 1e-4), 1.0 - 1e-4)
            with torch.no_grad():
                final.bias[0] = math.log(initial_probability / (1.0 - initial_probability))
                initial_position_variance = 0.01
                position_probability = initial_position_variance / max(
                    self.adaptive_q_max_position_variance, initial_position_variance
                )
                position_probability = min(max(position_probability, 1e-6), 1.0 - 1e-6)
                final.bias[1:].fill_(
                    math.log(position_probability / (1.0 - position_probability))
                )
        else:
            self.Q_innovation_encoder = None

        if self.use_GASF:
            self.GASF = JointGraphTemporalSpectralFilter(
                hidden_size,
                windows=tuple(int(value) for value in residual_length),
                signal_dim=z_dimension,
                chebyshev_order=2,
                filter_mode=gasf_filter_mode,
                graph_mode=gasf_graph_mode,
                use_sean=self.use_SEAN,
                use_context_activation=(
                    self.use_SEAN
                    and self.motion_model.interaction_physical_residual
                ),
                interaction_activation_floor=(
                    self.motion_model.interaction_activation_floor
                ),
                innovation_decomposition=innovation_decomposition,
                innovation_whiten_clip=innovation_whiten_clip,
                recurrent_filter=gasf_recurrent_filter,
                recurrent_gate_initial_update=(
                    gasf_recurrent_gate_initial_update
                ),
            )
        else:
            self.GASF = None

    @property
    def SEAN(self):
        """Expose embedded SEAN for diagnostics and staged optimization."""
        return self.GASF.SEAN if self.GASF is not None else None

    def process_noise_matrix(self, x1, x2, x3, batch_size, scale_features):
        sig1 = F.softplus(self.sig_1(x1))
        sig2 = F.softplus(self.sig_2(x2))
        rho = F.softsign(self.rho(x3))
        q_t = torch.zeros(
            batch_size,
            int(self.mixtures),
            self.output_size,
            self.output_size,
            device=x1.device,
        )
        q_t[..., 0, 0] = sig1.square()
        q_t[..., 1, 1] = sig2.square()
        q_t[..., 0, 1] = q_t[..., 1, 0] = sig1 * sig2 * rho
        noise_scale = self.Q_scaler(scale_features).unsqueeze(1).unsqueeze(1)
        return noise_scale * q_t

    def _causal_nis_statistics(self, innovation_bundle, residual):
        """Summarize only innovations already present before prediction t."""
        node_count, history_length = residual.shape[:2]
        if innovation_bundle is None:
            return residual.new_zeros(node_count, 10)
        valid = innovation_bundle.get('valid')
        whitened = innovation_bundle.get('whitened')
        raw = innovation_bundle.get('raw')
        nis = innovation_bundle.get('nis')
        if any(value is None for value in (valid, whitened, raw, nis)):
            return residual.new_zeros(node_count, 10)
        valid = valid[..., 0].bool()
        nis = nis[..., 0].to(residual).clamp(min=0.0, max=400.0)
        whitened = whitened.to(residual)
        raw = raw.to(residual)
        if valid.size(1) != history_length:
            length = min(valid.size(1), history_length)
            valid = valid[:, -length:]
            nis = nis[:, -length:]
            whitened = whitened[:, -length:]
            raw = raw[:, -length:]
            history_length = length
        weight = valid.to(residual.dtype)
        denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_nis = (nis * weight).sum(dim=1, keepdim=True) / denominator
        maximum_nis = torch.where(
            valid, nis, torch.zeros_like(nis)
        ).amax(dim=1, keepdim=True)
        indices = torch.arange(
            history_length, device=residual.device
        ).view(1, -1).expand(node_count, -1)
        last_index = torch.where(
            valid, indices, torch.zeros_like(indices)
        ).amax(dim=1)
        gather_index = last_index.view(-1, 1)
        latest_nis = nis.gather(1, gather_index)
        latest_white = whitened.gather(
            1, gather_index.unsqueeze(-1).expand(-1, 1, 2)
        ).squeeze(1)
        has_history = valid.any(dim=1, keepdim=True)
        latest_nis = torch.where(
            has_history, latest_nis, torch.zeros_like(latest_nis)
        )
        latest_white = torch.where(
            has_history, latest_white, torch.zeros_like(latest_white)
        )
        mean_white = (
            whitened * weight.unsqueeze(-1)
        ).sum(dim=1) / denominator
        raw_norm_square = raw.square().sum(dim=-1)
        raw_rms = (
            (raw_norm_square * weight).sum(dim=1, keepdim=True) / denominator
        ).sqrt() / float(CONFIG['S_RES'])
        if history_length > 1:
            pair_valid = valid[:, 1:] & valid[:, :-1]
            pair_weight = pair_valid.to(residual.dtype)
            pair_denominator = pair_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            raw_delta_square = (raw[:, 1:] - raw[:, :-1]).square().sum(dim=-1)
            raw_delta_rms = (
                (raw_delta_square * pair_weight).sum(dim=1, keepdim=True)
                / pair_denominator
            ).sqrt() / float(CONFIG['S_RES'])
        else:
            raw_delta_rms = raw_rms.new_zeros(raw_rms.shape)
        statistics = torch.cat((
            torch.log1p(mean_nis),
            torch.log1p(maximum_nis),
            torch.log1p(latest_nis),
            weight.mean(dim=1, keepdim=True),
            mean_white,
            latest_white,
            torch.log1p(raw_rms),
            torch.log1p(raw_delta_rms),
        ), dim=-1)
        return torch.nan_to_num(statistics, nan=0.0, posinf=20.0, neginf=-20.0)

    def adaptive_process_noise(
        self, hidden, gasf_features, residual, innovation_bundle=None
    ):
        """Return acceleration multiplier and normalized position-only Q.

        No current/future measurement is read here.  Position variance is
        represented separately because G q G^T couples position and velocity.
        """
        node_count = hidden.size(0)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        delta = residual[:, 1:] - residual[:, :-1]
        delta_norm = torch.linalg.vector_norm(delta, dim=-1)
        statistics = torch.stack((
            torch.log1p(residual_norm.square().mean(dim=-1).sqrt()),
            torch.log1p(residual_norm.amax(dim=-1)),
            torch.log1p(delta_norm.square().mean(dim=-1).sqrt()),
            torch.log1p(residual_norm[:, -1]),
        ), dim=-1)
        if self.Q_innovation_encoder is None:
            multiplier = hidden.new_ones(node_count, 1)
            if self.q_tail_boost > 0.0:
                tail = F.relu(statistics[:, :1] - float(self.q_tail_threshold))
                multiplier = (multiplier + self.q_tail_boost * tail).clamp(
                    max=self.adaptive_q_max_multiplier
                )
            position_q = hidden.new_zeros(node_count, 1, 4, 4)
            override = self.q_position_variance_override_physical
            if override is not None:
                normalized = float(override) / float(CONFIG['S_POS']) ** 2
                position_q[:, 0, 0, 0] = normalized
                position_q[:, 0, 1, 1] = normalized
            return multiplier, position_q, statistics[:, :1]
        if self.adaptive_q_mode == "causal_nis":
            statistics = self._causal_nis_statistics(
                innovation_bundle, residual
            )
        raw = self.Q_innovation_encoder(
            torch.cat((hidden, gasf_features, statistics), dim=-1)
        )
        if self.adaptive_q_mode == "causal_nis":
            log_minimum = math.log(0.25)
            log_maximum = math.log(self.adaptive_q_max_multiplier)
            multiplier = torch.exp(
                log_minimum
                + (log_maximum - log_minimum) * torch.sigmoid(raw[:, :1])
            )
        else:
            multiplier = 0.25 + (
                self.adaptive_q_max_multiplier - 0.25
            ) * torch.sigmoid(raw[:, :1])
        if not self.adaptive_q_acceleration:
            multiplier = torch.ones_like(multiplier)
        if self.q_tail_boost > 0.0:
            tail = F.relu(statistics[:, :1] - float(self.q_tail_threshold))
            multiplier = multiplier + float(self.q_tail_boost) * tail
            multiplier = multiplier.clamp(max=self.adaptive_q_max_multiplier)
        response_scale = self.q_response_scale_override
        if response_scale is not None:
            acceleration_center = float(self.q_response_acceleration_center)
            multiplier = acceleration_center + float(response_scale) * (
                multiplier - acceleration_center
            )
            multiplier = multiplier.clamp(
                min=0.25, max=self.adaptive_q_max_multiplier
            )
        # These heads predict physical position standard deviations.  Convert
        # their variances to normalized covariance coordinates here.
        if self.adaptive_q_mode == "causal_nis":
            position_floor = 0.25
            log_span = math.log1p(
                self.adaptive_q_max_position_variance / position_floor
            )
            position_variance_physical = position_floor * torch.expm1(
                log_span * torch.sigmoid(raw[:, 1:3])
            )
        else:
            position_variance_physical = (
                self.adaptive_q_max_position_variance
                * torch.sigmoid(raw[:, 1:3])
            )
        if response_scale is not None:
            position_center = float(self.q_response_position_center)
            position_variance_physical = position_center + float(response_scale) * (
                position_variance_physical - position_center
            )
            position_variance_physical = position_variance_physical.clamp(
                min=0.0, max=self.adaptive_q_max_position_variance
            )
        position_variance = position_variance_physical / float(CONFIG['S_POS']) ** 2
        position_q = hidden.new_zeros(node_count, 1, 4, 4)
        position_q[:, 0, 0, 0] = position_variance[:, 0]
        position_q[:, 0, 1, 1] = position_variance[:, 1]
        if self.q_tail_position_boost > 0.0:
            tail = F.relu(statistics[:, :1] - float(self.q_tail_threshold))
            extra_physical = float(self.q_tail_position_boost) * tail
            extra_normalized = extra_physical / float(CONFIG['S_POS']) ** 2
            position_q[:, 0, 0, 0] = position_q[:, 0, 0, 0] + extra_normalized[:, 0]
            position_q[:, 0, 1, 1] = position_q[:, 0, 1, 1] + extra_normalized[:, 0]
        override = self.q_position_variance_override_physical
        if override is not None:
            normalized = float(override) / float(CONFIG['S_POS']) ** 2
            position_q[:, 0, 0, 0] = normalized
            position_q[:, 0, 1, 1] = normalized
        return multiplier, position_q, statistics[:, :1]

    def forward(self, data):
        if len(data) not in (9, 10, 11, 12):
            raise ValueError(
                f"V2 decoder expects 9 to 12 inputs, got {len(data)}"
            )
        (
            x,
            hidden,
            encoder_out,
            edge_index,
            edge_feature,
            past_state,
            batch,
            nan_mask,
            residual,
        ) = data[:9]
        if len(data) >= 10:
            innovation_bundle = data[9]
        else:
            innovation_bundle = None
        gasf_recurrent_state = data[10] if len(data) >= 11 else None
        recurrent_valid_mask = data[11] if len(data) >= 12 else None
        batch_size = x.size(0)

        time_steps = encoder_out.shape[1]
        time = torch.arange(
            time_steps, device=encoder_out.device, dtype=encoder_out.dtype
        )
        decay = torch.sigmoid(self.time_decay)
        time_feature = torch.exp(-decay * (time_steps - 1 - time)).view(1, time_steps, 1)
        attention_input = self.linear_2(
            torch.cat(
                [
                    encoder_out,
                    time_feature.expand(encoder_out.size(0), -1, encoder_out.size(-1)),
                ],
                dim=-1,
            )
        )
        attended = self.attention(
            attention_input, attention_input, attention_input[:, -1, :].unsqueeze(1)
        )
        output = F.leaky_relu(self.attn_combine(attended[:, 0]), self.alpha)
        projected_edge = self.edge_projector(edge_feature.to(torch.float32))
        hidden = self.gru_cell(
            output, edge_index, hidden, edge_attr=projected_edge
        )

        node_entropy = torch.ones(batch_size, 1, device=hidden.device)
        adjacency_vector = hidden.new_empty(0)
        adjacency_mask = torch.empty(0, dtype=torch.bool, device=hidden.device)
        group_count = 0
        temporal_gasf = interaction_expert_gasf = torch.zeros_like(hidden)
        if self.use_GASF:
            spectral = self.GASF(
                residual,
                edge_index,
                projected_edge,
                node_entropy,
                nan_mask,
                batch=batch,
                states=past_state,
                innovation_bundle=innovation_bundle,
                recurrent_state=gasf_recurrent_state,
                recurrent_valid_mask=recurrent_valid_mask,
            )
            gasf_features = spectral.latent
            gasf_confidence = spectral.confidence
            spectral_weights = spectral.scale_weights
            temporal_gasf = spectral.temporal_latent
            interaction_expert_gasf = (
                spectral.temporal_latent + spectral.interaction_delta
            )
            if self.motion_model.interaction_physical_residual:
                # The physically aligned interaction path replaces, rather
                # than stacks with, the poorly conditioned latent increment.
                gasf_features = spectral.temporal_latent
                interaction_expert_gasf = spectral.temporal_latent
            structure = spectral.structure
            node_entropy = structure.node_entropy
            adjacency_vector = structure.adjacency_vector
            adjacency_mask = structure.vector_mask
            group_count = structure.group_count
            auxiliary_delta_v_temporal = self.motion_model.aux_velocity_head(
                temporal_gasf
            )
            auxiliary_delta_v_expert = self.motion_model.aux_velocity_head(
                interaction_expert_gasf
            )
            gasf_recurrent_state = spectral.recurrent_state
            gasf_parameter_update_gates = spectral.parameter_update_gates
        else:
            gasf_features = torch.zeros_like(hidden)
            gasf_confidence = torch.zeros(batch_size, 1, device=hidden.device)
            spectral_weights = torch.zeros(batch_size, 0, device=hidden.device)
            auxiliary_delta_v_temporal = torch.zeros(
                batch_size, 2, device=hidden.device
            )
            auxiliary_delta_v_expert = torch.zeros(
                batch_size, 2, device=hidden.device
            )
            gasf_recurrent_state = None
            gasf_parameter_update_gates = torch.zeros(
                batch_size, 0, device=hidden.device
            )

        generated = self.dropout(
            F.leaky_relu(
                self.generator(F.leaky_relu(hidden, self.alpha)), self.alpha
            )
        )
        x1, x2, x3, x4 = torch.split(generated, self.hidden_size, dim=-1)
        # Keep the base dynamics and process-noise path identical to the
        # no-SEAN baseline; SEAN controls only the residual GASF/FiLM path.
        scale_features = hidden
        process_noise = self.process_noise_matrix(
            x1, x2, x3, batch_size, scale_features
        )
        q_multiplier, q_position_noise, q_innovation_score = self.adaptive_process_noise(
            hidden, gasf_features, residual, innovation_bundle
        )
        # process_noise is [node, mixture, 2, 2] while q_multiplier is
        # [node, 1].  Two explicit matrix axes are required; unsqueeze(1)
        # alone broadcasts the node dimension into the mixture dimension and
        # creates an erroneous [node, node, 2, 2] tensor for mixture=1.
        process_noise = process_noise * q_multiplier.unsqueeze(-1).unsqueeze(-1)

        model_input = {
            'gnn_model_output': x4,
        }
        if self.motion_model.causal_innovation_corrector is not None:
            if innovation_bundle is None:
                raise ValueError(
                    "CIPIC requires a causal innovation history bundle"
                )
            # Deliberately expose only the four causal raw-history tensors to
            # CIPIC. The complete innovation bundle remains available to the
            # established MPSIF/Q paths above, but covariance, NIS, whitening
            # and association descriptors cannot enter the correction branch
            # even by accident.
            model_input['causal_innovation_bundle'] = {
                key: innovation_bundle[key]
                for key in ('raw', 'peer_raw', 'valid', 'peer_valid')
            }
        if self.use_GASF:
            model_input['GASF_output_fusion'] = gasf_features
            model_input['GASF_confidence'] = gasf_confidence
            # These features are consumed only by the optional trajectory-
            # supervised counterfactual branches in SGGT_Net.
            model_input['GASF_output_temporal'] = temporal_gasf
            model_input['GASF_output_interaction_expert'] = interaction_expert_gasf
            raw_interaction_context = spectral.interaction_context
            model_input['GASF_interaction_context'] = (
                gasf_confidence * raw_interaction_context
            )
            model_input['GASF_interaction_context_temporal'] = torch.zeros_like(
                raw_interaction_context
            )
            model_input['GASF_interaction_context_expert'] = raw_interaction_context

        next_state, model_input = self.motion_model(
            past_state, model_input, static_f=None
        )
        return DecoderV2Output(
            next_state,
            model_input,
            process_noise,
            hidden,
            adjacency_vector,
            adjacency_mask,
            group_count,
            spectral_weights,
            node_entropy,
            model_input['auxiliary_delta_v'],
            model_input['film_gate'],
            model_input['policy_gate'],
            model_input['gasf_confidence_effective'],
            q_multiplier,
            q_position_noise,
            q_innovation_score,
            auxiliary_delta_v_temporal,
            auxiliary_delta_v_expert,
            gasf_recurrent_state,
            gasf_parameter_update_gates,
        )

    def get_initial_state(self, last_enc_state, data):
        return last_enc_state


class GraphSpectralFilter(nn.Module):
    def __init__(self, fft_point, n_heads, n_layers, dropout, gnn_layer):
        super(GraphSpectralFilter, self).__init__()
        self.fft_point = fft_point

        self.freqs = torch.linspace(0, 1, self.fft_point // 2 + 1)


        self.gate = ResMLP((fft_point+2) * 2, (fft_point+2),
                            3 * 2 * 2, 2, 0.1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm((self.fft_point+2) * 2)

        self.gru_cell = GRUGNNCell((self.fft_point+2)* 2, (64+2) * 2, n_heads, n_layers, dropout,
                                   gnn_layer)

        self.edge_projector  = EdgeProjector()
        self.feature_dim = fft_point
        self.out_proj = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.temporal_conv = nn.Conv1d(2, 2, kernel_size=3, padding=1)
        # =================================
        # 导出onnx需要
        # self.custom_fft = CustomFFT(n=fft_point, norm="backward")
        # self.custom_ifft = CustomIFFT(n=fft_point, norm="backward")
        # =================================
    def forward(self, innovationError: torch.Tensor, nan_mask: torch.Tensor, edge_index, hidden=None, edge_feature=None):
        batch_size = innovationError.shape[0]
        innovationError_reshape = innovationError.permute(0, 2, 1).reshape(innovationError.shape[0],-1,innovationError.shape[1])


        # =================================
        innovationError_frequency_spectrum = torch.fft.fft(innovationError_reshape, n=self.fft_point, dim=-1)
        frequency_spectrum_real = innovationError_frequency_spectrum.real
        frequency_spectrum_imag = innovationError_frequency_spectrum.imag
        # =================================

        #=================================
        # 导出onnx需要
        # innovationError_frequency_spectrum =self.custom_fft(innovationError_reshape, innovationError_reshape.ndim - 1)
        # frequency_spectrum_real = innovationError_frequency_spectrum[:,:,:,0]
        # frequency_spectrum_imag = innovationError_frequency_spectrum[:,:,:,1]
        # =================================

        freq_dim = frequency_spectrum_real.shape[-1]  # 因为你拼接了实部+虚部，先还原原始频谱维度
        half_freq = (freq_dim + 1) // 2 + 1  # 处理奇数维度，确保包含中心频率
        freq_features = torch.cat([frequency_spectrum_real[...,:half_freq],
                                   frequency_spectrum_imag[...,:half_freq]], dim=-1)
        freq_features = freq_features.reshape(freq_features.shape[0], -1)

        freq_features_gnn = self.gru_cell(freq_features, edge_index, hidden, edge_attr=self.edge_projector(edge_feature.to(torch.float32)))

        params = self.gate(freq_features_gnn) # 缩放到[0,1]之间

        params = params.reshape(batch_size, 2, 2, 3)
        a = params[..., 0:1]
        b = params[..., 1:2]
        c = params[..., 2:3]
        f = self.freqs.to(innovationError.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        logit = a * f**2 + b * f + c
        smooth_mask = torch.sigmoid(logit)

        # 5. 拆分半gate的实部+虚部，生成对称的另一半gate
        gate_half_real = smooth_mask[:,:, 0,:]  # 半gate实部
        gate_half_imag =  smooth_mask[:,:,1,:]  # 半gate虚部
        # 6. 生成对称的负频率段gate（双边谱对称规则：实部对称，虚部反对称）
        gate_negative_real = torch.flip(gate_half_real[...,1:-1], dims=[-1])
        gate_negative_imag = torch.flip(gate_half_imag[...,1:-1], dims=[-1])

        # 7. 拼接完整的gate实部和虚部
        gate_real = torch.cat([gate_half_real, gate_negative_real], dim=-1)  # 完整实部gate
        gate_imag = torch.cat([gate_half_imag, gate_negative_imag], dim=-1)  # 完整虚部gate

        # 8. 拼接为最终完整的gate（与原freq_features维度一致）
        gate_final = torch.cat([gate_real.unsqueeze(-1), gate_imag.unsqueeze(-1)], dim=-1)  # [batch, 1, 2*freq_dim]

        frequency_spectrum_save_real = frequency_spectrum_real * gate_final[..., 0]
        frequency_spectrum_save_imag = frequency_spectrum_imag * gate_final[..., 1]

        # frequency_spectrum_save_real = frequency_spectrum_real * gate_final[...,0] - frequency_spectrum_imag * gate_final[...,1]
        # frequency_spectrum_save_imag = frequency_spectrum_real * gate_final[...,1] + frequency_spectrum_imag * gate_final[...,0]
        #
        # =================================
        manuver_compensation = torch.fft.ifft(torch.complex(frequency_spectrum_save_real, frequency_spectrum_save_imag),
                                                        n=self.fft_point, dim=-1).real
        # =================================

        # =================================
        # 导出onnx需要
        # frequency_spectrum_save = torch.cat([frequency_spectrum_save_real.unsqueeze(dim=-1),
        #                                      frequency_spectrum_save_imag.unsqueeze(dim=-1)],frequency_spectrum_save_real.ndim)
        # manuver_compensation_output= self.custom_ifft(frequency_spectrum_save, -1)
        # manuver_compensation = (manuver_compensation_output[:,:,:,0]**2 +
        #                         manuver_compensation_output[:,:,:,1]**2)**0.5
        # =================================

        manuver_compensation_latent = self.out_proj(manuver_compensation.reshape(batch_size, -1))
        # manuver_compensation_downsample = manuver_compensation[:, :, :innovationError_reshape.shape[2]]

        # manuver_compensation_downsample = self.temporal_conv(manuver_compensation_downsample)
        return manuver_compensation_latent, gate_final


class GASF(nn.Module):
    def __init__(self, GASF_layer, gnn_hidden_dim,gnn_n_heads=4, gnn_layers=2, gnn_dropout=0.1, gnn_layer="gat"):
        super(GASF, self).__init__()
        cell_list = nn.ModuleList([])
        for i in range(GASF_layer):
            cell_list.append(
                GraphSpectralFilter(gnn_hidden_dim,gnn_n_heads, gnn_layers, gnn_dropout, gnn_layer)
            )

        self.manuverCompensationLayer = cell_list

    def forward(self,normalized_detection_GASF: torch.Tensor, normalized_update_history_GASF: torch.Tensor):
        # # 残差 保留高频分量 高频分量意味着机动残差
        innovationError = (normalized_detection_GASF\
                          - normalized_update_history_GASF[:,:,0::2]).unsqueeze(dim=3)
        x = innovationError
        manuver_compensation_input_list = []
        manuver_compensation_output_list = []
        manuver_compensation_input_list.append(innovationError)
        layer_index = 0
        for layer in self.manuverCompensationLayer:
            output = layer(x)
            manuver_compensation_output_list.append(output)
            x = x - output

        manuver_compensation = torch.stack(manuver_compensation_output_list, dim=0).sum(0)

        return manuver_compensation


def make_cholesky(L_raw, min_diag=1e-3):
    """
    Convert raw lower-triangular parameters to a valid Cholesky factor.
    L_raw: (B, d, d), arbitrary values
    return L: (B, d, d), lower-triangular with positive diagonal
    """
    B, d, _ = L_raw.shape

    # keep only lower triangular
    L = torch.tril(L_raw) # 只保留下三角部分

    # enforce positive diagonal with softplus
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    diag = torch.clamp(F.softplus(diag) + min_diag, max=10.0)


    # replace diagonal
    L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
    return L


def joseph_update(P_pred, K, H, R):
    """
    P_pred: (B, dx, dx)  or (dx, dx)
    K:      (B, dx, dz)  or (dx, dz)
    H:      (B, dz, dx)  or (dz, dx)
    R:      (B, dz, dz)  or (dz, dz)
    """
    dx = P_pred.shape[-1]
    I = torch.eye(dx, device=P_pred.device, dtype=P_pred.dtype)

    if P_pred.dim() == 3:
        I = I.unsqueeze(0).expand(P_pred.shape[0], -1, -1)

    A = I - K @ H
    P_upd = A @ P_pred @ A.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
    return P_upd


class MaskedSubspaceResidual(nn.Module):
    def __init__(self, n_FFT, dt):
        super().__init__()
        self.n_FFT = n_FFT
        self.dt = dt

        self.hidden_dim = int(n_FFT) // 2 + 1

        self.gate = ResMLP(self.hidden_dim, self.hidden_dim * 2 ,
                            self.hidden_dim, 6, 0.1)
        self.sigmoid = nn.Sigmoid()

        self.alpha = nn.Parameter(torch.ones(2))
    def forward(self, R, batch, mask):
        # R: [N,T,D]
        # batch: [N]
        # mask: [N,1,1]

        unique_batches = batch.unique()
        outputs = []

        for b in unique_batches:
            batch_index = (batch == b)
            R_group = R[batch_index]  # [N_b, T, D]
            mask_group = mask[batch_index]

            # 做子空间分解
            R_new = self.group_subspace_residual(R_group, mask_group)

            outputs.append(R_new)

        return torch.cat(outputs, dim=0)

    def group_subspace_residual(self, R, mask):
        N, T, D = R.shape
        R = R.permute(0, 2, 1)

        mask = mask[:,-1,:].unsqueeze(1)
        R_fft = torch.fft.rfft(R, dim=-1, n=self.n_FFT) # [B,D,N,F]
        R_spec = torch.abs(R_fft)

        mask_exp = mask
        valid_counts = mask.sum(dim=0, keepdim=True).clamp(min=1)

        # 去均值
        mean_r = (R_spec * mask_exp).sum(dim=0, keepdim=True) \
                 / valid_counts

        R_centered = (R_spec - mean_r) * mask_exp

        R_centered = R_centered.permute(1, 0, 2)
        # 协方差
        C = torch.matmul(
            R_centered.transpose(-2, -1).conj(),
            R_centered
        ) / valid_counts

        # 特征分解（实对称）
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals = torch.clamp(eigvals, min=0)
        # 门控
        g = self.sigmoid(self.gate(eigvals))

        G = torch.diag_embed(g)

        C_new = eigvecs @ G @ eigvecs.transpose(-2, -1)

        R_new_spec = R_centered @ C_new + mean_r.permute(1, 0, 2)

        phase = torch.angle(R_fft).permute(1, 0, 2)
        R_new_fft = R_new_spec * torch.exp(1j * phase)

        #切换到加速度
        freqs = torch.fft.rfftfreq(self.n_FFT, d=self.dt)  # [Freq]
        omega2 = -(2 * torch.pi * freqs) ** 2            # [Freq]
        omega2 = omega2.view(1, 1, -1)                   # [1,F,1] 用于广播
        R_fft_acc = R_new_fft * omega2

        comp_acc  = torch.fft.irfft(R_fft_acc, dim=-1, n=T).abs().permute(1, 2, 0)

        return comp_acc
