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
    TemporalProbabilisticSEAN,
)

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
        self.use_SEAN = bool(use_SEAN)
        self.motion_model = motion_model
        self.input_size = motion_model.n_states
        self.output_size = z_dimension
        self.mixtures = motion_model.mixtures
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

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

        if self.use_SEAN:
            self.SEAN = TemporalProbabilisticSEAN(hidden_size, hidden_size)
        else:
            self.SEAN = None
        if self.use_GASF:
            self.GASF = JointGraphTemporalSpectralFilter(
                hidden_size,
                windows=tuple(int(value) for value in residual_length),
                signal_dim=z_dimension,
                chebyshev_order=2,
            )
        else:
            self.GASF = None

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

    def forward(self, data):
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
        ) = data
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

        valid_nodes = nan_mask.reshape(nan_mask.size(0), -1).all(dim=-1, keepdim=True)
        if self.use_SEAN:
            structure = self.SEAN(batch, past_state, hidden, valid_nodes)
            structural_features = structure.structural_features
            learned_edge_index = structure.edge_index
            learned_edge_weight = structure.edge_weight
            node_entropy = structure.node_entropy
            adjacency_vector = structure.adjacency_vector
            adjacency_mask = structure.vector_mask
            group_count = structure.group_count
        else:
            structural_features = torch.zeros_like(hidden)
            learned_edge_index = edge_index
            learned_edge_weight = projected_edge
            node_entropy = torch.ones(batch_size, 1, device=hidden.device)
            adjacency_vector = hidden.new_empty(0)
            adjacency_mask = torch.empty(0, dtype=torch.bool, device=hidden.device)
            group_count = 0

        if self.use_GASF:
            spectral = self.GASF(
                residual,
                learned_edge_index,
                learned_edge_weight,
                node_entropy,
                nan_mask,
            )
            gasf_features = spectral.latent
            gasf_confidence = spectral.confidence
            spectral_weights = spectral.scale_weights
        else:
            gasf_features = torch.zeros_like(hidden)
            gasf_confidence = torch.zeros(batch_size, 1, device=hidden.device)
            spectral_weights = torch.zeros(batch_size, 0, device=hidden.device)

        generated = self.dropout(
            F.leaky_relu(
                self.generator(F.leaky_relu(hidden, self.alpha)), self.alpha
            )
        )
        x1, x2, x3, x4 = torch.split(generated, self.hidden_size, dim=-1)
        scale_features = structural_features if self.use_SEAN else hidden
        process_noise = self.process_noise_matrix(
            x1, x2, x3, batch_size, scale_features
        )

        model_input = {'gnn_model_output': x4}
        if self.use_SEAN:
            model_input['struct_feats'] = structural_features
        if self.use_GASF:
            model_input['GASF_output_fusion'] = gasf_features
            model_input['GASF_confidence'] = gasf_confidence

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
