"""
运动模型基类和工具函数，用于减少重复代码

将多个相似的运动模型类重构为更通用的基类实现
"""

import math
import torch
import torch.nn as nn
from config import CONFIG
from .ode_solvers import solvers
from model.MLP import ResMLP


class GroupTrackMotionModelBase(nn.Module):
    """
    群跟踪运动模型基类

    统一处理 SecondOrderNeuralODE_groupTrack 及其变体的通用逻辑
    """

    def __init__(self, solver='rk4', dt=4e-2, n_states=4, mixtures=8, static_f_dim=8,
                 n_hidden=128, n_layers=2, use_SEAN=True, use_GASF=True, *u_lims):
        super().__init__()
        self.dt = dt
        self.mixtures = mixtures
        self.n_states = n_states
        self.static_f_dim = static_f_dim
        self.n_inputs = len(u_lims)
        self.solver = solvers[solver]
        self.n_hidden = n_hidden
        # 输入约束
        self.u_constrain = []
        for u_lim in u_lims:
            self.u_constrain.append(nn.Hardtanh(-u_lim, u_lim))

        # 构建输入转移矩阵
        self.G = nn.Parameter(self._build_input_transition_matrix(), requires_grad=False)

        # 初始化状态转移参数
        self._init_state_parameters()

        self.use_SEAN = bool(use_SEAN)
        self.use_GASF = bool(use_GASF)

        # Neural dynamics backbone.
        self.backbone = ResMLP(self.n_hidden * 2, self.n_hidden, 2, layer_num=3, dropout_rate=0.1)
        self.x_embeding = nn.Linear(self.n_states, self.n_hidden)
        self.config = CONFIG
        self.base_projections = nn.ModuleDict({
            'gnn_model_output': nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.LayerNorm(self.n_hidden),
                nn.SiLU(),
            ),
            'struct_feats': nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.LayerNorm(self.n_hidden),
                nn.SiLU(),
            ),
        })
        self.base_fusion = ResMLP(
            self.n_hidden, self.n_hidden, self.n_hidden, 2, 0.1
        )
        self.base_norm = nn.LayerNorm(self.n_hidden)
        self.gasf_film = nn.Linear(self.n_hidden, 2 * self.n_hidden + 1)
        self.aux_velocity_head = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden // 2),
            nn.SiLU(),
            nn.Linear(self.n_hidden // 2, 2),
        )
        nn.init.zeros_(self.gasf_film.weight)
        nn.init.zeros_(self.gasf_film.bias)
        with torch.no_grad():
            self.gasf_film.bias[-1] = -2.0
    def _build_input_transition_matrix(self):
        """构建输入转移矩阵 G"""
        G = torch.zeros(1, self.mixtures, self.n_states, 2)
        idx_offset = self.n_states - 2

        G[..., 0, 0] = torch.ones(1, self.mixtures) * self.dt**2 / 2
        G[..., 1, 1] = torch.ones(1, self.mixtures) * self.dt**2 / 2
        G[..., idx_offset, 0] = torch.ones(1, self.mixtures) * self.dt
        G[..., idx_offset + 1, 1] = torch.ones(1, self.mixtures) * self.dt
        return G

    def _init_state_parameters(self):
        """初始化状态转移参数矩阵"""
        # m0: x, y 到 dx, dy 的映射
        self.m0 = nn.Parameter(
            torch.zeros(self.n_states, self.n_states // 2), requires_grad=False
        )
        self.m0[2, 0] = self.m0[3, 1] = 1.

        # m1: vy 的特征选择
        self.m1 = nn.Parameter(
            torch.zeros(10, 7), requires_grad=False
        )
        self.m1[0, 0] = self.m1[1, 1] = self.m1[2, 2] = self.m1[3, 3] = 1.
        self.m1[4, 4] = self.m1[6, 5] = self.m1[8, 6] = 1.

        # m2: vx 的特征选择
        self.m2 = nn.Parameter(
            torch.zeros(10, 7), requires_grad=False
        )
        self.m2[0, 0] = self.m2[1, 1] = self.m2[2, 2] = self.m2[3, 3] = 1.
        self.m2[5, 4] = self.m2[7, 5] = self.m2[9, 6] = 1.

    def model_update(self, X, u, static_f):
        """
        状态更新函数 (由子类实现具体逻辑)

        Args:
            X: [batch, mixtures, n_states] 当前状态
            u: [batch, mixtures, n_inputs] 控制输入
            static_f: 静态特征

        Returns:
            dX: [batch, mixtures, n_states] 状态导数
        """
        raise NotImplementedError("Subclasses must implement model_update")

    def _compute_dxdt(self, X):
        """计算位置变化率 dx/dt"""
        scale = self.config['S_VEL'] / self.config['S_POS']
        return scale * (X @ self.m0)

    def _extract_inputs(self, inputs):
        """Fuse base motion/structure features and modulate them with GASF."""
        base_features = [
            self.base_projections['gnn_model_output'](
                inputs['gnn_model_output']
            )
        ]
        if self.use_SEAN and 'struct_feats' in inputs:
            base_features.append(
                self.base_projections['struct_feats'](inputs['struct_feats'])
            )
        base_feature = self.base_fusion(torch.stack(base_features, dim=0).sum(dim=0))

        node_count = base_feature.size(0)
        film_gate = torch.zeros(node_count, 1, device=base_feature.device)
        auxiliary_delta_v = torch.zeros(node_count, 2, device=base_feature.device)
        if self.use_GASF and 'GASF_output_fusion' in inputs:
            gasf_feature = inputs['GASF_output_fusion']
            film_parameters = self.gasf_film(gasf_feature)
            gamma, beta, gate_logit = torch.split(
                film_parameters, [self.n_hidden, self.n_hidden, 1], dim=-1
            )
            confidence = inputs.get('GASF_confidence')
            film_gate = torch.sigmoid(gate_logit)
            if confidence is not None:
                film_gate = film_gate * confidence
            modulation = torch.tanh(gamma) * self.base_norm(base_feature) + beta
            base_feature = base_feature + film_gate * modulation
            auxiliary_delta_v = self.aux_velocity_head(gasf_feature)

        inputs['film_gate'] = film_gate
        inputs['auxiliary_delta_v'] = auxiliary_delta_v
        return base_feature

    def _compute_acceleration(self, inputs, X):
        """计算加速度 (由子类实现)"""
        raise NotImplementedError("Subclasses must implement _compute_acceleration")

    def forward(self, past_state, inputs, static_f):
        """前向传播"""

        next_state = self.solver(
            self.model_update, past_state, inputs, static_f, self.dt
        )
        return next_state, inputs

    def state_transition_matrix(self, X, inp, static_f=None):
        """Jacobian of the same RK4 flow used for the state prediction.

        Decoder conditioning is node-wise rather than mixture-wise.  Slicing it
        here avoids differentiating unrelated nodes and keeps the Jacobian shape
        compatible with the EKF covariance tensor ``[N, M, D, D]``.
        """
        node_count, mixtures, state_dim = X.shape
        dynamic_inputs = {
            key: value
            for key, value in inp.items()
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == node_count
        }
        rows = []
        for node_index in range(node_count):
            node_inputs = {
                key: value[node_index:node_index + 1]
                for key, value in dynamic_inputs.items()
            }
            mixture_rows = []
            for mixture_index in range(mixtures):
                def one_step(state):
                    state_batch = state.view(1, 1, state_dim)
                    return solvers['rk4'](
                        self.model_update, state_batch, dict(node_inputs), static_f, self.dt
                    ).view(state_dim)

                mixture_rows.append(
                    torch.func.jacrev(one_step)(X[node_index, mixture_index])
                )
            rows.append(torch.stack(mixture_rows, dim=0))
        F = torch.stack(rows, dim=0)
        return F, F.transpose(-1, -2)

    def input_transition_matrix(self, X, u):
        """计算输入转移矩阵"""
        batch_size = X.size(0)
        G = self.G.clone().expand(batch_size, -1, -1, -1)
        return G, torch.transpose(G, dim0=-2, dim1=-1)


class GroupTrackFullModel(GroupTrackMotionModelBase):
    """完整的群跟踪模型 (包含结构加速度和机动补偿)"""

    def model_update(self, X, u, static_f):
        feature_fusion = self._extract_inputs(u)
        feature_fusion = feature_fusion.unsqueeze(1).expand(-1, X.size(1), -1)
        inp = torch.cat((self.x_embeding(X), feature_fusion), dim=-1)

        acc = self.backbone(inp)

        dvx = acc[..., 0:1]
        dvy = acc[..., 1:2]

        dxdt = self._compute_dxdt(X)
        dX = torch.cat((dxdt, dvx, dvy), dim=-1)
        return dX


# 向后兼容的别名
SecondOrderNeuralODE_groupTrack = GroupTrackFullModel
