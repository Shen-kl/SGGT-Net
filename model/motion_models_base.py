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

        # 神经网络 backbone
        self.backbone = ResMLP(self.n_hidden * 2, self.n_hidden, 2, layer_num=3, dropout_rate=0.1)
        self.x_embeding = nn.Linear(self.n_states, self.n_hidden)
        feature_cnt = 1 + int(use_SEAN) + int(use_GASF)
        self.alpha_net = nn.Linear(self.n_hidden * feature_cnt, feature_cnt)
        self.config = CONFIG
        self.proj = [nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.LayerNorm(self.n_hidden), nn.LeakyReLU()
        ) for _ in range(feature_cnt)]
        self.proj_res = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.LayerNorm(self.n_hidden), nn.LeakyReLU()
        )
        self.proj_struct = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.LayerNorm(self.n_hidden), nn.LeakyReLU()
        )
        self.gate = nn.Sequential(
            nn.Linear(self.n_hidden * feature_cnt, self.n_hidden * feature_cnt),
            nn.Sigmoid()  # 每个通道独立0~1
        )
        self.fusion =ResMLP(self.n_hidden,self.n_hidden,self.n_hidden,2,0.1)
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

    def _extract_inputs(self, u):
        """
        从控制输入中提取各分量

        """
        v_list = []
        for idx, (k, v) in enumerate(u.items()):
            v_list.append(self.proj[idx](v))
        v_stack = torch.cat(v_list, dim=-1)

        alpha = self.gate(v_stack)
        gated = alpha * v_stack

        v_fusion = self.fusion(gated[...,:self.n_hidden]+gated[...,self.n_hidden:2*self.n_hidden]+gated[...,2*self.n_hidden:3*self.n_hidden])


        return v_fusion

    def _compute_acceleration(self, inputs, X):
        """计算加速度 (由子类实现)"""
        raise NotImplementedError("Subclasses must implement _compute_acceleration")

    def forward(self, past_state, inputs, static_f):
        """前向传播"""

        next_state = self.solver(
            self.model_update, past_state, inputs, static_f, self.dt
        )
        return next_state, inputs

    def state_transition_matrix(self, X, inp, static_f):
        """计算状态转移矩阵 (用于雅可比)"""
        N, m, feat_dim = X.shape

        def fx(state, inputs, static_input):
            return solvers['rk4'](
                self.model_update, state, inputs, static_input, self.dt
            )

        jacobian_rev = torch.func.jacrev(fx, argnums=0)
        J = torch.func.vmap(jacobian_rev, randomness='same')(
            X.flatten(0, 1), inp.flatten(0, 1), static_f.flatten(0, 1)
        )
        F = J.view(N, m, feat_dim, feat_dim)
        return F, torch.transpose(F, dim0=-2, dim1=-1)

    def input_transition_matrix(self, X, u):
        """计算输入转移矩阵"""
        batch_size = X.size(0)
        G = self.G.clone().expand(batch_size, -1, -1, -1)
        return G, torch.transpose(G, dim0=-2, dim1=-1)


class GroupTrackFullModel(GroupTrackMotionModelBase):
    """完整的群跟踪模型 (包含结构加速度和机动补偿)"""

    def model_update(self, X, u, static_f):
        feature_fusion = self._extract_inputs(u)
        inp = torch.cat((self.x_embeding(X).squeeze(), feature_fusion), dim=-1)

        acc = self.backbone(inp)

        dvx = acc[..., 0:1].unsqueeze(-1)
        dvy = acc[..., 1:2].unsqueeze(-1)

        dxdt = self._compute_dxdt(X)
        dX = torch.cat((dxdt, dvx, dvy), dim=-1)
        return dX


# 向后兼容的别名
SecondOrderNeuralODE_groupTrack = GroupTrackFullModel

