"""
卡尔曼滤波器工具类模块

将卡尔曼滤波相关的复杂计算逻辑提取到独立模块中，提高代码可维护性
"""

import torch
from config import CONFIG


class MatrixBuilder:
    """矩阵构建工具类"""

    @staticmethod
    def build_scaling_matrix(device=None):
        """构建位置-速度缩放矩阵"""
        scale_values = [
            CONFIG['S_POS'], CONFIG['S_POS'],
            CONFIG['S_VEL'], CONFIG['S_VEL']
        ]
        return torch.diag_embed(torch.tensor(scale_values, device=device))

    @staticmethod
    def build_cv_transition_matrix(T, batch_size, device=None):
        """
        构建恒定速度状态转移矩阵

        Args:
            T: 时间步长
            batch_size: 批次大小
            device: 设备

        Returns:
            F_cv: [batch_size, 4, 4] 恒定速度转移矩阵
        """
        F_cv = torch.tensor([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        return F_cv.repeat([batch_size, 1, 1, 1])

    @staticmethod
    def build_observation_matrix(device=None):
        """构建观测矩阵 H: [2, 4]"""
        H = torch.zeros(2, 4, device=device)
        H[0, 0] = H[1, 1] = 1.0
        return H


class CovarianceUtils:
    """协方差矩阵工具类"""

    @staticmethod
    def ensure_symmetry(P, eps=1e-7):
        """确保协方差矩阵对称性"""
        P_sym = (P + P.transpose(-1, -2)) / 2
        # 添加小的正则化项确保正定性
        P_reg = P_sym + eps * torch.eye(P.shape[-1], device=P.device)
        return P_reg

    @staticmethod
    def init_covariance_from_R(R, delta_T, device=None):
        """
        从测量噪声协方差矩阵 R 初始化状态协方差矩阵 P

        Args:
            R: [batch, 2, 2] 测量噪声协方差
            delta_T: 时间步长
            device: 设备

        Returns:
            P: [batch, 4, 4] 状态协方差矩阵
        """
        batch = R.shape[0]
        P_matlab = torch.zeros(batch, 4, 4, device=device)

        # 填充 P 矩阵 (根据运动学关系)
        P_matlab[..., 0, 0] = R[..., 0, 0]
        P_matlab[..., 0, 1] = R[..., 0, 0] / delta_T
        P_matlab[..., 0, 2] = R[..., 0, 1]
        P_matlab[..., 0, 3] = R[..., 0, 1] / delta_T

        P_matlab[..., 1, 0] = R[..., 0, 0] / delta_T
        P_matlab[..., 1, 1] = 2 * R[..., 0, 0] / delta_T**2
        P_matlab[..., 1, 2] = R[..., 0, 1] / delta_T
        P_matlab[..., 1, 3] = 2 * R[..., 0, 1] / delta_T**2

        P_matlab[..., 2, 0] = R[..., 0, 1]
        P_matlab[..., 2, 1] = R[..., 0, 1] / delta_T
        P_matlab[..., 2, 2] = R[..., 1, 1]
        P_matlab[..., 2, 3] = R[..., 1, 1] / delta_T

        P_matlab[..., 3, 0] = R[..., 0, 1] / delta_T
        P_matlab[..., 3, 1] = 2 * R[..., 0, 1] / delta_T**2
        P_matlab[..., 3, 2] = R[..., 1, 1] / delta_T
        P_matlab[..., 3, 3] = 2 * R[..., 1, 1] / delta_T**2

        # 转换到 [x, vx, y, vy] 顺序
        T_mat = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=P_matlab.dtype, device=device)

        return T_mat @ P_matlab @ T_mat.transpose(-1, -2)


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器"""

    def __init__(self, delta_T):
        """
        Args:
            delta_T: 时间步长
        """
        self.delta_T = delta_T
        self.H = MatrixBuilder.build_observation_matrix()

    def time_update(self, P_t, G_t, q_t, device=None, F_t=None):
        """
        时间更新步骤

        Args:
            P_t: [batch, 4, 4] 当前协方差
            G_t: [batch, 4, 2] 输入转移矩阵
            q_t: [batch, 4, 4] 过程噪声
            device: 设备

        Returns:
            P_next: [batch, 4, 4] 预测协方差
            Q_t: [batch, 4, 4] 过程噪声
        """
        batch_size = P_t.shape[0]

        # The learned flow Jacobian is already expressed in normalized state
        # coordinates.  V1 callers retain the scaled constant-velocity fallback.
        D = MatrixBuilder.build_scaling_matrix(device)
        D_inv = torch.inverse(D)
        if F_t is None:
            F_cv = MatrixBuilder.build_cv_transition_matrix(self.delta_T, batch_size, device)
            F_t = D_inv @ F_cv @ D

        # 计算过程噪声
        Q_t = D_inv @ G_t @ q_t @ G_t.transpose(-1, -2) @ D_inv.transpose(-1, -2)
        Q_t = CovarianceUtils.ensure_symmetry(Q_t, eps=0)

        # 时间更新
        P_next = F_t @ P_t @ F_t.transpose(-1, -2) + Q_t
        P_next = CovarianceUtils.ensure_symmetry(P_next, eps=1e-7)

        return P_next, Q_t

    def measurement_update(self, X_pred, P_pred, Z, R):
        """
        测量更新步骤

        Args:
            X_pred: [batch, 1, 4] 预测状态
            P_pred: [batch, 4, 4] 预测协方差
            Z: [batch, 1, 2] 测量值
            R: [batch, 2, 2] 测量噪声

        Returns:
            X_update: [batch, 1, 4] 更新后的状态
            P_update: [batch, 4, 4] 更新后的协方差
            innovation: [batch, 2, 1] 新息
            S: [batch, 2, 2] 新息协方差
        """
        batch = P_pred.shape[0]
        device = P_pred.device
        dx = P_pred.shape[-1]
        I = torch.eye(dx, device=device, dtype=P_pred.dtype)

        # 构建归一化的观测矩阵
        D = MatrixBuilder.build_scaling_matrix(device)
        H_bar = self.H @ D
        R_renorm = R * (CONFIG['S_POS'])**2

        # 扩展到批次维度
        H_batch = self.H.unsqueeze(0).expand(batch, -1, -1)
        H_bar_batch = H_bar.unsqueeze(0).expand(batch, -1, -1)

        # 计算新息
        innovation = CONFIG['S_POS'] * (Z - X_pred @ self.H.T)

        # 计算新息协方差
        HP = torch.bmm(H_bar_batch, P_pred)
        S = torch.bmm(HP, H_bar_batch.transpose(1, 2)) + R_renorm

        # 计算卡尔曼增益
        PHt = torch.bmm(P_pred, H_bar_batch.transpose(1, 2))
        X = torch.linalg.solve(S, PHt.transpose(1, 2))
        K = X.transpose(1, 2).clone()

        # 状态和协方差更新
        I = I.unsqueeze(0).expand(P_pred.shape[0], -1, -1)
        dx_update = torch.bmm(K, innovation.permute(0, 2, 1)).squeeze(-1)
        X_update = X_pred + dx_update.unsqueeze(1)

        A = I - K @ H_bar
        P_update = A @ P_pred @ A.transpose(-1, -2) + K @ R_renorm @ K.transpose(-1, -2)
        P_update = CovarianceUtils.ensure_symmetry(P_update, eps=0)

        return X_update, P_update, innovation, S


def gmm_to_single_gaussian(x, P, pi, eps=1e-9):
    """
    将高斯混合模型转换为单个高斯分布

    Args:
        x: [B, M, dx] 混合模型均值
        P: [B, M, dx, dx] 混合模型协方差
        pi: [B, M] 混合权重 (logits 或 probabilities)
        eps: 数值稳定性的小量

    Returns:
        x_bar: [B, dx] 合并后的均值
        P_bar: [B, dx, dx] 合并后的协方差
    """
    # 归一化权重
    pi = torch.softmax(pi, dim=-1) if pi.min() < 0 or pi.max() > 1 else pi
    pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)

    # 合并均值
    x_bar = torch.sum(pi.unsqueeze(-1) * x, dim=1)

    # 合并协方差
    dx = x.shape[-1]
    diff = x - x_bar.unsqueeze(1)
    outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)

    P_bar = torch.sum(pi.unsqueeze(-1).unsqueeze(-1) * (P + outer), dim=1)

    return x_bar, P_bar
