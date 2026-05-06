"""
训练工具模块

将训练循环中的常用逻辑提取到独立函数中，提高代码可维护性
"""

import numpy as np
import torch
from torch_geometric.data import Data
from config import CONFIG


class MeasurementNoiseCalculator:
    """测量噪声计算工具"""

    @staticmethod
    def enu2aer(enu_x, enu_y, enu_z):
        """
        ENU 坐标转换为 AER (Azimuth, Elevation, Range)

        Args:
            enu_x, enu_y, enu_z: ENU 坐标

        Returns:
            azi: 方位角 (度)
            elev: 仰角 (度)
            range: 距离
        """
        azi = torch.atan2(enu_x, enu_y) * 180 / np.pi
        elev = torch.atan2(enu_z, (enu_x**2 + enu_y**2)**0.5) * 180 / np.pi
        range_val = (enu_x**2 + enu_y**2 + enu_z**2)**0.5

        return azi, elev, range_val


class CovarianceNormalizer:
    """协方差矩阵归一化工具"""

    @staticmethod
    def normalize_measurement_cov(R, config):
        """
        归一化测量噪声协方差矩阵

        Args:
            R: [batch, 2, 2] 测量噪声协方差
            config: 配置字典

        Returns:
            R_norm: 归一化后的协方差
        """
        return R / (config['S_POS'])**2

    @staticmethod
    def normalize_state_cov(P, config):
        """
        归一化状态协方差矩阵

        Args:
            P: [batch, 4, 4] 状态协方差
            config: 配置字典

        Returns:
            P_norm: 归一化后的协方差
        """
        P_norm = P.clone()
        # 归一化位置分量
        P_norm[..., [0, 1], :] = P_norm[..., [0, 1], :] / config['S_POS']
        P_norm[..., :, [0, 1]] = P_norm[..., :, [0, 1]] / config['S_POS']
        # 归一化速度分量
        P_norm[..., [2, 3], :] = P_norm[..., [2, 3], :] / config['S_VEL']
        P_norm[..., :, [2, 3]] = P_norm[..., :, [2, 3]] / config['S_VEL']
        return P_norm


class GraphDataProcessor:
    """图数据处理器"""

    @staticmethod
    def init_input_window(graph_data, config, args):
        """
        初始化输入窗口数据

        Args:
            graph_data: 原始图数据
            config: 配置字典
            args: 参数对象

        Returns:
            graph_data_tmp: 处理后的图数据
        """
        from utils.minmax_scaler import minmax_scaler

        batch = graph_data.x.shape[0]
        graph_data_tmp = graph_data.clone()

        # 裁剪到输入窗口
        graph_data_tmp.x = graph_data_tmp.x[:, :config['input_graph_window_len'], :]
        graph_data_tmp.y = graph_data_tmp.y[:, config['input_graph_window_len']:
                                          config['input_graph_window_len'] + config['output_graph_window_len'], :]
        graph_data_tmp.measurement = graph_data_tmp.measurement[:, :config['input_graph_window_len'], :]

        graph_data_tmp.measurement_future = graph_data.measurement[:, config['input_graph_window_len']:
                                                       config['input_graph_window_len'] + config['output_graph_window_len'], :]

        # 初始化残差
        zeros_len = np.max(args.decoder_residual_length) - config['input_graph_window_len']
        residual_init = torch.cat([
            torch.zeros([batch, zeros_len, config['z_dimension']]),
            graph_data_tmp.measurement - graph_data_tmp.x[:, :, [0, 2]]
        ], dim=1)
        graph_data_tmp.residual = residual_init

        # 裁剪边索引和特征
        graph_data_tmp.edge_index = graph_data_tmp.edge_index[:config['input_graph_window_len']]
        graph_data_tmp.edge_features = graph_data_tmp.edge_features[:config['input_graph_window_len']]

        graph_data_tmp.tar_edge_index = graph_data_tmp.tar_edge_index[config['input_graph_window_len']:
                                                                  config['input_graph_window_len'] + config['output_graph_window_len']]
        graph_data_tmp.tar_edge_features = graph_data_tmp.tar_edge_features[config['input_graph_window_len']:
                                                                      config['input_graph_window_len'] + config['output_graph_window_len']]

        # 更新掩码
        graph_data_tmp.nan_mask = graph_data_tmp.nan_mask[:, :config['input_graph_window_len'], :]
        graph_data_tmp.real_mask = graph_data.real_mask[:, config['input_graph_window_len']:
                                                        config['input_graph_window_len'] + config['output_graph_window_len'], :]

        return graph_data_tmp

    @staticmethod
    def update_input_window(graph_data_tmp, graph_data, config, x_update_renorm, bias, args):
        """
        更新输入窗口数据 (滑动窗口)

        Args:
            graph_data_tmp: 当前处理的图数据
            graph_data: 原始图数据
            config: 配置字典
            x_update_renorm: 更新后的状态
            bias: 时间步偏移
            args: 参数对象

        Returns:
            graph_data_tmp: 更新后的图数据
        """
        batch = graph_data.x.shape[0]
        graph_data.x[:, bias + CONFIG['input_graph_window_len'] - 1, :] = x_update_renorm.squeeze()
        # 滑动窗口: 移除最早的帧，添加最新的帧
        graph_data_tmp.x[:, :-1, :] = graph_data_tmp.x[:, 1:, :]
        graph_data_tmp.x[:, -1, :] = x_update_renorm.squeeze(dim=1)

        # 更新目标
        start_idx = bias + config['input_graph_window_len']
        graph_data_tmp.y = graph_data.y[:, start_idx:
                                        start_idx + config['output_graph_window_len'], :]

        # 更新测量
        graph_data_tmp.measurement = graph_data.measurement[:, bias:config['input_graph_window_len'] + bias, :]
        graph_data_tmp.measurement_future = graph_data.measurement[:, start_idx:
                                                                   start_idx + config['output_graph_window_len'], :]

        # 更新残差
        if bias + config['input_graph_window_len'] < np.max(args.decoder_residual_length):
            zeros_len = np.max(args.decoder_residual_length) - config['input_graph_window_len'] - bias
            graph_data_tmp.residual[:, :zeros_len, :].zero_()
            graph_data_tmp.residual[:, zeros_len:, :] = (
                graph_data.measurement[:, :config['input_graph_window_len'] + bias, :] -
                graph_data.x[:, :config['input_graph_window_len'] + bias, [0, 2]]
            )
        else:
            begin_idx = bias + config['input_graph_window_len'] - np.max(args.decoder_residual_length)
            data = (
                graph_data.measurement[:, begin_idx:config['input_graph_window_len'] + bias, :] -
                graph_data.x[:, begin_idx:config['input_graph_window_len'] + bias, [0, 2]]
            )
            graph_data_tmp.residual[:, -data.shape[1]:, :] = data

        # 更新边索引和特征
        graph_data_tmp.edge_index = graph_data.edge_index[bias:config['input_graph_window_len'] + bias]
        graph_data_tmp.edge_features = graph_data.edge_features[bias:config['input_graph_window_len'] + bias]

        graph_data_tmp.tar_edge_index = graph_data.tar_edge_index[start_idx:
                                                                  start_idx + config['output_graph_window_len']]
        graph_data_tmp.tar_edge_features = graph_data.tar_edge_features[start_idx:
                                                                       start_idx + config['output_graph_window_len']]

        # 更新掩码
        graph_data_tmp.nan_mask = graph_data.nan_mask[:, bias:config['input_graph_window_len'] + bias, :]
        graph_data_tmp.real_mask = graph_data.real_mask[:, start_idx:
                                                           start_idx + config['output_graph_window_len'], :]

        return graph_data_tmp

    @staticmethod
    def init_covariance_from_measurement(measureNoiseCov, delta_T, config, device=None):
        """
        从测量噪声初始化状态协方差

        Args:
            measureNoiseCov: 测量噪声协方差
            delta_T: 时间步长
            config: 配置字典
            device: 设备

        Returns:
            P_predict: 初始化的状态协方差
        """
        batch = measureNoiseCov.shape[0]

        # 计算基础 P 矩阵
        P_matlab = torch.zeros(batch, 4, 4, device=device)
        R = measureNoiseCov

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

        P_init = T_mat @ P_matlab @ T_mat.transpose(-1, -2)

        # 归一化
        P_init_norm = CovarianceNormalizer.normalize_state_cov(P_init, config)

        return P_init_norm * 1e-1


class TrainingMetrics:
    """训练指标收集器"""

    def __init__(self):
        self.metrics = {
            'loss': [],
            'location_rmse_prediction': [],
            'velocity_rmse_prediction': [],
            'location_rmse_update': [],
            'velocity_rmse_update': [],
            'location_rmse_measurement': [],
            'f1': [],
        }

    def add_batch_metrics(self, batch_metrics):
        """
        添加批次指标

        Args:
            batch_metrics: 包含各项指标的字典
        """
        for key, value in batch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_batch_mean(self):
        """获取当前批次的平均指标"""
        return {key: np.mean(values) if values else 0.0
                for key, values in self.metrics.items()}

    def reset(self):
        """重置所有指标"""
        for key in self.metrics:
            self.metrics[key] = []

    def get_epoch_mean(self):
        """获取整个 epoch 的平均指标"""
        return {key: np.mean(values) if values else 0.0
                for key, values in self.metrics.items()}

    def get_all_values(self, key):
        """获取指定指标的所有值"""
        return self.metrics.get(key, [])
