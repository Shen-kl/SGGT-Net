"""
优化后的训练器模块

主要改进:
1. 模块化设计，提取重复逻辑到工具类
2. 分离训练和验证逻辑
3. 简化主训练循环
4. 移除注释代码
5. 改进可读性和可维护性
"""

import os
import time
import math
import shutil
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from scipy.special import expit
from torch_geometric.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from train.training_utils import (
    MeasurementNoiseCalculator,
    CovarianceNormalizer,
    GraphDataProcessor,
    TrainingMetrics
)
from utils.minmax_scaler import minmax_scaler
from utils.losses import *
from utils.common.caculate_measurement_noise_cov import calculate_measurement_noise_cov


class Trainer:
    """训练器类"""

    def __init__(self, model, optimizer, lr_schedule, logger, config, args):
        """
        Args:
            model: 模型
            optimizer: 优化器
            lr_schedule: 学习率调度器
            logger: 日志记录器
            config: 配置字典
            args: 训练参数
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.logger = logger
        self.config = config
        self.args = args
        self.device = torch.device(args.device)

        # 初始化损失函数
        self.wta_loss = EWTALoss()
        self.nll_loss = NLLMDNLoss()
        self.nll_loss_single = LossCompute_NLL()

        # 训练参数
        self.train_epoches = int(args.train_epochs if hasattr(args, 'train_epochs') else 100)
        self.wta_epochs = int(self.train_epoches // 20)
        self.warm_epochs = int(self.train_epoches - (self.train_epoches // 8))
        self.P_change_epochs = int(self.train_epoches // 3)

        # Teacher forcing 参数
        self.tf_prob_start = 1.0
        self.tf_prob_min = 0.0
        self.tf_decay_epochs = 20

        # 状态
        self.min_evaluation = 1e9
        self.best_model_path = None
        self.full_model_unfrozen = False

    def get_tf_prob(self, current_epoch):
        """计算当前 epoch 的 teacher forcing 概率"""
        return max(
            self.tf_prob_min,
            self.tf_prob_start - current_epoch / self.tf_decay_epochs
        )

    def compute_measurement_noise(self, graph_data, time_index=0):
        """
        计算测量噪声协方差矩阵

        Args:
            graph_data: 图数据

        Returns:
            measureNoiseCov: 测量噪声协方差
            measureNoiseCov_norm: 归一化的测量噪声协方差
        """
        # 转换坐标
        measurement = graph_data.measurement[:, time_index].detach().cpu()
        azi, elev, distance = MeasurementNoiseCalculator.enu2aer(
            measurement[:, 0],
            measurement[:, 1],
            torch.zeros_like(measurement[:, 0])
        )

        # 计算噪声协方差
        mu, measureNoiseCov = calculate_measurement_noise_cov(
            distance, azi, elev,
            CONFIG['measurement_noise_std'][0], CONFIG['measurement_noise_std'][1], CONFIG['measurement_noise_std'][2]
        )

        # 转换为 PyTorch 张量并裁剪
        measureNoiseCov = torch.from_numpy(measureNoiseCov).permute(2, 0, 1)
        measureNoiseCov = measureNoiseCov[:, :self.config['z_dimension'],
                                            :self.config['z_dimension']]

        # 归一化
        measureNoiseCov_norm = CovarianceNormalizer.normalize_measurement_cov(
            measureNoiseCov, self.config
        )

        return measureNoiseCov, measureNoiseCov_norm

    def init_state_covariance(self, measureNoiseCov):
        """
        初始化状态协方差矩阵

        Args:
            measureNoiseCov: 测量噪声协方差

        Returns:
            P_predict: 初始化的状态协方差
        """
        return GraphDataProcessor.init_covariance_from_measurement(
            measureNoiseCov, self.args.T, self.config, device=measureNoiseCov.device
        )

    def _max_window_start(self, graph_data, rollout_steps):
        return max(
            0,
            graph_data.y.shape[1]
            - self.config['input_graph_window_len']
            - self.config['output_graph_window_len']
            - rollout_steps,
        )

    def sample_training_starts(self, graph_data, current_epoch, batch_idx):
        """Deterministically mix uniform and change-centered rollout windows."""
        rollout_steps = min(
            self.args.rollout_steps,
            graph_data.y.shape[1] - self.config['input_graph_window_len']
            - self.config['output_graph_window_len'],
        )
        max_start = self._max_window_start(graph_data, rollout_steps)
        count = 1
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.args.seed + current_epoch * 1000003 + batch_idx * 9176)

        changes = graph_data.param_change_time.detach().cpu().reshape(-1)
        changes = changes[(changes > 1) & (changes < graph_data.y.shape[1])]
        starts = []
        for index in range(count):
            choose_event = (
                changes.numel() > 0
                and torch.rand((), generator=generator).item()
                < self.args.event_window_fraction
            )
            if choose_event:
                change = int(changes[torch.randint(changes.numel(), (1,), generator=generator)].item())
                center = change - self.config['input_graph_window_len'] - rollout_steps // 2
                jitter = int(torch.randint(-rollout_steps, rollout_steps + 1, (1,), generator=generator).item())
                start = center + jitter
            else:
                start = int(torch.randint(max_start + 1, (1,), generator=generator).item())
            starts.append(max(0, min(max_start, start)))
        return starts, rollout_steps

    def validation_starts(self, graph_data):
        """Use a fixed, evenly spaced window list for comparable validation."""
        rollout_steps = min(
            self.args.rollout_steps,
            graph_data.y.shape[1] - self.config['input_graph_window_len']
            - self.config['output_graph_window_len'],
        )
        max_start = self._max_window_start(graph_data, rollout_steps)
        return [max_start // 2], rollout_steps

    def compute_loss(self, outputs, graph_data, time_step, previous_adjacency=None,
                     previous_adjacency_mask=None, use_wta_loss=True):
        """
        计算训练损失

        Args:
            outputs: 模型输出
            graph_data: 图数据
            time_step： 时间步
            use_wta_loss: 是否使用 WTA 损失

        Returns:
            loss: 总损失
            loss_predict: 预测损失
            loss_update: 更新损失
            loss_struct: 结构损失
        """

        all_states_prediction, all_Ps_prediction, mixture_coeffs, real_mask, target = outputs[:5]
        x_update = outputs[5]
        dec_mask = outputs[3]
        A_vec = outputs[8]
        all_A_vec_mask = outputs[9]

        # 预测损失
        n_mixtures = mixture_coeffs.shape[1]

        state_weights = target.new_tensor([
            1.0, 1.0,
            getattr(self.args, 'velocity_loss_weight', 1.0),
            getattr(self.args, 'velocity_loss_weight', 1.0),
        ])
        loss_predict = self.wta_loss(
            all_states_prediction, target, dec_mask, n_mixtures, state_weights
        )

        # 更新损失
        loss_update = self.wta_loss(
            x_update.unsqueeze(1), target[:, 0:1, :],
            dec_mask[:, 0:1], 1, state_weights
        )

        # 结构损失
        # 计算 分类损失
        zero = loss_predict.new_zeros(())
        loss_struct = zero
        if A_vec.numel() > 0:
            struct_target = graph_data.struct_feat[time_step].to(A_vec).reshape(-1)
            if struct_target.numel() == A_vec.size(0):
                struct_target = struct_target.view(-1, 1).expand_as(A_vec)
                loss_struct = focal_loss(
                    A_vec.clamp(1e-6, 1 - 1e-6), struct_target, all_A_vec_mask
                )

        loss_temporal = zero
        if previous_adjacency is not None and previous_adjacency.shape == A_vec.shape:
            temporal_mask = all_A_vec_mask & previous_adjacency_mask
            if temporal_mask.any():
                loss_temporal = torch.nn.functional.smooth_l1_loss(
                    A_vec[temporal_mask], previous_adjacency[temporal_mask]
                )

        loss_aux = zero
        if outputs.auxiliary_delta_v.numel() > 0 and getattr(self.model.decoder, 'use_GASF', False):
            # Raw dataset order is [x, vx, y, vy]; convert the one-step
            # velocity increment to the normalized internal coordinates.
            velocity_delta = (
                graph_data.y[:, time_step:time_step + 1, [1, 3]]
                - graph_data.x[:, time_step - 1:time_step, [1, 3]]
            ).to(outputs.auxiliary_delta_v) / self.config['S_VEL']
            maneuver_weight = 1.0 + velocity_delta.norm(dim=-1).clamp(max=4.0)
            auxiliary_error = torch.nn.functional.smooth_l1_loss(
                outputs.auxiliary_delta_v, velocity_delta, reduction='none'
            ).mean(dim=-1)
            loss_aux = (
                auxiliary_error * maneuver_weight * dec_mask
            ).sum() / dec_mask.sum().clamp_min(1.0)

        loss_nll = zero
        if outputs.covariances.numel() > 0:
            difference = (outputs.target.unsqueeze(2) - outputs.states).unsqueeze(-1)
            covariance = outputs.covariances
            eye = torch.eye(covariance.size(-1), device=covariance.device)
            covariance = covariance + 1e-5 * eye
            solution = torch.linalg.solve(covariance, difference)
            mahalanobis = (difference.transpose(-1, -2) @ solution).squeeze(-1).squeeze(-1)
            logdet = torch.linalg.slogdet(covariance).logabsdet
            state_dim = covariance.size(-1)
            gaussian_constant = state_dim * math.log(2.0 * math.pi)
            # Report and optimize NLL per state dimension. Including the
            # Gaussian constant makes the value comparable across dimensions;
            # the small lambda keeps calibration secondary to trajectory error.
            best_nll = (
                0.5 * (mahalanobis + logdet + gaussian_constant) / state_dim
            ).min(dim=2).values
            best_nll = best_nll.clamp(min=-5.0, max=25.0)
            loss_nll = (best_nll * dec_mask).sum() / dec_mask.sum().clamp_min(1.0)

        loss = (
            loss_predict + loss_update
            + getattr(self.args, 'lambda_struct', 0.5) * loss_struct
            + getattr(self.args, 'lambda_temporal', 0.05) * loss_temporal
            + getattr(self.args, 'lambda_aux', 0.1) * loss_aux
            + getattr(self.args, 'lambda_nll', 1e-4) * loss_nll
        )
        self.last_loss_components = {
            'loss_predict': float(loss_predict.detach()),
            'loss_update': float(loss_update.detach()),
            'loss_struct': float(loss_struct.detach()),
            'loss_temporal': float(loss_temporal.detach()),
            'loss_aux': float(loss_aux.detach()),
            'loss_nll': float(loss_nll.detach()),
        }
        if outputs.counterfactual_base.numel():
            base_error = (
                outputs.counterfactual_base[:, :, 0] - outputs.target
            ).square().mean(dim=-1)
            expert_error = (
                outputs.counterfactual_expert[:, :, 0] - outputs.target
            ).square().mean(dim=-1)
            valid = dec_mask.bool()
            loss_expert = expert_error[valid].mean()
            benefit_target = torch.sigmoid(
                (
                    (base_error.detach() - expert_error.detach())
                    / base_error.detach().clamp_min(1e-6)
                    - getattr(self.args, 'gate_margin', 0.01)
                ) / getattr(self.args, 'gate_temperature', 0.05)
            )
            predicted_gate = outputs.policy_gate.squeeze(-1).clamp(1e-5, 1 - 1e-5)
            loss_gate = torch.nn.functional.binary_cross_entropy(
                predicted_gate[valid], benefit_target[valid]
            )
            loss_sparse = (
                (1.0 - benefit_target[valid]) * predicted_gate[valid]
            ).mean()
            stage = getattr(self.args, 'training_stage', 'joint')
            if stage == 'expert':
                loss = loss_expert + getattr(self.args, 'lambda_aux', 0.1) * loss_aux
            elif stage == 'gate':
                loss = (
                    loss_gate
                    + getattr(self.args, 'lambda_gate_sparse', 0.01) * loss_sparse
                )
            else:
                loss = (
                    loss
                    + getattr(self.args, 'lambda_expert', 0.2) * loss_expert
                    + getattr(self.args, 'lambda_gate', 0.1) * loss_gate
                    + getattr(self.args, 'lambda_gate_sparse', 0.01) * loss_sparse
                )
        return loss, loss_predict, loss_update, loss_struct

    def train_epoch(self, train_dataloader, current_epoch):
        """训练一个 epoch"""
        self.model.train()
        metrics = TrainingMetrics()

        tf_prob = self.get_tf_prob(current_epoch)

        with tqdm(total=len(train_dataloader), desc=f'Train {current_epoch}/{self.train_epoches}') as pbar:
            for batch_idx, graph_data in enumerate(train_dataloader):
                # 计算噪声协方差
                if getattr(self.args, 'windowed_training', False):
                    starts, rollout_steps = self.sample_training_starts(
                        graph_data, current_epoch, batch_idx
                    )
                else:
                    starts = [0]
                    rollout_steps = getattr(self.args, 'debug_time_steps', 0)
                    if rollout_steps <= 0:
                        rollout_steps = (
                            graph_data.y.shape[1]
                            - self.config['input_graph_window_len']
                            - self.config['output_graph_window_len']
                        )
                first_time_step = self.config['input_graph_window_len'] + starts[0]
                measureNoiseCov, measureNoiseCov_norm = self.compute_measurement_noise(
                    graph_data, first_time_step
                )
                graph_data = graph_data.to(self.device, non_blocking=True)
                measureNoiseCov = measureNoiseCov.to(self.device, non_blocking=True)
                measureNoiseCov_norm = measureNoiseCov_norm.to(self.device, non_blocking=True)

                # 初始化窗口
                graph_data_tmp = GraphDataProcessor.init_input_window(
                    graph_data, self.config, self.args, start_bias=starts[0]
                )
                graph_data_norm = graph_data_tmp.clone()  # 准备归一化版本
                P_predict = self.init_state_covariance(measureNoiseCov)

                # 训练循环
                batch_metrics = self.train_single_batch(
                    graph_data, graph_data_tmp, graph_data_norm, P_predict,
                    measureNoiseCov_norm, tf_prob, current_epoch,
                    starts[0], rollout_steps
                )
                metrics.add_batch_metrics(batch_metrics)

                # 更新进度条
                pbar.set_postfix(metrics.get_batch_mean())
                pbar.update(1)

        return metrics.get_epoch_mean()

    def train_single_batch(self, graph_data, graph_data_tmp, graph_data_norm,
                           P_predict, measureNoiseCov_norm, tf_prob, current_epoch,
                           start_bias=0, rollout_steps=1):
        """
        训练单个批次

        Args:
            graph_data: 原始图数据
            graph_data_tmp: 处理后的图数据
            P_predict: 初始协方差
            measureNoiseCov_norm: 归一化的测量噪声
            tf_prob: teacher forcing 概率
            current_epoch: 当前 epoch

        Returns:
            batch_metrics: 批次指标
        """
        time_total = graph_data.y.shape[1]
        encoder_hidden = None
        encoder_output_cache = None
        decoder_hidden_cache = None
        mixture_coeffs_cache = None
        gasf_recurrent_state = None

        batch_metrics = {
            'loss': [],
            'location_rmse_prediction': [],
            'velocity_rmse_prediction': [],
            'location_rmse_update': [],
            'velocity_rmse_update': [],
            'location_rmse_measurement': [],
            'f1': [],
            'grad_norm': [],
            'cov_min_eig': [],
            'cov_symmetry_error': [],
            'counterfactual_base_mse': [],
            'counterfactual_expert_mse': [],
            'counterfactual_benefit_rate': [],
            'loss_predict': [],
            'loss_update': [],
            'loss_struct': [],
            'loss_temporal': [],
            'loss_aux': [],
            'loss_nll': [],
        }

        # 滑动窗口训练
        P_predict_init = P_predict.clone()
        previous_adjacency = None
        previous_adjacency_mask = None
        first_time_step = self.config['input_graph_window_len'] + start_bias
        time_steps = range(first_time_step, first_time_step + rollout_steps)
        self.optimizer.zero_grad(set_to_none=True)
        last_covariance = None
        rollout_loss = P_predict.new_zeros(())
        for time_step in time_steps:
            _, measureNoiseCov_norm = self.compute_measurement_noise(
                graph_data, time_step
            )
            measureNoiseCov_norm = measureNoiseCov_norm.to(
                self.device, non_blocking=True
            )
            # 归一化数据
            graph_data_norm, centroids_x = minmax_scaler(
                graph_data_tmp, graph_data_norm, self.config, self.args.neural_net
            )

            # 前向传播
            outputs = self.model(
                graph_data_norm, tf_prob, P_predict, encoder_hidden,
                measureNoiseCov_norm,
                encoder_output_cache=encoder_output_cache,
                decoder_hidden_cache=decoder_hidden_cache,
                mixture_coeffs_cache=mixture_coeffs_cache,
                gasf_recurrent_state=gasf_recurrent_state,
            )
            encoder_output_cache = outputs.encoder_output_cache
            decoder_hidden_cache = outputs.decoder_hidden_cache
            mixture_coeffs_cache = outputs.mixture_coeffs_cache
            gasf_recurrent_state = outputs.gasf_recurrent_state

            # 计算损失
            loss, _, _ ,_ = self.compute_loss(
                outputs, graph_data, time_step,
                previous_adjacency, previous_adjacency_mask
            )
            previous_adjacency = outputs.adjacency.detach()
            previous_adjacency_mask = outputs.adjacency_mask.detach()
            batch_metrics['loss'].append(loss.item())
            for key, value in self.last_loss_components.items():
                batch_metrics[key].append(value)

            # 反向传播
            rollout_loss = rollout_loss + loss
            last_covariance = outputs.updated_covariance.detach()

            # 计算指标
            evaluation_metrics, x_update_renorm = self.compute_evaluation_metrics(
                graph_data_tmp, outputs, centroids_x
            )
            for key, value in evaluation_metrics.items():
                batch_metrics[key].append(value)

            # 更新协方差
            P_update = outputs[6]
            if current_epoch < self.P_change_epochs:
                P_predict = P_predict_init
            else:
                P_predict = P_update.detach()

            # 更新窗口
            bias = time_step - self.config['input_graph_window_len'] + 1
            graph_data_tmp = GraphDataProcessor.update_input_window(
                graph_data_tmp, graph_data, self.config, x_update_renorm, bias, self.args
            )

        (rollout_loss / rollout_steps).backward()
        clip_norm = getattr(self.args, 'grad_clip_norm', 10.0)
        if clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
        else:
            squared_grad_norm = P_predict.new_zeros(())
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    squared_grad_norm += parameter.grad.detach().norm(2).square()
            grad_norm = squared_grad_norm.sqrt()
        batch_metrics['grad_norm'].append(float(grad_norm))
        if last_covariance is not None:
            batch_metrics['cov_min_eig'].append(
                torch.linalg.eigvalsh(last_covariance).min().item()
            )
            batch_metrics['cov_symmetry_error'].append(
                (last_covariance - last_covariance.transpose(-1, -2)).abs().max().item()
            )
        self.optimizer.step()

        # 计算批次平均值
        return {key: np.mean(values) for key, values in batch_metrics.items()}

    def compute_evaluation_metrics(self, graph_data_tmp, outputs, centroids_x):
        """
        计算评估指标

        Args:
            graph_data_tmp: 处理后的图数据
            outputs: 模型输出
            centroids_x: 质心

        Returns:
            metrics: 评估指标
        """
        all_states_prediction = outputs[0]
        x_update = outputs[5]
        A_vec = outputs[8]

        # 计算误差指标
        (location_rmse_prediction, velocity_rmse_prediction,
         location_rmse_update, velocity_rmse_update,
         x_update_renorm, location_rmse_measurement) = caculate_evaluation(
            graph_data_tmp, all_states_prediction, x_update, centroids_x, self.config, 1
        )

        # 计算 F1 分数
        from utils.losses import edge_f1_score
        if A_vec.numel() == 0:
            f1_struct = 0.0
        else:
            target = graph_data_tmp.struct_feat[0].to(A_vec).reshape(-1)
            if target.numel() == A_vec.size(0):
                target = target.view(-1, 1).expand_as(A_vec)
                valid = outputs.adjacency_mask.bool()
                f1_struct, _, _ = edge_f1_score(
                    A_vec.detach()[valid], target[valid]
                )
            else:
                f1_struct = 0.0

        counterfactual_base_mse = 0.0
        counterfactual_expert_mse = 0.0
        counterfactual_benefit_rate = 0.0
        if outputs.counterfactual_base.numel():
            valid_nodes = outputs.real_mask.bool()
            base_node_error = (
                outputs.counterfactual_base[:, :, 0] - outputs.target
            ).square().mean(-1)
            expert_node_error = (
                outputs.counterfactual_expert[:, :, 0] - outputs.target
            ).square().mean(-1)
            counterfactual_base_mse = float(
                base_node_error[valid_nodes].mean().detach()
            )
            counterfactual_expert_mse = float(
                expert_node_error[valid_nodes].mean().detach()
            )
            counterfactual_benefit_rate = float(
                (expert_node_error[valid_nodes] < base_node_error[valid_nodes])
                .float().mean().detach()
            )

        return {
            'location_rmse_prediction': location_rmse_prediction,
            'velocity_rmse_prediction': velocity_rmse_prediction,
            'location_rmse_update': location_rmse_update,
            'velocity_rmse_update': velocity_rmse_update,
            'location_rmse_measurement': location_rmse_measurement,
            'f1': f1_struct,
            'counterfactual_base_mse': counterfactual_base_mse,
            'counterfactual_expert_mse': counterfactual_expert_mse,
            'counterfactual_benefit_rate': counterfactual_benefit_rate,
        }, x_update_renorm

    def validate(self, val_dataloader, current_epoch):
        """验证一个 epoch"""
        self.model.eval()
        metrics = TrainingMetrics()

        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'Val {current_epoch}/{self.train_epoches}') as pbar:
                for graph_data in val_dataloader:
                    # 计算噪声协方差
                    starts, rollout_steps = self.validation_starts(graph_data)
                    score_start = starts[len(starts) // 2]
                    burn_in_steps = min(
                        max(0, getattr(self.args, 'validation_burn_in_steps', 0)),
                        score_start,
                    )
                    start_bias = score_start - burn_in_steps
                    first_time_step = self.config['input_graph_window_len'] + start_bias
                    measureNoiseCov, measureNoiseCov_norm = self.compute_measurement_noise(
                        graph_data, first_time_step
                    )
                    graph_data = graph_data.to(self.device, non_blocking=True)
                    measureNoiseCov = measureNoiseCov.to(self.device, non_blocking=True)
                    measureNoiseCov_norm = measureNoiseCov_norm.to(self.device, non_blocking=True)

                    # 初始化窗口
                    graph_data_tmp = GraphDataProcessor.init_input_window(
                        graph_data, self.config, self.args, start_bias=start_bias
                    )
                    graph_data_norm = graph_data_tmp.clone()  # 准备归一化版本
                    P_predict = self.init_state_covariance(measureNoiseCov)

                    # 验证循环
                    batch_metrics = self.validate_single_batch(
                        graph_data, graph_data_tmp, graph_data_norm, P_predict,
                        measureNoiseCov_norm, start_bias,
                        rollout_steps + burn_in_steps, burn_in_steps
                    )
                    metrics.add_batch_metrics(batch_metrics)

                    # 更新进度条
                    pbar.set_postfix(metrics.get_batch_mean())
                    pbar.update(1)

        return metrics.get_epoch_mean()

    def validate_single_batch(self, graph_data, graph_data_tmp, graph_data_norm,
                              P_predict, measureNoiseCov_norm, start_bias=0,
                              rollout_steps=1, score_after=0):
        """
        验证单个批次

        Args:
            graph_data: 原始图数据
            graph_data_tmp: 处理后的图数据
            P_predict: 初始协方差
            measureNoiseCov_norm: 归一化的测量噪声

        Returns:
            batch_metrics: 批次指标
        """
        time_total = graph_data.y.shape[1]
        encoder_hidden = None
        encoder_output_cache = None
        decoder_hidden_cache = None
        mixture_coeffs_cache = None
        gasf_recurrent_state = None

        batch_metrics = {
            'loss': [],
            'location_rmse_prediction': [],
            'velocity_rmse_prediction': [],
            'location_rmse_update': [],
            'velocity_rmse_update': [],
            'location_rmse_measurement': [],
            'f1': [],
            'counterfactual_base_mse': [],
            'counterfactual_expert_mse': [],
            'counterfactual_benefit_rate': [],
            'loss_predict': [],
            'loss_update': [],
            'loss_struct': [],
            'loss_temporal': [],
            'loss_aux': [],
            'loss_nll': [],
        }

        # 滑动窗口验证
        previous_adjacency = None
        previous_adjacency_mask = None
        first_time_step = self.config['input_graph_window_len'] + start_bias
        time_steps = range(first_time_step, first_time_step + rollout_steps)
        for rollout_index, time_step in enumerate(time_steps):
            _, measureNoiseCov_norm = self.compute_measurement_noise(
                graph_data, time_step
            )
            measureNoiseCov_norm = measureNoiseCov_norm.to(
                self.device, non_blocking=True
            )
            # 归一化数据
            graph_data_norm, centroids_x = minmax_scaler(
                graph_data_tmp, graph_data_norm, self.config, self.args.neural_net
            )

            # 前向传播
            outputs = self.model(
                graph_data_norm, 0.0, P_predict, encoder_hidden,
                measureNoiseCov_norm,
                encoder_output_cache=encoder_output_cache,
                decoder_hidden_cache=decoder_hidden_cache,
                mixture_coeffs_cache=mixture_coeffs_cache,
                gasf_recurrent_state=gasf_recurrent_state,
            )
            encoder_output_cache = outputs.encoder_output_cache
            decoder_hidden_cache = outputs.decoder_hidden_cache
            mixture_coeffs_cache = outputs.mixture_coeffs_cache
            gasf_recurrent_state = outputs.gasf_recurrent_state

            # 计算损失
            should_score = rollout_index >= score_after
            if should_score:
                loss, _, _ ,_ = self.compute_loss(
                    outputs, graph_data, time_step,
                    previous_adjacency, previous_adjacency_mask
                )
                batch_metrics['loss'].append(loss.item())
                for key, value in self.last_loss_components.items():
                    batch_metrics[key].append(value)
            previous_adjacency = outputs.adjacency.detach()
            previous_adjacency_mask = outputs.adjacency_mask.detach()

            # 计算指标
            evaluation_metrics, x_update_renorm = self.compute_evaluation_metrics(
                graph_data_tmp, outputs, centroids_x
            )
            if should_score:
                for key, value in evaluation_metrics.items():
                    batch_metrics[key].append(value)

            # 更新协方差
            P_update = outputs[6]
            P_predict = P_update.detach()

            # 更新窗口
            bias = time_step - self.config['input_graph_window_len'] + 1
            graph_data_tmp = GraphDataProcessor.update_input_window(
                graph_data_tmp, graph_data, self.config, x_update_renorm, bias, self.args
            )

        # 计算批次平均值
        return {key: np.mean(values) for key, values in batch_metrics.items()}

    def _validation_score(self, val_metrics):
        """Select the metric that corresponds to the component being trained."""
        if (
            getattr(self.args, 'training_stage', 'joint') == 'expert'
            and getattr(self.args, 'counterfactual_training', False)
        ):
            return val_metrics['counterfactual_expert_mse']
        return (
            val_metrics['location_rmse_prediction']
            + val_metrics['velocity_rmse_prediction']
            + val_metrics['location_rmse_update']
            + val_metrics['velocity_rmse_update']
        )

    def save_checkpoint(self, current_epoch, val_metrics):
        """
        保存模型检查点

        Args:
            current_epoch: 当前 epoch
            val_metrics: 验证指标
        """
        # 计算评估分数
        evaluation = self._validation_score(val_metrics)

        # 保存最佳模型或定期保存
        is_best = evaluation < self.min_evaluation
        if is_best or current_epoch % 25 == 0:
            if is_best:
                self.min_evaluation = evaluation

            # 生成文件名
            now = time.localtime()
            nowt = time.strftime("%Y_%m_%d_%H_%M_", now)
            checkpoint_dir = Path(self.args.output_dir) / 'GNN_groupTargetTracking'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{nowt}epoch_{current_epoch}.pth"

            # 保存检查点
            torch.save({
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_metrics.get('loss', 0),
                'lr_schedule': self.lr_schedule.state_dict(),
                'epoch': current_epoch,
                'metrics': val_metrics,
                'model_variant': getattr(self.model, 'model_variant', 'sggt_v1'),
                'covariance_transition': getattr(self.model, 'covariance_transition', 'cv'),
                'use_SEAN': getattr(self.model.decoder, 'use_SEAN', False),
                'use_GASF': getattr(self.model.decoder, 'use_GASF', False),
                'optimizer': type(self.optimizer).__name__,
                'scheduler': type(self.lr_schedule).__name__,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gasf_filter_mode': getattr(self.args, 'gasf_filter_mode', 'abc'),
                'gasf_graph_mode': getattr(self.args, 'gasf_graph_mode', 'graph'),
                'gasf_fusion_mode': getattr(self.args, 'gasf_fusion_mode', 'concat'),
                'gasf_recurrent_filter': getattr(
                    self.args, 'gasf_recurrent_filter', True
                ),
                'gasf_recurrent_gate_initial_update': getattr(
                    self.args, 'gasf_recurrent_gate_initial_update', 0.1
                ),
                'gasf_aux_source': getattr(self.args, 'gasf_aux_source', 'gasf'),
                'counterfactual_training': getattr(
                    self.args, 'counterfactual_training', False
                ),
                'training_stage': getattr(self.args, 'training_stage', 'joint'),
            }, str(checkpoint_path))

            if is_best:
                self.best_model_path = checkpoint_path
                shutil.copyfile(
                    checkpoint_path,
                    checkpoint_dir / 'best.pth',
                )

    def train_and_evaluation(self, train_dataloader, test_dataloader):
        """完整的训练和评估循环"""
        process = psutil.Process(os.getpid())
        best_validation_score = float('inf')
        epochs_without_improvement = 0

        start_epoch = int(getattr(self.args, 'start_epoch', 0))
        if start_epoch == 0:
            print("Evaluating initialization as epoch -1...")
            initial_metrics = self.validate(test_dataloader, -1)
            self.log_metrics(-1, {}, initial_metrics)
            initial_score = self._validation_score(initial_metrics)
            self.logger.debug(
                f"validation_score:{initial_score:.6f},index:-1"
            )
            self.save_checkpoint(-1, initial_metrics)
            best_validation_score = initial_score

        for current_epoch in range(start_epoch, self.train_epoches):
            adapt_epochs = max(0, getattr(self.args, 'fusion_adapt_epochs', 0))
            should_unfreeze = (
                getattr(self.args, 'staged_joint_finetune', False)
                and adapt_epochs > 0
                and current_epoch >= adapt_epochs
                and not self.full_model_unfrozen
            )
            if should_unfreeze:
                for parameter in self.model.parameters():
                    parameter.requires_grad = True
                self.full_model_unfrozen = True
                self.logger.debug(
                    f"full_model_unfrozen:true,index:{current_epoch}"
                )
            self.active_learning_rates = [
                group['lr'] for group in self.optimizer.param_groups
            ]
            # 打印资源使用情况
            print(f"RAM: {process.memory_info().rss / 1024 ** 3:.2f} GB")
            print(f"GPU: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

            # 训练一个 epoch
            train_metrics = self.train_epoch(train_dataloader, current_epoch)

            # 验证一个 epoch
            val_metrics = self.validate(test_dataloader, current_epoch)

            # 更新学习率
            self.lr_schedule.step()

            # 记录日志
            self.log_metrics(current_epoch, train_metrics, val_metrics)

            # 保存检查点
            self.save_checkpoint(current_epoch, val_metrics)

            validation_score = self._validation_score(val_metrics)
            min_delta = getattr(self.args, 'early_stopping_min_delta', 1e-3)
            if validation_score < best_validation_score - min_delta:
                best_validation_score = validation_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            self.logger.debug(
                f"validation_score:{validation_score:.6f},index:{current_epoch}"
            )
            if epochs_without_improvement >= getattr(
                self.args, 'early_stopping_patience', 10
            ):
                print(
                    f"Early stopping at epoch {current_epoch}: "
                    f"best validation score {best_validation_score:.6f}"
                )
                break

        print("\n训练完成!")
        if self.best_model_path:
            print(f"最佳模型已保存至: {self.best_model_path}")

    def log_metrics(self, current_epoch, train_metrics, val_metrics):
        """记录训练和验证指标"""
        metrics_to_log = [
            ('train_loss', train_metrics, 'loss'),
            ('validation_loss', val_metrics, 'loss'),
            ('location_prediction_rmse', val_metrics, 'location_rmse_prediction'),
            ('velocity_prediction_rmse', val_metrics, 'velocity_rmse_prediction'),
            ('location_update_rmse', val_metrics, 'location_rmse_update'),
            ('velocity_update_rmse', val_metrics, 'velocity_rmse_update'),
            ('f1', val_metrics, 'f1'),
            ('grad_norm', train_metrics, 'grad_norm'),
            ('cov_min_eig', train_metrics, 'cov_min_eig'),
            ('cov_symmetry_error', train_metrics, 'cov_symmetry_error'),
            ('counterfactual_base_mse', val_metrics, 'counterfactual_base_mse'),
            ('counterfactual_expert_mse', val_metrics, 'counterfactual_expert_mse'),
            ('counterfactual_benefit_rate', val_metrics, 'counterfactual_benefit_rate'),
            ('loss_predict', val_metrics, 'loss_predict'),
            ('loss_update', val_metrics, 'loss_update'),
            ('loss_struct', val_metrics, 'loss_struct'),
            ('loss_temporal', val_metrics, 'loss_temporal'),
            ('loss_aux', val_metrics, 'loss_aux'),
            ('loss_nll', val_metrics, 'loss_nll'),
        ]

        for name, metrics, key in metrics_to_log:
            value = metrics.get(key, 0)
            if isinstance(value, (list, np.ndarray)):
                value = np.mean(value) if len(value) > 0 else 0
            self.logger.debug(f"{name}:{value:.6f},index:{current_epoch}")
        learning_rates = getattr(
            self,
            'active_learning_rates',
            [group['lr'] for group in self.optimizer.param_groups],
        )
        self.logger.debug(
            f"learning_rate:{learning_rates[0]:.10f},index:{current_epoch}"
        )
        for group_index, learning_rate in enumerate(learning_rates):
            self.logger.debug(
                f"learning_rate_group_{group_index}:{learning_rate:.10f},"
                f"index:{current_epoch}"
            )


# 向后兼容的函数
def train_and_evaluation(model, train_dataloader, test_dataloader,
                     optimizer, lr_schedule, logger, config, args):
    """
    向后兼容的训练函数

    Args:
        model: 模型
        train_dataloader: 训练数据加载器
        test_dataloader: 测试数据加载器
        optimizer: 优化器
        lr_schedule: 学习率调度器
        logger: 日志记录器
        config: 配置字典
        args: 训练参数
    """
    trainer = Trainer(model, optimizer, lr_schedule, logger, config, args)
    trainer.train_and_evaluation(train_dataloader, test_dataloader)
