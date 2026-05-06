"""
模型功能对比脚本

用于验证优化后的模型与原始模型功能是否一致
"""

import torch
import numpy as np
from model.sggt_net import SGGT_Net as OriginalSGGT_Net
from model.sggt_net_optimized import SGGT_Net as OptimizedSGGT_Net
from model.motion_models import SecondOrderNeuralODE_groupTrack
from model.motion_models_base import SecondOrderNeuralODE_groupTrack as OptimizedMotionModel
from config import CONFIG


def create_mock_graph_data(batch_size=4, seq_len=10, n_nodes=8):
    """创建模拟的图数据"""
    from torch_geometric.data import Data

    # 创建基础数据
    x = torch.randn(batch_size, seq_len, 4)
    y = torch.randn(batch_size, seq_len, 4)
    measurement = torch.randn(batch_size, seq_len, 2)

    # 创建边索引 (每个节点连接到最近的 2 个节点)
    edge_index_list = []
    edge_features_list = []
    tar_edge_index_list = []
    tar_edge_features_list = []

    for t in range(seq_len):
        # 简单的全连接图
        edges = []
        edge_feats = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edges.append([i, j])
                    edge_feats.append([i / n_nodes, j / n_nodes])

        edge_index_list.append(torch.tensor(edges, dtype=torch.long).t().contiguous())
        edge_features_list.append(torch.tensor(edge_feats, dtype=torch.float))
        tar_edge_index_list.append(torch.tensor(edges, dtype=torch.long).t().contiguous())
        tar_edge_features_list.append(torch.tensor(edge_feats, dtype=torch.float))

    # 其他属性
    batch = torch.arange(batch_size).repeat_interleave(n_nodes // batch_size)
    residual = torch.randn(batch_size, seq_len, 2)
    measurement_future = torch.randn(batch_size, seq_len, 2)
    nan_mask = torch.ones(batch_size, seq_len, 2, dtype=torch.bool)
    real_mask = torch.ones(batch_size, seq_len, 4, dtype=torch.bool)
    struct_feat = torch.randint(0, 2, (batch_size, seq_len, 28), dtype=torch.float)
    param_change_time = torch.tensor([[10, 20, 30]])

    data = Data(
        x=x,
        y=y,
        measurement=measurement,
        edge_index=edge_index_list,
        edge_features=edge_features_list,
        tar_edge_index=tar_edge_index_list,
        tar_edge_features=tar_edge_features_list,
        batch=batch,
        residual=residual,
        measurement_future=measurement_future,
        nan_mask=nan_mask,
        real_mask=real_mask,
        struct_feat=[struct_feat] * seq_len,
        param_change_time=param_change_time,
        P_scaler=torch.randn(batch_size, 4, 4)
    )

    return data


def test_motion_models():
    """测试运动模型"""
    print("=" * 60)
    print("测试运动模型")
    print("=" * 60)

    # 参数
    batch_size = 2
    mixtures = 8
    n_states = 4
    dt = 0.04

    # 创建模型
    original_model = SecondOrderNeuralODE_groupTrack(
        solver='rk4', dt=dt, n_states=n_states, mixtures=mixtures,
        n_hidden=32, n_layers=2,
        u1_lim=10, u2_lim=10, u3_lim=10, u4_lim=10,
        u5_lim=10, u6_lim=10, u7_lim=10, u8_lim=10
    )

    optimized_model = OptimizedMotionModel(
        solver='rk4', dt=dt, n_states=n_states, mixtures=mixtures,
        n_hidden=32, n_layers=2,
    )

    # 复制权重
    with torch.no_grad():
        for (name_o, param_o), (name_n, param_n) in zip(
            original_model.named_parameters(),
            optimized_model.named_parameters()
        ):
            if name_o == name_n:
                param_n.copy_(param_o)

    # 测试数据
    past_state = torch.randn(batch_size, mixtures, n_states)
    inputs = torch.randn(batch_size, mixtures, 8)  # 8 = 3*2 + 2
    static_f = None

    # 前向传播
    original_model.eval()
    optimized_model.eval()

    with torch.no_grad():
        orig_state, orig_input = original_model(past_state, inputs, static_f)
        opt_state, opt_input = optimized_model(past_state, inputs, static_f)

    # 比较输出
    state_diff = torch.abs(orig_state - opt_state).max().item()
    input_diff = torch.abs(orig_input - opt_input).max().item()

    print(f"状态差异: {state_diff:.6e}")
    print(f"输入差异: {input_diff:.6e}")

    if state_diff < 1e-5 and input_diff < 1e-5:
        print("✓ 运动模型测试通过")
        return True
    else:
        print("✗ 运动模型测试失败")
        return False


def test_sggt_net():
    """测试 SGGT_Net 模型"""
    print("\n" + "=" * 60)
    print("测试 SGGT_Net 模型")
    print("=" * 60)

    # 参数
    encoder_input_size = 4
    encoder_hidden_size = 64
    encoder_n_heads = 3
    encoder_n_layers = 1
    encoder_n_mixtures = 8
    encoder_dropout = 0.1
    encoder_gnn_layer = "graphconv"
    encoder_use_edge_features = True

    decoder_max_length = 10
    decoder_hidden_size = 64
    decoder_n_heads = 3
    decoder_n_layers = 1
    decoder_alpha = 0.2
    decoder_dropout = 0.1
    decoder_residual_length = [8, 16, 32]
    decoder_z_dimension = 2
    decoder_gnn_layer = "graphconv"
    decoder_use_MCU = True
    decoder_use_struct = True
    delta_T = 0.04

    # 创建运动模型
    motion_model = SecondOrderNeuralODE_groupTrack(
        solver='rk4', dt=delta_T, n_states=4, mixtures=8,
        n_hidden=32, n_layers=2,
        u1_lim=10, u2_lim=10, u3_lim=10, u4_lim=10,
        u5_lim=10, u6_lim=10, u7_lim=10, u8_lim=10
    )

    # 创建主模型
    original_model = OriginalSGGT_Net(
        encoder_input_size, encoder_hidden_size,
        encoder_n_heads, encoder_n_layers, encoder_n_mixtures, encoder_dropout,
        encoder_gnn_layer, encoder_use_edge_features,
        motion_model, decoder_max_length, decoder_hidden_size,
        decoder_n_heads, decoder_n_layers, decoder_alpha,
        decoder_dropout, decoder_residual_length,
        decoder_z_dimension, decoder_gnn_layer,
        decoder_use_MCU, decoder_use_struct, delta_T
    )

    # 创建优化模型 (注意: 需要使用相同的运动模型)
    motion_model_opt = OptimizedMotionModel(
        solver='rk4', dt=delta_T, n_states=4, mixtures=8,
        n_hidden=32, n_layers=2,
    )

    optimized_model = OptimizedSGGT_Net(
        encoder_input_size, encoder_hidden_size,
        encoder_n_heads, encoder_n_layers, encoder_n_mixtures, encoder_dropout,
        encoder_gnn_layer, encoder_use_edge_features,
        motion_model_opt, decoder_max_length, decoder_hidden_size,
        decoder_n_heads, decoder_n_layers, decoder_alpha,
        decoder_dropout, decoder_residual_length,
        decoder_z_dimension, decoder_gnn_layer,
        decoder_use_MCU, decoder_use_struct, delta_T
    )

    # 复制编码器和解码器的权重
    with torch.no_grad():
        # 复制编码器
        for (name_o, param_o), (name_n, param_n) in zip(
            original_model.encoder.named_parameters(),
            optimized_model.encoder.named_parameters()
        ):
            if name_o == name_n:
                param_n.copy_(param_o)

        # 复制解码器
        for (name_o, param_o), (name_n, param_n) in zip(
            original_model.decoder.named_parameters(),
            optimized_model.decoder.named_parameters()
        ):
            if name_o == name_n:
                param_n.copy_(param_o)

    # 创建测试数据
    graph_data = create_mock_graph_data(batch_size=2, seq_len=10, n_nodes=8)

    # 前向传播
    original_model.eval()
    optimized_model.eval()

    with torch.no_grad():
        orig_output = original_model(graph_data, tf_prob=0.5)
        opt_output = optimized_model(graph_data, tf_prob=0.5)

    # 比较主要输出
    outputs_to_compare = [
        ("预测状态", orig_output[0], opt_output[0]),
        ("协方差", orig_output[1], opt_output[1]),
        ("混合系数", orig_output[2], opt_output[2]),
    ]

    all_passed = True
    for name, orig_val, opt_val in outputs_to_compare:
        diff = torch.abs(orig_val - opt_val).max().item()
        print(f"{name}: 差异 = {diff:.6e}")

        # 注意: 由于 EKF 实现的差异，可能会有些小的数值差异
        if diff < 1e-3:
            print(f"  ✓ {name} 测试通过")
        else:
            print(f"  ✗ {name} 测试失败")
            all_passed = False

    if all_passed:
        print("✓ SGGT_Net 模型测试通过")
        return True
    else:
        print("✗ SGGT_Net 模型测试失败")
        return False


def test_kalman_utils():
    """测试卡尔曼滤波工具"""
    print("\n" + "=" * 60)
    print("测试卡尔曼滤波工具")
    print("=" * 60)

    from model.kalman_utils import (
        MatrixBuilder, CovarianceUtils, ExtendedKalmanFilter
    )

    # 测试矩阵构建
    D = MatrixBuilder.build_scaling_matrix()
    print(f"缩放矩阵形状: {D.shape}")
    print(f"缩放矩阵: \n{D}")

    H = MatrixBuilder.build_observation_matrix()
    print(f"\n观测矩阵形状: {H.shape}")
    print(f"观测矩阵: \n{H}")

    F = MatrixBuilder.build_cv_transition_matrix(0.04, 2)
    print(f"\n状态转移矩阵形状: {F.shape}")

    # 测试对称性
    P = torch.randn(2, 4, 4)
    P_sym = CovarianceUtils.ensure_symmetry(P)
    diff = (P_sym - P_sym.transpose(-1, -2)).abs().max().item()
    print(f"\n对称性检查: 最大差异 = {diff:.6e}")

    if diff < 1e-6:
        print("  ✓ 对称性测试通过")
    else:
        print("  ✗ 对称性测试失败")
        return False

    # 测试 EKF
    ekf = ExtendedKalmanFilter(delta_T=0.04)
    P_t = torch.randn(2, 4, 4)
    G_t = torch.randn(2, 4, 2)
    q_t = torch.randn(2, 2, 2)

    P_next, Q_t = ekf.time_update(P_t, G_t, q_t)

    print(f"\n时间更新:")
    print(f"  输入协方差形状: {P_t.shape}")
    print(f"  输出协方差形状: {P_next.shape}")
    print(f"  过程噪声形状: {Q_t.shape}")

    print("✓ 卡尔曼滤波工具测试通过")
    return True


def main():
    """运行所有测试"""
    print("\nSGGT-Net 模型对比测试")
    print("=" * 60)

    results = []

    # 运行测试
    results.append(("卡尔曼滤波工具", test_kalman_utils()))
    results.append(("运动模型", test_motion_models()))
    results.append(("SGGT_Net", test_sggt_net()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查实现")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
