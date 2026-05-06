import torch.nn as nn
import torch
import numpy as np

CONFIG = {
    # 1. 位置缩放 (用于去质心后的局部坐标)
    # 目的：保留宏观几何结构，范围 [-1, 1]。例如：500米
    'S_POS': 1500.0,

    # 2. 速度缩放 (用于速度 Vxy)
    # 目的：保留速度物理意义。例如：50 m/s
    'S_VEL': 300.0,

    # 3. 残差缩放 (用于残差 Rxy)
    # 目的：保证高灵敏度，范围 [-1, 1]。例如：10 米
    'S_RES': 110.0,

    # 4. 边特征缩放 (欧氏距离是正数，范围 [0, 1])
    # 连接距离阈值。例如：50米
    'S_EDGE_P_DIST': 450.0,  # 150
    # 最大速度差。例如：20 m/s
    'S_EDGE_V_DIST': 150.0,  # 20

    # 5. 模式 1:位置信息使用预测值  2: 位置信息使用量测值(模型初步搭建时期的模式)
    'MODE': 1,

    # 6. 量测维度
    'z_dimension': 2,

    # 7. 状态向量维度
    'x_dimension': 4,

    # 8. 输入的历史窗口长度
    'input_graph_window_len': 8,

    # 9. 输出的窗口长度
    'output_graph_window_len': 1,

    # 10. 归一化偏移量
    'centroids_offset': 0,

    # 11. 量测噪声标准差
    'measurement_noise_std': [5, 0.2 * np.pi / 180, 0.2 * np.pi / 180],
}

activations = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'lrelu': nn.LeakyReLU,
}


import argparse
import torch

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(description='GNN')
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoint/',
                            help='the output dir for model checkpoints')
        parser.add_argument('--checkpoint', default='../checkpoint/gnn/2026_03_18_19_45_.pth', type=str,
                            help='Path to model checkpoint')
        parser.add_argument('--data_dir', default='D:/Dataset/GroupTargetsDataset_dataRate_2_v8_pack/', type=str,
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='./log/training.log', type=str,
                            help='log dir for uer')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')
        parser.add_argument('--T', type=float, default=2,help='data rate')
        parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), #
                            help='cpu or gpu')
        parser.add_argument('--train_batch_size', default=4, type=int)
        parser.add_argument('--train_epochs', default=100, type=int,
                            help='Max training epoch')
        parser.add_argument('--test_epochs', default=10, type=int,
                            help='Max training epoch')
        parser.add_argument('--eval_batch_size', default=1, type=int)
        parser.add_argument('--optimizer_factor', default=0.8, type=int)
        parser.add_argument('--optimizer_patience', default=5, type=int)
        parser.add_argument('--lr', default=1e-3, type=float,
                            help='learning rate')


        # train args
        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')
        parser.add_argument('--u1-lim', type=float, default=10,
                            help='u1 control limit')
        parser.add_argument('--u2-lim', type=float, default=10,
                            help='u2 control limit')
        parser.add_argument('--ode-solver', type=str, default='rk4',
                            help='choose explicit solver [ef | mp | heun | rk3 | ssprk3 | rk4 | dopri5 | impl_adam]')
        parser.add_argument('--n-mixtures', type=int, default=1,
                            help='number of mixtures (default: 8)')
        parser.add_argument('--motion-model', type=str, default='group',
                            help='choice of motion model (default: 2Xnode)') #  group_residual / group_residual_without_MCU
        parser.add_argument('--n-ode-hidden', type=int, default=64,
                            help='n graph layers (default: 16)') # residula: 64
        parser.add_argument('--n-ode-layers', type=int, default=3,
                            help='n ode layers (default: 1)') # residula: 3

        parser.add_argument('--encoder_input_size', default=4, type=float)
        parser.add_argument('--encoder_hidden_size', default=64, type=float)
        parser.add_argument('--encoder_n_heads', default=3, type=float,)
        parser.add_argument('--encoder_n_layers', default=2, type=float,)
        parser.add_argument('--encoder_n_mixtures', default=4, type=int)
        parser.add_argument('--encoder_dropout', default=0.1, type=float)
        parser.add_argument('--encoder_gnn_layer', default="graphconv", type=str)
        parser.add_argument('--encoder_use_edge_features', default=True, type=bool)


        parser.add_argument('--decoder_motion_model', default='neuralode', type=str)
        parser.add_argument('--decoder_max_length', default=9, type=float) # 输入历史帧的长度加1
        parser.add_argument('--decoder_hidden_size', default=64, type=float,
                            help='(default: 64)')
        parser.add_argument('--decoder_n_heads', default=3, type=float)
        parser.add_argument('--decoder_n_layers', default=2, type=int)
        parser.add_argument('--decoder_alpha', default=0.2, type=float)  #
        parser.add_argument('--decoder_dropout', default=0.1, type=float)  #
        parser.add_argument('--decoder_gnn_layer', default="graphconv", type=str)  #
        parser.add_argument('--decoder_residual_length', default=[16], type=list)  #
        parser.add_argument('--decoder_z_dimension', default=2, type=float)  #
        parser.add_argument('--decoder_node_num', default=200, type=int) # 允许跟踪的最大目标数量
        parser.add_argument('--decoder_use_MCU', default=True, type=bool)
        parser.add_argument('--decoder_use_struct', default=True, type=bool)


        parser.add_argument('--neural_net', default='sggt_net', type=str)  # baseline/ sggt_net/  MCST
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()


