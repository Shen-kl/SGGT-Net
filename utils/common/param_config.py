import numpy as np
from utils.common.radar import Radar
from utils.common.generateLeaderFollowerTransMat import generateLeaderFollowerTransMat
from utils.common.kalman_filter import KalmanFilter
from model.sggt_net import SGGT_Net

from config import Args, CONFIG
from model.motion_models import *
from utils.common.graph_input_builder import GraphInputBuilder
import torch
from utils.mcst.MinMaxScaler import *
# ----------------------------------------------------------------------
# 外部未定义的变量 (需要用户提供实际值)
# ----------------------------------------------------------------------

def param_config(trajectory_info, measurement_noise_std):
    print('=======paramConfig==========')
    delta_T = trajectory_info['delta_T']  # 假设帧周期 (Frame period T)

    # 假设这些变量用于 set_resoultion 方法
    radar_range_resolution = trajectory_info['radar_range_resolution']
    radar_azi_resolution = trajectory_info['radar_azi_resolution']

    # ----------------------------------------------------------------------
    # 参数配置 (external_input 字典)
    # ----------------------------------------------------------------------
    external_input = {}

    ## 雷达站址位置
    external_input['location_enu'] = np.array([0, 0, 0])
    external_input['location_geography'] = np.array([0, 0, 0])

    ## 雷达采样速率和探测范围
    external_input['T'] = delta_T
    external_input['detection_distance_range'] = [0, 6e3]
    external_input['detection_azimuth_range'] = [-180, 180]
    external_input['detection_elevation_range'] = [-90, 90]

    ## 雷达跟踪器配置
    external_input['tracker_method'] = 'BGM'
    external_input['tracker_model'] = 'VL'
    external_input['associate_method'] = 'BP_R_RGTT'
    external_input['Chi_large'] = 16.0  # 卡尔分布门限
    external_input['Chi_small'] = 5.0
    external_input['PG'] = 1.0  # 门概率质量

    ## 起批参数配置
    external_input['trackProbabilityThreshold'] = 0.6
    external_input['Vmin'] = np.array([-100, -100, -100])  # 目标最小速度 (注意：Python 中通常使用列表或 NumPy 数组)
    external_input['Vmax'] = np.array([100, 100, 100])  # 目标最大速度
    external_input['Pd'] = 0.99  # 检测概率
    external_input['Pf'] = 1e-6  # 虚警概率

    observationSpaceVolume = 4e6
    external_input['lambda_FA'] = 10.0  # 杂波率 泊松分布参数
    external_input['lambda_FA_PerUnitVolume'] = external_input['lambda_FA'] / observationSpaceVolume
    external_input['lambda_NT'] = 10 * 1e-3  # 新目标空间率

    ## 分区参数
    external_input['distance_step'] = 300
    external_input['azi_step'] = 3

    ## 观测噪声
    external_input['distance_measurement_error'] = measurement_noise_std[0]
    external_input['azimuth_measurement_error'] = measurement_noise_std[1]
    external_input['elevation_measurement_error'] = 0.0

    ## 生存概率 出生概率
    external_input['Ps'] = 0.98
    external_input['Pb'] = 8e-2

    ## 存在概率 和 门限
    external_input['r_init'] = 0.2
    external_input['existence_threshold_legacyPT'] = 2e-2
    external_input['existence_threshold_newPT'] = 0.15

    ## 状态维度
    external_input['x_dimension'] = 4
    external_input['z_dimension'] = 2

    ## 高斯混合配置
    external_input['maximumNumberOfGmComponents'] = 12
    external_input['gmWeightThreshold'] = 1e-6

    ## 用于统计legacy PT 的最短航迹长度
    external_input['confirm_trajectory_len'] = 2

    ## 群结构相关参数
    external_input['group_extend_space'] = 20  # 单位：m
    external_input['group_sigma_r'] = 90  # 基于群的相关距离参数  90
    external_input['group_sigma_v'] = 6  # 基于群的相关速度参数
    external_input['max_group_partition'] = 50
    ## 启用神经网络
    external_input['use_neural_network'] = True
    external_input['track_len_for_nn'] = 8  # 用于神经网络的航迹最短长度
    external_input['track_len_for_adj'] = 2  # 基于建立节点之间关系的航迹最短长度


    # --- 实例化和设置 ---

    # 假设 Radar 类在环境中可用
    radar = Radar(external_input)

    radar_num = 1 # size(radar, 2)
    radar.set_resoultion(radar_range_resolution, radar_azi_resolution)

    prob_detection = 1.0  # 检测概率 (原始代码中未用于 Radar 初始化，单独赋值)
    # print(external_input)


    # 滤波器初始化
    match radar.tracker_method:
        case 'KF':
            pass
        case 'BGM':
            match radar.tracker_model:
                case 'VL':
                    match radar.associate_method:
                        case 'BP_R_RGTT':
                            F_CV = np.array([ [1, delta_T],[0, 1] ])
                            F = np.block([[F_CV, np.zeros((2, 2))],
                                          [np.zeros((2, 2)), F_CV]])

                            H = np.array([[1,0,0,0],[0,0,1,0]]) # 观测矩阵

                            R = np.array([[11, 28],[28, 80]]) # 观测噪声

                            # 模型超参数
                            hype_param = {}
                            hype_param['alpha'] = 0.01
                            hype_param['beta'] = 0.4
                            hype_param['gamma'] = 1e-4
                            hype_param['R1'] = 10 # 斥力参数
                            hype_param['R2'] = 8 # 斥力参数

                            F_follower, E_follower, F_leader, Q = (
                                generateLeaderFollowerTransMat(hype_param['alpha'], hype_param['beta'], hype_param['gamma'], 0, delta_T))
                            nvar = 1e1 # 过程噪声系数
                            Q = nvar * Q
                            hype_param['F_follower'] = F_follower
                            hype_param['E_follower'] = E_follower
                            hype_param['F_leader'] = F_leader

                            radar.tracker = KalmanFilter(F, H, Q, R, radar.tracker_model, hype_param)

                            if radar.use_neural_network:
                                # 如果启用了神经网络 加载以下参数
                                args = Args().get_parser()
                                args.T = delta_T
                                config = CONFIG

                                # Pure integrators
                                if args.motion_model == '1Xint':
                                    m_model = SingleIntegrator(solver=args.ode_solver, dt=args.T,
                                                               mixtures=args.n_mixtures)
                                elif args.motion_model == '2Xint':
                                    m_model = DoubleIntegrator(solver=args.ode_solver, dt=args.T,
                                                               mixtures=args.n_mixtures)
                                elif args.motion_model == '3Xint':
                                    m_model = TripleIntegrator(solver=args.ode_solver, dt=args.T,
                                                               mixtures=args.n_mixtures)

                                # Orientation-based
                                elif args.motion_model == 'singletrack':
                                    m_model = KinematicSingleTrack(solver=args.ode_solver, dt=args.T,
                                                                   mixtures=args.n_mixtures)
                                    if args.init_static:
                                        static_f_dim = 2
                                elif args.motion_model == 'unicycle':
                                    m_model = Unicycle(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)
                                elif args.motion_model == 'curvature':
                                    m_model = Curvature(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)
                                elif args.motion_model == 'curvilinear':
                                    m_model = CurviLinear(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures,
                                                          u1_lim=args.u1_lim)

                                # Neural ODEs
                                elif args.motion_model == 'neuralode':
                                    m_model = FirstOrderNeuralODE(solver=args.ode_solver, dt=args.T,
                                                                  mixtures=args.n_mixtures,
                                                                  static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                                  n_layers=args.n_ode_layers)
                                elif args.motion_model == '2Xnode':
                                    m_model = SecondOrderNeuralODE(solver=args.ode_solver, dt=args.T,
                                                                   mixtures=args.n_mixtures,
                                                                   static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                                   n_layers=args.n_ode_layers)
                                elif args.motion_model == 'group_residual_without_MCU':
                                    m_model = SecondOrderNeuralODE_groupTrack_without_MCU(solver=args.ode_solver, dt=args.T,
                                                                   mixtures=args.n_mixtures,
                                                                   static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                                   n_layers=args.n_ode_layers)
                                elif args.motion_model == 'group_residual_without_struct':
                                    m_model = SecondOrderNeuralODE_groupTrack_without_struct(solver=args.ode_solver, dt=args.T,
                                                                   mixtures=args.n_mixtures,
                                                                   static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                                   n_layers=args.n_ode_layers)
                                elif args.motion_model == 'group_residual_without_struct_MCU':
                                    m_model = SecondOrderNeuralODE_groupTrack_without_struct_MCU(solver=args.ode_solver, dt=args.T,
                                                                   mixtures=args.n_mixtures,
                                                                   static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                                   n_layers=args.n_ode_layers)
                                else:
                                    m_model = SecondOrderNeuralODE_groupTrack(solver=args.ode_solver, dt=args.T,
                                                                              mixtures=args.n_mixtures,
                                                                              static_f_dim=0,
                                                                              n_hidden=args.n_ode_hidden,
                                                                              n_layers=args.n_ode_layers)

                                # 神经网络模型
                                track_model = SGGT_Net(args.encoder_input_size, args.encoder_hidden_size,
                                                    args.encoder_n_heads,
                                                    args.encoder_n_layers, m_model.mixtures, args.encoder_dropout,
                                                    args.encoder_gnn_layer, args.encoder_use_edge_features,
                                                    m_model, args.decoder_max_length, args.decoder_hidden_size,
                                                    args.decoder_n_heads, args.decoder_n_layers, args.decoder_alpha,
                                                    args.decoder_dropout,  args.decoder_residual_length,
                                                             args.decoder_z_dimension, args.decoder_gnn_layer,
                                 args.decoder_use_MCU, args.decoder_use_struct, delta_T)
                                # 加载模型参数
                                track_model.load_state_dict(
                                    torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict'])

                                track_model.eval()

                                radar.neural_network['model_name'] = args.neural_net
                                radar.neural_network['model'] = track_model
                                radar.neural_network['config'] = config

                                # 保留的最大高斯混合分量的数量 设置为 1
                                # radar.maximumNumberOfGmComponents = 1

                                # 声明一个创建神经网络输入的实例
                                builder = GraphInputBuilder(window_size=radar.track_len_for_nn, max_targets=args.decoder_node_num, group_connect_threshold=150.0,
                                                residual_window_size=32)

                                radar.graph_builder = builder


    return radar