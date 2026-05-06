import numpy as np
from utils.common.coordinate_transformation import *
from utils.common.caculate_measurement_noise_cov import *
from utils.common.track import Track
from utils.minmax_scaler import *
import torch
import torch.nn as nn
from config import CONFIG
"""
执行所有legacy PT 的预测 以及 new PT 的创建
"""
def prediction_step(radar):
    if radar.legacy_PT_num > 0:
        if radar.use_neural_network and radar.graph_builder.execute_cnt >= 5:
            group_partition, group_state, group_prob, IDX = radar.compute_group_manager(1, 1)
        else:
            group_partition, group_state, group_prob, IDX = radar.compute_group_manager(1, 1)

    track_index = 0

    for track_index, current_legacy_PT in enumerate(radar.legacy_PT_set):
        radar.legacy_PT_set[track_index].r = radar.legacy_PT_set[track_index].r * radar.Ps # 存在概率
        current_legacy_PT.predict_flag = False # 初始化为False
        match radar.tracker_method:
            case 'BGM':
                for gmComponentIndex in range(radar.legacy_PT_set[track_index].numberOfGmComponents): # 分别对高斯分量进行预测
                    gm_comp = radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex]
                    s_prev = gm_comp.Pd['s']
                    t_prev = gm_comp.Pd['t']
                    k = gm_comp.Pd['k']
                    mu = s_prev / (s_prev + t_prev)
                    sigma_2 = k * (s_prev * t_prev) / ((s_prev + t_prev)**2 *(s_prev + t_prev + 1))
                    if sigma_2 == 0:
                        # 避免除零，如果方差为 0，保持不变或使用默认值
                        s_pred = s_prev
                        t_pred = t_prev
                    else:
                        temp_factor = mu * (1 - mu) / sigma_2 - 1
                        s_pred = mu * temp_factor
                        t_pred = (1 - mu) * temp_factor
                    radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex].Pd['s'] = s_pred
                    radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex].Pd['t'] = t_pred
                    radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex].Pd['Pd'] = s_pred / (s_pred + t_pred)
                    match radar.tracker_model:
                        case 'VL':
                            match radar.associate_method:
                                case 'BP_R_RGTT':
                                    if (radar.use_neural_network and (radar.graph_builder.execute_cnt >= 5 or
                                                                      radar.neural_network['model_name'] == "MCST")and
                                            len(current_legacy_PT.X) >= radar.track_len_for_nn):
                                        # 在启用神经网络 并且经历五帧后 使用网络执行预测 而不是物理模型
                                        continue
                                    # 遍历所有可能的群分区
                                    X_predict_group_partition_list = []
                                    P_predict_group_partition_list = []
                                    C_index = IDX[track_index] #所在的簇
                                    C_idx_0based = C_index - 1
                                    # 遍历所有可能的群分区假设
                                    current_group_partitions = group_partition[C_idx_0based]
                                    current_group_states = group_state[C_idx_0based]
                                    current_group_probs = group_prob[C_idx_0based]
                                    for index, partition_hypo in enumerate(
                                            current_group_partitions):  # index 是 0-based 假设索引
                                        # 找到自己所在的分区 (part_idx 是 1-based 分区索引)
                                        part_idx = 0
                                        current_partition_sets = partition_hypo
                                        # index_1 是 0-based
                                        for index_1, track_set in enumerate(current_partition_sets):
                                            # ismember(track_index, set)
                                            if track_index in track_set:
                                                part_idx = index_1  # 找到 0-based 分区索引
                                                break
                                        # 确定分区大小
                                        partition_size = len(current_partition_sets[part_idx])
                                        if partition_size == 1:
                                            # 群内只有一个目标 -> CV 外推 (KalmanPredict)
                                            X_pred_k, P_pred_k = radar.tracker.KalmanPredict(gm_comp.X, gm_comp.P, current_legacy_PT.Q_coe)
                                        else:
                                            # 群内有多个目标 -> VL 外推 (VLPredict)
                                            # 找到当前航迹在分区集中的索引 I (MATLAB: I = ismember(track_index, track_set))
                                            # 注意：MATLAB 代码中的 I 是一个布尔向量，这里我们只取 true 对应的索引
                                            # 由于分区内的目标数量可能较少，我们直接找到其在 track_set 中的位置（1-based）
                                            try:
                                                I = current_partition_sets[part_idx].index(
                                                    track_index)   # 0-based index within the partition
                                            except ValueError:
                                                # 不应该发生，因为前面已经检查过 ismember
                                                I = 0
                                                # 获取群状态信息
                                            group_state_hypo = current_group_states[
                                                index]  # group_state{C_index}{index}{part_idx}
                                            center = group_state_hypo[part_idx]['center']
                                            # repulsion(:,I)
                                            repulsion = group_state_hypo[part_idx]['repulsion'][:, I].reshape(-1,
                                                                                                        1)  # I-1 是 0-based
                                            F_follower = radar.tracker.hype_param['F_follower']
                                            E_follower = radar.tracker.hype_param['E_follower']
                                            Q_group = radar.tracker.Q
                                            X_pred_k, P_pred_k = radar.tracker.VLPredict(gm_comp.X, gm_comp.P, center,
                                                                                   repulsion, F_follower,
                                                                               E_follower, Q_group)
                                        X_predict_group_partition_list.append(X_pred_k)
                                        P_predict_group_partition_list.append(P_pred_k)
                                    # 3. 混合预测结果 (GM 状态和协方差)
                                    # 将列表转换为 NumPy 数组
                                    X_predict_group_partition = np.hstack(
                                        X_predict_group_partition_list)  # X_dim x num_hypotheses
                                    P_predict_group_partition = np.dstack(
                                        P_predict_group_partition_list)  # X_dim x X_dim x num_hypotheses
                                    # 计算混合状态 X_predict = sum(prob * X_k)
                                    group_prob_array = current_group_probs.reshape(1, -1)  # 1 x num_hypotheses
                                    X_predict = np.sum(group_prob_array * X_predict_group_partition, axis=1,
                                                       keepdims=True)  # X_dim x 1
                                    # 计算混合协方差 P_predict = sum(prob_k * (P_k + (X_k - X_pred)(X_k - X_pred)'))
                                    P_predict = np.zeros(
                                        (radar.x_dimension, radar.x_dimension))
                                    for k in range(len(current_group_probs)):
                                        prob_k = current_group_probs[k]
                                        P_k = P_predict_group_partition[:, :, k]
                                        X_k = X_predict_group_partition[:, k].reshape(-1, 1)
                                        # (X_k - X_predict) * (X_k - X_predict)'
                                        diff = X_k - X_predict
                                        covariance_term = P_k + diff @ diff.T
                                        P_predict += prob_k * covariance_term
                                    # 4. 更新航迹对象
                                    radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex].X = X_predict.copy()
                                    radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex].P = P_predict.copy()
                                    radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex].X_group_partition = X_predict_group_partition
                                    radar.legacy_PT_set[track_index].gmComponents[gmComponentIndex].P_group_partition = P_predict_group_partition

        if radar.use_neural_network  and (
                radar.neural_network['model_name'] == "MCST"  and len(
                current_legacy_PT.X) >= radar.track_len_for_nn):
            # 在启用神经网络 并且经历五帧后 使用网络执行预测 而不是物理模型
            radar.perform_mcst_predict(current_legacy_PT)
            current_legacy_PT.predict_flag = True  # 完成了预测
            continue
        if radar.use_neural_network  and ( radar.neural_network['model_name'] != "MCST" and radar.graph_builder.execute_cnt >= 5 and len(current_legacy_PT.X) >= radar.track_len_for_nn):
            # 在启用神经网络 并且经历五帧后 使用网络执行预测 而不是物理模型
            continue
        current_legacy_PT.predict_flag = True # 完成了预测
        radar.legacy_PT_set[track_index].X.append(X_predict)
        radar.legacy_PT_set[track_index].P = P_predict
        radar.legacy_PT_set[track_index].X_predict = X_predict
        radar.legacy_PT_set[track_index].P_predict = P_predict
        az, elev, slantRange = enu2aer(X_predict[0], X_predict[2], 0)
        if az > 180:
            az = az - 360
        radar.legacy_PT_set[track_index].Polar_X = np.array([slantRange, az, elev]).T
        # 判断是否超出探测范围
        in_dist = (radar.detection_distance_range[0] <= slantRange <=
                   radar.detection_distance_range[1])
        in_azi = (radar.detection_azimuth_range[0] <= az <=
                  radar.detection_azimuth_range[1])
        in_elev = (radar.detection_elevation_range[0] <= elev <=
                   radar.detection_elevation_range[1])
        if in_dist and in_azi and in_elev:
            radar.legacy_PT_set[track_index].track_property = radar.legacy_PT_set[track_index].track_property
        else:
            radar.legacy_PT_set[track_index].track_property = 2 # 超出范围 不再继续跟踪

    graph_input = None
    total_group_num = 0
    if  radar.use_neural_network and radar.neural_network['model_name'] != 'MCST'  and radar.graph_builder.execute_cnt >= radar.track_len_for_nn:
        # 在启用神经网络 并且经历五帧后 使用网络执行预测 而不是物理模型
        # 构造输入
        graph_input = radar.graph_builder.get()

        # 对数据进行归一化处理
        graph_data_norm, centroids_x = radar.minmax_scaler(graph_input, radar.neural_network['config'])

        # 输入网络
        all_states, P_scaler, total_group_num = radar.neural_network['model'].predict(graph_data_norm)

        # 反归一化
        radar.neural_network['config']['centroids_offset'] = centroids_x

        radar.rescaler(all_states, centroids_x, radar.neural_network['config'], radar.track_len_for_nn, graph_input, P_scaler)

    radar.total_group_num.append(total_group_num)

    # 使用量测初始化 初始化当前的 new PT
    # 遍历每个点迹
    for plot_track_index in range(radar.plot_track_num):
        current_plot_track = radar.plot_track_set[plot_track_index]

        # 1. 初始化状态向量 X (4x1)
        X = np.zeros((4, 1))
        # MATLAB: X(1:2:4,1) = radar.plot_track_set(plot_track_index).X(:,end);
        # 提取点迹的最新位置 x 和 y，并设置速度为 0。
        # 假设 plot_track.X 是 [x, vx, y, vy]^T
        X[0, 0] = current_plot_track.X[0, 0]  # x position
        X[2, 0] = current_plot_track.X[1, 0]  # y position

        # 2. 获取极坐标量测 [R, A (度)]
        Polar_X = current_plot_track.Polar_X

        # 3. 计算笛卡尔坐标系下的观测噪声协方差 R_full
        # 注意：MATLAB 代码中 Elevation Angle (E) 和 EleNoise 都设为 0
        azi_noise_rad = radar.azimuth_measurement_error * np.pi / 180  # 假设已是弧度

        # [ ~, R_full] = CalMeasurementNoiseCov(...)
        _, R_full = calculate_measurement_noise_cov(
            R=Polar_X[0, 0],
            A=Polar_X[1, 0],  # 传入度数
            E=0,
            rangeNoise=radar.distance_measurement_error,
            aziNoise=azi_noise_rad,
            eleNoise=0
        )

        # 4. 噪声协方差矩阵 R_full 乘 100
        # R = R * 100;
        R_full = R_full * 100

        # 提取 2x2 位置噪声协方差 R_xy
        R_xy = R_full[0:2, 0:2]

        # 5. 初始误差协方差矩阵 P 的计算 (MATLAB 代码中被覆盖)


        # 6. P 覆盖块：根据 MATLAB 注释，为了在目标初始化时增大误差协方差矩阵，
        # 以便下一帧目标能被包含在协方差椭圆内，直接使用一个大的对角矩阵。
        # P = diag( 50 * ones(radar.x_dimension, 1) ).^2;
        # P = (50 * I)^2 = 2500 * I
        P = np.diag(50 * np.ones(radar.x_dimension)) ** 2

        # 初始化 缩放后的P
        initial_uncertainty = 1e-2
        n_mixtures = 1
        P_scaler = torch.diag_embed(torch.ones(n_mixtures, 4)
                               ) * initial_uncertainty

        P_predict = init_P_from_R(torch.from_numpy(R_xy), n_mixtures, radar.T)
        P_scaler = P_predict * 1e2
        P_scaler[..., [0, 1], :] = P_scaler[..., [0, 1], :] / (CONFIG['S_POS'])
        P_scaler[..., :, [0, 1]] = P_scaler[..., :, [0, 1]] / (CONFIG['S_POS'])
        P_scaler[..., [2, 3], :] = P_scaler[..., [2, 3], :] / (CONFIG['S_VEL'])
        P_scaler[..., :, [2, 3]] = P_scaler[..., :, [2, 3]] / (CONFIG['S_VEL'])

        # 7. 航迹初始化和管理
        if not radar.track_index_set:
            print("Warning: No more available track indices.")
            continue

        current_track_index = radar.track_index_set.pop(0)  # 弹出航迹号

        new_track = Track(X=X, P=P, R=R_xy, doppler_vel=0, track_ID=current_track_index, zone_index=0, track_quality=0,
                        tracker_method=radar.tracker_method, tracker_model=radar.tracker_model,
                          hype_param=radar.tracker.hype_param, associate_method=radar.associate_method, r_init=radar.r_init,
                          P_scaler=P_scaler)
        radar.new_PT_set.append(new_track)
        if radar.use_neural_network and radar.neural_network['model_name'] == "MCST":
            radar.track_mcst_param_init()

    # 8. 更新新航迹数量
    radar.new_PT_num = radar.plot_track_num

    return radar


def init_P_from_R(measurement_noise_matrix, n_mixtures, delta_T, device=None):

    R = measurement_noise_matrix

    P_matlab = torch.zeros(n_mixtures, 4, 4, device=device)

    P_matlab[...,0,0] = R[...,0,0]
    P_matlab[...,0,1] = R[...,0,0] / delta_T
    P_matlab[...,0,2] = R[...,0,1]
    P_matlab[...,0,3] = R[...,0,1] / delta_T

    P_matlab[...,1,0] = R[...,0,0] / delta_T
    P_matlab[...,1,1] = 2 * R[...,0,0] / delta_T**2
    P_matlab[...,1,2] = R[...,0,1] / delta_T
    P_matlab[...,1,3] = 2 * R[...,0,1] / delta_T**2

    P_matlab[...,2,0] = R[...,0,1]
    P_matlab[...,2,1] = R[...,0,1] / delta_T
    P_matlab[...,2,2] = R[...,1,1]
    P_matlab[...,2,3] = R[...,1,1] / delta_T

    P_matlab[...,3,0] = R[...,0,1] / delta_T
    P_matlab[...,3,1] = 2 * R[...,0,1] / delta_T**2
    P_matlab[...,3,2] = R[...,1,1] / delta_T
    P_matlab[...,3,3] = 2 * R[...,1,1] / delta_T**2

    T = torch.tensor([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ], dtype=P_matlab.dtype, device=device)

    return T @ P_matlab @ T.T