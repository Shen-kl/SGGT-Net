import numpy as np
from scipy.stats.contingency import association

from utils.common.kalman_filter import KalmanFilter
import copy
from utils.common.coordinate_transformation import *
from utils.common.caculate_measurement_noise_cov import *
import torch

"""
函数功能： 量测评估
输出： 因子图中传递至a节点的信息beta 与 传递至 b节点的信息 xi
"""

def measurement_evaluation(radar):
    beta_message = np.zeros((radar.legacy_PT_num + 1, radar.plot_track_num + 1))
    xi_message = np.ones((radar.legacy_PT_num + 1, radar.plot_track_num + 1))
    posterior_parameters = {}
    if radar.use_neural_network:
        graph_data = radar.graph_builder.get()

    # 新息 数组 用于存储所有目标以及对应假设下的残差
    residual_set = []

    match radar.tracker_method:
        case 'BGM':
            # 新建后验变量 数量与目标数量一致
            posterior_parameters['gmWeight'] = []
            posterior_parameters['gmWeight_prior'] = []
            posterior_parameters['gmComponents'] = []
            posterior_parameters = [copy.deepcopy(posterior_parameters) for _ in range(radar.legacy_PT_num)]

            # 似然概率初始化
            LPT_likelihood = [] # legacy PT 的似然概率
            NPT_likelihood = [] # new PT 的似然概率
            if radar.legacy_PT_num > 0 :
                for tar_index in range(radar.legacy_PT_num):
                    LPT_likelihood.append(
                        np.zeros((radar.plot_track_num + 1, radar.legacy_PT_set[tar_index].numberOfGmComponents))) # 存放legacy PT 的似然值
                    residual_set.append(
                        np.zeros((radar.plot_track_num + 1, radar.legacy_PT_set[tar_index].numberOfGmComponents)))
                for tar_index in range(radar.new_PT_num):
                    NPT_likelihood.append(
                        np.zeros((1, radar.new_PT_set[tar_index].numberOfGmComponents))) # 存放new PT 的似然值


    if radar.legacy_PT_num > 0 and radar.plot_track_num > 0:
        # 计算消息
        alpha0 = np.zeros((radar.legacy_PT_num, 1))
        for tar_index, current_target in enumerate(radar.legacy_PT_set):
            logLikelihoodRatioTerms = np.log(radar.Pd) - np.log(radar.lambda_FA_PerUnitVolume)
            match radar.tracker_method:
                case 'BGM': # 更新检测概率
                    posterior_parameters[tar_index]['gmWeight'] = \
                        [radar.legacy_PT_set[tar_index].gmWeight * radar.Ps * radar.legacy_PT_set[tar_index].r for
                         _ in range(radar.plot_track_num + 1)]

                    posterior_parameters[tar_index]['gmComponents'] = []
                    for _ in range(radar.plot_track_num + 1):
                        # 创建 GM 分量的独立副本，以供后续更新
                        plot_components = [copy.deepcopy(comp) for comp in radar.legacy_PT_set[tar_index].gmComponents]
                        posterior_parameters[tar_index]['gmComponents'].append(plot_components)

                    for weight_index in range(radar.legacy_PT_set[tar_index].numberOfGmComponents):
                        # 量测维度的第一个维度代表漏检的情况
                        posterior_parameters[tar_index]['gmComponents'][0][weight_index].Pd['t'] += 1
                        posterior_parameters[tar_index]['gmComponents'][0][weight_index].Pd['Pd'] = (
                                posterior_parameters[tar_index]['gmComponents'][0][weight_index].Pd['s'] /
                                (posterior_parameters[tar_index]['gmComponents'][0][weight_index].Pd['t'] +
                                 posterior_parameters[tar_index]['gmComponents'][0][weight_index].Pd['s']))

                        posterior_parameters[tar_index]['gmComponents'][0][weight_index].associate_plot_track = (
                            copy.deepcopy(posterior_parameters[tar_index]['gmComponents'][0][weight_index].X[[0,2],0].reshape(-1,1)))
                        posterior_parameters[tar_index]['gmComponents'][0][weight_index].residual = 10
            for plot_track_index, plot_track in enumerate(radar.plot_track_set):
                match radar.tracker_method:
                    case 'BGM':
                        eps = 1e-6
                        match radar.associate_method:
                            case 'BP_R_RGTT':
                                C_index = radar.group_manager['IDX'][tar_index]  # 所在的簇
                                C_idx_0based = C_index - 1
                                for weight_index, gmComponent in enumerate(radar.legacy_PT_set[tar_index].gmComponents):
                                    Pd = gmComponent.Pd['Pd']

                                    ##############################  增加MCST稳定性
                                    # Pd = radar.Pd * 0.99
                                    ##############################
                                    logLikelihoodRatioTerms = np.log(Pd) - np.log(radar.lambda_FA_PerUnitVolume)

                                    num_group_partitions = len(radar.group_manager['group_partition'][C_idx_0based])
                                    X_update_group_partition = np.zeros((radar.x_dimension, num_group_partitions))
                                    P_update_group_partition = np.zeros((radar.x_dimension, radar.x_dimension,
                                                                         num_group_partitions))
                                    residual_group_partition = 0
                                    group_prob = radar.group_manager['group_prob'][C_idx_0based]
                                    # 遍历群分区
                                    for group_index in range(num_group_partitions):
                                        # 获取当前群分区的状态和协方差
                                        X_group = gmComponent.X_group_partition[:, group_index].reshape(-1, 1)
                                        P_group = gmComponent.P_group_partition[:, :, group_index]

                                        # 1. 预测量测 (Pred_Z)
                                        # pred_Z = H * X_group
                                        pred_Z = radar.tracker.H @ X_group

                                        # 2. 坐标转换和噪声协方差计算
                                        # 假设 pred_Z 是 [x, y]^T (2x1)
                                        X_pred = pred_Z[0, 0]
                                        Y_pred = pred_Z[1, 0]
                                        az_deg, elev, slantRange = enu2aer(X_pred, Y_pred, 0)

                                        azi_noise_rad = radar.azimuth_measurement_error * np.pi / 180  # 假设已是弧度

                                        # [ ~,measureNoiseCov] = CalMeasurementNoiseCov(...)
                                        _, measureNoiseCov_full = calculate_measurement_noise_cov(
                                            R=slantRange,
                                            A=az_deg,
                                            E=elev,
                                            rangeNoise=radar.distance_measurement_error,
                                            aziNoise=azi_noise_rad,
                                            eleNoise=0  # 假设俯仰角误差为 0
                                        )

                                        # 3. 协方差 S 和增益 K 的计算
                                        # S = H * P_group * H' + 10 * measureNoiseCov(1:z_dim, 1:z_dim)
                                        R_obs = 10 * measureNoiseCov_full[0:radar.z_dimension, 0:radar.z_dimension]
                                        S = radar.tracker.H @ P_group @ radar.tracker.H.T + R_obs

                                        # 检查 S 是否可逆且行列式不为零
                                        try:
                                            det_S = np.linalg.det(S)
                                            SInv = np.linalg.inv(S)
                                        except np.linalg.LinAlgError:
                                            print(
                                                "Warning: Singular matrix S detected. Skipping likelihood calculation.")
                                            gaussianLogLikelihood = -1e10  # 极小的似然
                                            SInv = np.eye(S.shape[0])  # 使用单位矩阵作为备用 SInv，但通常不用于实际计算

                                        # logGaussianNormalisingConstant = - (0.5 * z_dim) * log(2 * pi) - 0.5 * log(det(S))
                                        logGaussianNormalisingConstant = - (0.5 * radar.z_dimension) * np.log(
                                            2 * pi) - 0.5 * np.log(det_S if det_S > 1e-10 else 1e-10)

                                        # K = P * H' * SInv (注意：K 没有在此处使用，它将在 KalmanUpdate 中使用，但 MATLAB 代码中计算了它)
                                        K = P_group @ radar.tracker.H.T @ SInv

                                        # 4. 新息 (Innoviation) 和速度门约束
                                        innoviation = pred_Z - radar.plot_track_set[plot_track_index].X

                                        # 速度门约束：sum(abs(innoviation / radar.T) > radar.Vmax(1)) > 0
                                        velocity_innovation = np.abs(innoviation / radar.T)
                                        if np.any(velocity_innovation > radar.Vmax[0]):  # 假设 Vmax 是一个包含最大允许速度的数组/列表
                                            gaussianLogLikelihood = -1e9  # 超过速度门，似然极低 (近似为零)
                                        else:
                                            # gaussianLogLikelihood = logNormalisingConstant - 0.5 * innoviation' * SInv * innoviation
                                            gaussianLogLikelihood = logGaussianNormalisingConstant - 0.5 * innoviation.T @ SInv @ innoviation
                                            gaussianLogLikelihood = gaussianLogLikelihood[0, 0]  # 提取标量值

                                        # 5. 权重和边缘似然累加
                                        # logGMWeight = log(r) + log(gmWeight(weight_index))
                                        logGMWeight = np.log(current_target.r) + np.log(current_target.gmWeight[weight_index])

                                        # log(group_message)
                                        log_group_message = np.log(radar.group_manager['group_message'][C_idx_0based][group_index])

                                        # beta_message(tar_index + 1, plot_track_index + 1) += exp(logGMWeight + logLikelihoodRatioTerms + gaussianLogLikelihood + log(group_message))
                                        beta_message[tar_index + 1, plot_track_index + 1] += exp(
                                            logGMWeight + logLikelihoodRatioTerms + gaussianLogLikelihood + log_group_message)

                                        # LPT_likelihood{tar_index}(plot_track_index+1, weight_index) += exp(log(group_message) + logLikelihoodRatioTerms + gaussianLogLikelihood)
                                        LPT_likelihood[tar_index][plot_track_index + 1, weight_index] += exp(
                                            log_group_message + logLikelihoodRatioTerms + gaussianLogLikelihood)

                                        residual_group_partition += (
                                                (innoviation.T @ SInv @ innoviation) * group_prob[group_index])

                                        # 6. 状态更新 (使用 KalmenUpdate)
                                        # 假设 radar.tracker.KalmanUpdate 接受 (X, P, Z, R_obs)
                                        X_up, P_up = radar.tracker.KalmanUpdate(X_group, P_group, radar.plot_track_set[
                                            plot_track_index].X, R_obs)

                                        # 判断是否要用MCST预测
                                        if ( radar.use_neural_network and radar.neural_network['model_name'] == 'MCST'  and
                                                len(radar.legacy_PT_set[tar_index].X) > radar.track_len_for_nn):
                                            X_up = radar.perform_mcst_update(radar.legacy_PT_set[tar_index], radar.plot_track_set[plot_track_index].X)

                                        X_update_group_partition[:, group_index] = X_up.flatten()
                                        P_update_group_partition[:, :, group_index] = P_up

                                    # --- 群分区融合 (Group Partition Fusion) ---

                                    # X_update = sum(group_prob * X_update_group_partition, 2)
                                    group_prob = radar.group_manager['group_prob'][C_idx_0based]  # 1 x num_group_partitions
                                    X_update = np.sum(group_prob * X_update_group_partition, axis=1, keepdims=True)

                                    # P_update = sum_index (group_prob(index) * (P_up(:,:,index) + (X_up(:,index) - X_update) * (X_up(:,index) - X_update)'))
                                    P_update = np.zeros((radar.x_dimension, radar.x_dimension))
                                    for index in range(num_group_partitions):
                                        P_i = P_update_group_partition[:, :, index]
                                        X_i = X_update_group_partition[:, index].reshape(-1, 1)
                                        X_diff = X_i - X_update
                                        P_update += group_prob[index] * (P_i + X_diff @ X_diff.T)
                                    # 7. 更新后验参数结构

                                    # 更新 GM Component 的状态和协方差
                                    # posteriorParameters(tar_index).gmComponents{plot_track_index + 1, weight_index}.X = X_update
                                    # posteriorParameters(tar_index).gmComponents{plot_track_index + 1, weight_index}.P = P_update

                                    # 目标 i, 量测 j, 分量 k 的后验参数更新。
                                    # MATLAB: {plot_track_index + 1, weight_index}
                                    # Python: [plot_track_index, weight_index]
                                    posterior_parameters[tar_index]['gmComponents'][plot_track_index + 1][weight_index].X = X_update
                                    posterior_parameters[tar_index]['gmComponents'][plot_track_index + 1][weight_index].P = P_update
                                    posterior_parameters[tar_index]['gmComponents'][plot_track_index + 1][weight_index].associate_plot_track = (
                                        radar.plot_track_set[plot_track_index].X.reshape(-1,1))
                                    posterior_parameters[tar_index]['gmComponents'][plot_track_index + 1][
                                        weight_index].residual = residual_group_partition
                                    # 8. 更新未检测似然和 Pd 参数

                                    # LPT_likelihood{tar_index}(1, weight_index) = 1 - Pd (未检测似然)
                                    LPT_likelihood[tar_index][0, weight_index] = 1 - Pd

                                    # Pd.s (Hit count) + 1
                                    post_component = posterior_parameters[tar_index]['gmComponents'][plot_track_index + 1][weight_index]
                                    post_component.Pd['s'] += 1

                                    # Pd.Pd = s / (s + t)
                                    s = float(post_component.Pd['s'])
                                    t = float(post_component.Pd['t'])
                                    post_component.Pd['Pd'] = s / (s + t)

            match radar.tracker_method:
                case 'BGM':
                    alpha0[tar_index] = 1 - current_target.r
                    for gm_comp_index in range(current_target.numberOfGmComponents):
                        # 遍历高斯分量
                        Pd = current_target.gmComponents[gm_comp_index].Pd['Pd']
                        alpha0[tar_index] = alpha0[tar_index] + (1 - Pd) * current_target.r * current_target.gmWeight[gm_comp_index]

        beta_message[1:, 0] = alpha0.flatten()

        # 计算消息 xi
        for tar_index, current_target in enumerate(radar.new_PT_set):
            logLikelihoodRatioTerms = np.log(radar.lambda_NT) - np.log(radar.lambda_FA_PerUnitVolume)
            match radar.tracker_method:
                case 'BGM':
                    # 获取当前群分区的状态和协方差
                    X_group = current_target.X[-1].reshape(-1, 1)
                    P_group = current_target.P

                    # 1. 预测量测 (Pred_Z)
                    # pred_Z = H * X_group
                    pred_Z = radar.tracker.H @ X_group

                    # 2. 坐标转换和噪声协方差计算
                    # 假设 pred_Z 是 [x, y]^T (2x1)
                    X_pred = pred_Z[0, 0]
                    Y_pred = pred_Z[1, 0]
                    az_deg, elev, slantRange = enu2aer(X_pred, Y_pred, 0)

                    azi_noise_rad = radar.azimuth_measurement_error * np.pi / 180  # 假设已是弧度

                    # [ ~,measureNoiseCov] = CalMeasurementNoiseCov(...)
                    _, measureNoiseCov_full = calculate_measurement_noise_cov(
                        R=slantRange,
                        A=az_deg,
                        E=elev,
                        rangeNoise=radar.distance_measurement_error,
                        aziNoise=azi_noise_rad,
                        eleNoise=0  # 假设俯仰角误差为 0
                    )

                    # 3. 协方差 S 和增益 K 的计算
                    # S = H * P_group * H' + measureNoiseCov(1:z_dim, 1:z_dim)
                    R_obs = measureNoiseCov_full[0:radar.z_dimension, 0:radar.z_dimension]
                    S = radar.tracker.H @ P_group @ radar.tracker.H.T + R_obs

                    # 检查 S 是否可逆且行列式不为零
                    try:
                        det_S = np.linalg.det(S)
                        SInv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        print(
                            "Warning: Singular matrix S detected. Skipping likelihood calculation.")
                        gaussianLogLikelihood = -1e10  # 极小的似然
                        SInv = np.eye(S.shape[0])  # 使用单位矩阵作为备用 SInv，但通常不用于实际计算

                    logGaussianNormalisingConstant = - (0.5 * radar.z_dimension) * np.log(2 * pi) - 0.5 * np.log(
                        np.linalg.det(S))
                    match radar.tracker_model:
                        case 'VL':
                            innoviation = pred_Z - radar.plot_track_set[tar_index].X
                            gaussianLogLikelihood = logGaussianNormalisingConstant - 0.5 * innoviation.T @ SInv @ innoviation
                            xi_message[0, tar_index + 1] = xi_message[0, tar_index+ 1] + radar.Pb * exp(logLikelihoodRatioTerms
                                                                                                 + gaussianLogLikelihood)
                            NPT_likelihood[tar_index][0,0] = exp(logLikelihoodRatioTerms + gaussianLogLikelihood)

    # 输出
    beta_message[0, :] = 1
    association_info = {}
    association_info['beta_message'] = beta_message
    association_info['xi_message'] = xi_message
    association_info['LPT_likelihood'] = LPT_likelihood
    association_info['NPT_likelihood'] = NPT_likelihood

    return radar, association_info, posterior_parameters
