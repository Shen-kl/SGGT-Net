import numpy as np
from typing import List, Dict, Set
import torch
from config import CONFIG
from utils.common.caculate_measurement_noise_cov import calculate_measurement_noise_cov

def enu2aer(enu_x, enu_y, enu_z):
    azi = torch.atan2(enu_x,enu_y) * 180 / torch.pi
    elev = torch.atan2(enu_z, (enu_x**2+enu_y**2)**0.5)* 180 / torch.pi
    range = (enu_x**2+enu_y**2+enu_z**2)**0.5

    return azi, elev, range

def pure_PT(radar, all_trajectory_cnt: int, all_trajectory_history: list):
    """
    根据存在概率对目标进行剪枝 (对应 MATLAB 的 purePT 函数)。

    参数:
        radar (RadarSim): 雷达系统参数和航迹集合。
        all_trajectory_cnt (int): 已保存轨迹的总数。
        all_trajectory_history (list): 已保存的轨迹历史列表。

    返回:
        radar (RadarSim): 更新后的雷达对象。
        all_trajectory_cnt (int): 更新后的已保存轨迹总数。
        all_trajectory_history (list): 更新后的轨迹历史列表。
    """

    pt_delete_set = []  # 存放要删除的航迹
    lpt_to_keep = []  # 存放要保留的遗留航迹

    # --- 1. 遗留航迹 (LPT) 剪枝逻辑 ---

    # 遍历 LPT 集合并确定保留或删除
    for track in radar.legacy_PT_set:
        # MATLAB: size(X,2) 对应 Python: len(track.X)
        track_len = len(track.X)

        # 根据航迹长度选择阈值
        if track_len <= radar.confirm_trajectory_len:
            # 航迹未确认，使用 newPT 阈值
            threshold = radar.existence_threshold_newPT
        else:
            # 航迹已确认，使用 legacyPT 阈值
            threshold = radar.existence_threshold_legacyPT

        # 检查存在概率
        if track.r >= threshold:
            lpt_to_keep.append(track)
        else:
            pt_delete_set.append(track)

    # --- 2. 新航迹 (NPT) 剪枝逻辑 ---

    npt_to_promote = []  # 存放要提升为 LPT 的新航迹

    # 遍历 NPT 集合并确定保留或删除
    for track in radar.new_PT_set:
        # 新航迹总是使用 newPT 阈值
        threshold = radar.existence_threshold_newPT

        if track.r >= threshold:
            npt_to_promote.append(track)
        else:
            pt_delete_set.append(track)

    # --- 3. 处理要删除的航迹 (回收 ID, 保存历史) ---
    if pt_delete_set:
        for track in pt_delete_set:
            # 如果轨迹长度大于 10，则保存轨迹历史 (MATLAB代码中的硬编码阈值)
            if len(track.X) > 10:
                all_trajectory_cnt += 1
                all_trajectory_history.append(track.X)

            # 回收航迹 ID
            radar.track_index_set.append(track.track_index)

    # --- 4. 更新航迹集合 ---

    # 更新遗留航迹集: 保留的 LPT 加上提升的 NPT
    radar.legacy_PT_set = lpt_to_keep + npt_to_promote

    # 清空新航迹集
    radar.new_PT_set = []

    # 更新航迹数量
    radar.legacy_PT_num = len(radar.legacy_PT_set)
    radar.new_PT_num = 0

    # 计算转换更新后目标群的连通图
    # states = [track.X[-1] for track in radar.legacy_PT_set]
    # states = np.hstack(states)
    # survival_duration = [len(track.X) for track in radar.legacy_PT_set]
    # survival_duration = np.hstack(survival_duration)
    # clusters = radar.cluster_targets(states.T, survival_duration, duration_threshold=2 , max_radius=150)
    # radar.group_history.append(clusters) # 存储历史群


    if (radar.use_neural_network and  radar.neural_network['model_name'] != 'MCST'
            and radar.graph_builder.execute_cnt >= radar.track_len_for_nn):
        # 使用神经网络的更新器 重新更新航迹
        track_id_to_row: Dict[int, int] = {}
        row_idx = 0
        state_input = []
        measurement_input = []
        P_scaler_input = []
        for idx, track in enumerate(radar.legacy_PT_set):
            if len(track.X) > radar.track_len_for_nn:
                # 存入数据
                track_id_to_row[idx] = row_idx
                row_idx += 1

                # 准备数据
                X1 = torch.from_numpy(
                    np.stack(track.X[-radar.track_len_for_nn - 1:-1])).permute(2, 0, 1)
                X2 = torch.from_numpy(track.X_predict).unsqueeze(0).permute(0, 2, 1)
                state_input_single = torch.cat([X1, X2], dim=1)

                measurement_input_single = torch.from_numpy(
                    np.stack(track.associate_plot_track[-radar.track_len_for_nn - 1:])).permute(2, 0, 1)

                state_input.append(state_input_single)
                measurement_input.append(measurement_input_single)

                P_scaler_input.append(track.P_scaler)
        if len(state_input) != 0:
            state_input = torch.cat(state_input, dim=0)
            measurement_input = torch.cat(measurement_input, dim=0)
            P_scaler_input = torch.cat(P_scaler_input, dim=0)
            # 归一化
            state_input_norm, measurement_input_norm, centroids_x = radar.minmax_scaler_update_step(state_input, measurement_input)

            # 更新
            # 计算R矩阵
            azi, elev, distance = enu2aer(state_input[:,-1,0], state_input[:,-1,2],
                                          torch.zeros_like(state_input[:,-1,0]))
            mu, measureNoiseCov = calculate_measurement_noise_cov(distance, azi, elev, radar.distance_measurement_error, radar.azimuth_measurement_error * torch.pi / 180,
                                                                  radar.azimuth_measurement_error * torch.pi / 180.0)
            measureNoiseCov = torch.from_numpy(measureNoiseCov).permute(2, 0, 1)[:, : CONFIG['z_dimension'],
                              : CONFIG['z_dimension']].to(dtype = torch.float32)

            measureNoiseCov = measureNoiseCov * 100.0
            measureNoiseCov_norm = measureNoiseCov / (CONFIG['S_POS']) ** 2

            state_output, P_update = radar.neural_network['model'].update(state_input_norm[:,-1,:].unsqueeze(1), P_scaler_input, measurement_input_norm[:,-1,:].unsqueeze(1), measureNoiseCov_norm)

            # 反归一化
            X_update = radar.rescaler_update_step(state_output, centroids_x)

            # 遍历字典
            for key, value in track_id_to_row.items():
                radar.legacy_PT_set[key].X[-1] = X_update[value,0,:].reshape(-1,1)
                radar.legacy_PT_set[key].P_scaler = P_update[value, :, :]


    if radar.use_neural_network and radar.neural_network['model_name'] != 'MCST':
        # 计算图神经网络的输入
        radar.graph_builder.update(radar, duration_threshold=radar.track_len_for_adj)


    # 5. 打印确认航迹数量 (对应 MATLAB 的 disp 语句)
    confirmed_pt_num = sum(1 for track in radar.legacy_PT_set if len(track.X) >= radar.confirm_trajectory_len)
    print(f'legacy PT num: {confirmed_pt_num}')

    return radar, all_trajectory_cnt, all_trajectory_history
