import numpy as np
from scipy.stats import poisson
from utils.common.coordinate_transformation import *
import torch
import matplotlib.pyplot as plt
import os
import scipy.io as io
from path import Path

# ----------------------------------------------------------------------
# 主脚本: 产生量测点 (Generate Measurements)
# ----------------------------------------------------------------------

def generateMeasurements(trajectory, born_dead_set, target_num, frame_max, delta_T, radar_azi_resolution,
                         radar_range_resolution, measurement_noise_std):
    """
    根据雷达参数和分辨率生成量测点并合并。

    参数:
        trajectory (list of np.array): 目标的真实轨迹。
        born_dead_set (list of list): 目标的出生和消亡帧数。
        target_num (int): 目标总数。
        frame_max (int): 总帧数。
        delta_T (float): 采样间隔。
        radar_azi_resolution (float): 雷达角度分辨率 (rad)。
        radar_range_resolution (float): 雷达径向距离分辨率 (m)。

    返回:
        measurement_plots_each_frame (list of np.array): 每一帧在 ENU 坐标系下合并后的量测 (2xN)。
        measurement_plots_ra_each_frame (list of np.array): 每一帧在极坐标系下合并后的量测 (2xN)。
        zone_x, zone_y (np.array): 轨迹范围。
    """
    print('=======generateMeasurements==========')

    # 初始化
    measurement_plots = {}
    measurement_plots_ra = {}
    zone_x = np.array([np.inf, -np.inf])
    zone_y = np.array([np.inf, -np.inf])
    # np.random.seed(1)
    # --- 1. 为每个目标生成带噪声的量测轨迹 ---
    for tar_index in range(target_num):
        single_trajectory = trajectory[tar_index]
        single_trajectory_length = single_trajectory.shape[1]

        # 获取真实位置 (x, y)
        x_true = single_trajectory[0, :]
        y_true = single_trajectory[2, :]
        z_true = np.zeros(single_trajectory_length)

        # 真实 ENU -> AER 坐标转换
        az_true, elev_true, slantRange_true = enu2aer(x_true, y_true, z_true)

        # 添加噪声 (径向距离 + 角度)
        slantRange_noisy = slantRange_true + measurement_noise_std[0] * np.random.randn(single_trajectory_length)
        az_noisy = az_true + measurement_noise_std[1] * np.random.randn(single_trajectory_length)

        single_target_measurement_plots_ra = np.vstack([slantRange_noisy, az_noisy])

        # AER -> ENU 转换
        xEast, yNorth, zUp = aer2enu(az_noisy, np.zeros(single_trajectory_length), slantRange_noisy)
        single_target_measurement_plots = np.vstack([xEast, yNorth])

        measurement_plots[tar_index] = single_target_measurement_plots  # ENU (2xN)
        measurement_plots_ra[tar_index] = single_target_measurement_plots_ra  # RA (2xN)

        # 更新观测空间大小
        zone_x[0] = min(zone_x[0], np.min(single_trajectory[0, :]))
        zone_x[1] = max(zone_x[1], np.max(single_trajectory[0, :]))
        zone_y[0] = min(zone_y[0], np.min(single_trajectory[2, :]))
        zone_y[1] = max(zone_y[1], np.max(single_trajectory[2, :]))

    # --- 2. 逐帧处理量测并进行分辨率合并 ---

    measurement_plots_each_frame = [None] * frame_max
    measurement_plots_ra_each_frame = [None] * frame_max

    for index in range(1, frame_max + 1):  # 从帧 1 到 frame_max

        # 初始化当前帧数据
        measurement_current = []
        measurement_current_ra = []
        azi_index_set = []
        target_indices_current = []  # 记录当前存活目标的原始索引 (0到target_num-1)

        # 收集当前时刻存活目标的量测
        for tar_index in range(target_num):
            born_frame, dead_frame = born_dead_set[tar_index]
            if born_frame <= index and dead_frame >= index:
                # 目标轨迹中的帧索引
                # MATLAB: index - born_dead_set{tar_index}(1) + 1
                loc = index - born_frame

                # 收集 ENU 和 RA 量测
                measurement_current.append(measurement_plots[tar_index][:, loc])
                measurement_current_ra.append(measurement_plots_ra[tar_index][:, loc])

                # 计算角度分区编号
                azi_index = np.floor(measurement_plots_ra[tar_index][1, loc] / radar_azi_resolution)
                azi_index_set.append(azi_index)
                target_indices_current.append(tar_index)  # 记录原始目标索引

        target_num_this_frame = len(measurement_current)

        if target_num_this_frame == 0:
            measurement_plots_each_frame[index - 1] = np.zeros((2, 0))
            measurement_plots_ra_each_frame[index - 1] = np.zeros((2, 0))
            continue

        measurement_current = np.column_stack(measurement_current)  # 2 x N_active
        measurement_current_ra = np.column_stack(measurement_current_ra)  # 2 x N_active
        azi_index_set = np.array(azi_index_set)

        # --- 2.1 角度分区 (波束内聚类) ---

        # 使用 unique 和 inverse 找出相同的分区
        azi_unique, azi_inverse = np.unique(azi_index_set, return_inverse=True)
        num_clusters = len(azi_unique)
        cluster = [[] for _ in range(num_clusters)]  # 存储的是在 `measurement_current` 中的索引 (0 到 N_active-1)

        # 聚类 (将目标索引 I 放入对应的 cluster)
        for i in range(target_num_this_frame):
            cluster[azi_inverse[i]].append(i)  # i 是当前存活目标中的索引

        # --- 2.2 径向距离合并 ---

        measurement_current_merge = []
        measurement_current_ra_merge = []

        for tar_index_set_in_cluster in cluster:  # tar_index_set_in_cluster 是在 measurement_current 中的索引列表

            if len(tar_index_set_in_cluster) == 1:
                # 1. 如果该波束内只有一个目标 (无需合并)
                idx = tar_index_set_in_cluster[0]
                measurement_current_merge.append(measurement_current[:, idx].reshape(-1, 1))
                measurement_current_ra_merge.append(measurement_current_ra[:, idx].reshape(-1, 1))
            else:
                # 2. 如果该波束内有多个目标 (需要径向距离合并)

                # 获取该波束内所有目标的 RA 量测
                ra_measurements = measurement_current_ra[:, tar_index_set_in_cluster]  # 2 x N_cluster
                distance_set = ra_measurements[0, :]  # 径向距离 (1 x N_cluster)

                # 按照径向距离排序
                sort_indices = np.argsort(distance_set)  # 升序排序的索引
                distance_set_sort = distance_set[sort_indices]
                ra_measurements_sort = ra_measurements[:, sort_indices]

                merge_ra = []
                merge_measurement_num = 0

                for k in range(len(distance_set_sort)):
                    current_ra = ra_measurements_sort[:, k]
                    current_dist = distance_set_sort[k]

                    if k == 0:
                        # 第一个点，作为新的合并量测的起点
                        merge_ra.append(current_ra.reshape(-1, 1))
                        merge_measurement_num = 1
                    else:
                        # 与前一个合并量测的径向距离进行比较
                        prev_merge_ra = merge_ra[-1]  # 2x1 向量
                        prev_merge_dist = prev_merge_ra[0, 0]

                        if abs(prev_merge_dist - current_dist) <= radar_range_resolution:
                            # 不可分辨 (合并)

                            # 合并的距离和角度 (加权平均)
                            new_merge_distance = (prev_merge_dist * merge_measurement_num + current_dist) / (
                                        merge_measurement_num + 1)
                            new_merge_azi = (prev_merge_ra[1, 0] * merge_measurement_num + current_ra[1]) / (
                                        merge_measurement_num + 1)

                            merge_ra[-1] = np.array([[new_merge_distance], [new_merge_azi]])
                            merge_measurement_num += 1
                        else:
                            # 可以分辨 (创建新的量测)
                            merge_ra.append(current_ra.reshape(-1, 1))
                            merge_measurement_num = 1  # 重新计数

                # 合并后的极坐标量测 (RA)
                merge_ra_mat = np.hstack(merge_ra) if merge_ra else np.zeros((2, 0))

                # 转换回 ENU 坐标
                xEast_merge, yNorth_merge, zUp_merge = aer2enu(merge_ra_mat[1, :], np.zeros(merge_ra_mat.shape[1]),
                                                               merge_ra_mat[0, :])
                merge_enu_mat = np.vstack([xEast_merge, yNorth_merge])

                measurement_current_ra_merge.append(merge_ra_mat)
                measurement_current_merge.append(merge_enu_mat)

        # 合并所有波束/簇的结果
        if measurement_current_merge:
            measurement_current_merge_mat = np.hstack(measurement_current_merge)
            measurement_current_ra_merge_mat = np.hstack(measurement_current_ra_merge)
        else:
            measurement_current_merge_mat = np.zeros((2, 0))
            measurement_current_ra_merge_mat = np.zeros((2, 0))

        # --- 2.3 增加杂波点 (Clutter) ---

        mu_c = 15  # 泊松分布均值  3/10/15
        # MATLAB poissrnd(mu_c, 1, 1) 对应 scipy.stats.poisson.rvs(mu=mu_c, size=1)
        samples = poisson.rvs(mu=mu_c, size=1)[0]

        # 杂波在观测空间内均匀分布 (假设是一个 3000x3000 的矩形区域)
        # 注意：MATLAB 代码中 zone_x/y 似乎没有被正确用于限制杂波，
        # 而是使用了 rand(2, samples) * 3e3。我们保持这个 3e3 的范围。
        clutter_enu = np.random.rand(2, samples) * 3000  # 2xSamples

        # 转换杂波到 RA 坐标
        az_clutter, elev_clutter, slantRange_clutter = enu2aer(clutter_enu[0, :], clutter_enu[1, :], np.zeros(samples))
        clutter_ra = np.vstack([slantRange_clutter, az_clutter])  # 2xSamples

        # 将合并后的量测与杂波合并 (MATLAB 代码中杂波部分被注释，这里也注释，但保留逻辑)
        # measurement_current_merge_mat = np.hstack([measurement_current_merge_mat, clutter_enu])
        # measurement_current_ra_merge_mat = np.hstack([measurement_current_ra_merge_mat, clutter_ra])

        # --- 2.4 排序与存储 ---

        # 按照 Y 坐标 (第二行) 排序，便于后续处理
        if measurement_current_merge_mat.shape[1] > 0:
            sort_index = np.argsort(measurement_current_merge_mat[1, :])
            measurement_plots_each_frame[index - 1] = measurement_current_merge_mat[:, sort_index]
            measurement_plots_ra_each_frame[index - 1] = measurement_current_ra_merge_mat[:, sort_index]
        else:
            measurement_plots_each_frame[index - 1] = measurement_current_merge_mat
            measurement_plots_ra_each_frame[index - 1] = measurement_current_ra_merge_mat



    # 加载matlab保存的点迹文件 统一两个平台的数据
    # matr = io.loadmat(Path('../generate_trajectory/scene02/trajectory_measurement.mat'))
    #
    # measurement_plots_each_frame = matr.get('measurement_plots_each_frame')
    # # 1. 先压成 1D，方便遍历
    # measurement_plots_each_frame = measurement_plots_each_frame.ravel()  # 现在长度 87
    #
    # # 2. 逐个剥开，按需要转 list
    # measurement_plots_each_frame = [item.copy() for item in measurement_plots_each_frame]
    #
    # measurement_plots_ra_each_frame = matr.get('measurement_plots_ra_each_frame')
    # # 1. 先压成 1D，方便遍历
    # measurement_plots_ra_each_frame = measurement_plots_ra_each_frame.ravel()  # 现在长度 87
    #
    # # 2. 逐个剥开，按需要转 list
    # measurement_plots_ra_each_frame = [item.copy() for item in measurement_plots_ra_each_frame]

    return measurement_plots_each_frame, measurement_plots_ra_each_frame, zone_x, zone_y

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # 加载航迹相关信息
    trajectory_path = '../../generate_trajectory/scene01/trajectory01.pt'
    trajectory_info = torch.load(trajectory_path, weights_only=False)
    measurement_noise_std = np.array([3.0, 0.01])
    measurement_plots_each_frame, measurement_plots_ra_each_frame, zone_x, zone_y = (
        generateMeasurements(trajectory_info['trajectory'], trajectory_info['born_dead_set'], trajectory_info['target_num'],
                         trajectory_info['frame_max'], trajectory_info['delta_T'], trajectory_info['radar_azi_resolution'],
                         trajectory_info['radar_range_resolution'],measurement_noise_std))

    plt.figure()
    for measurement_plots in measurement_plots_each_frame:
        plt.plot(measurement_plots[0,:], measurement_plots[1,:], 'rx')

    plt.show()