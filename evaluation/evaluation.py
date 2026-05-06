import torch
from utils.common.generateMeasurements import generateMeasurements
from utils.common.param_config import param_config
from utils.common.simulate_measurements import simulate_measurements
from utils.common.prediction_step import prediction_step
from utils.common.measurement_evaluation import measurement_evaluation
from utils.common.loopy_belief_propagation import loopy_belief_propagation
from utils.common.compute_posterior_spatial_distributions import compute_posterior_spatial_distributions
from utils.common.cal_error import cal_error
from utils.common.pure_PT import pure_PT
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import time
from utils.common.generateMeasurements import enu2aer, aer2enu
'''
在群目标运动场景下，使用模型验证。 包含常规的VL模型预测
'''

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # 加载航迹相关信息
    trajectory_path = '../generate_trajectory/scene02/trajectory01.pt'
    trajectory_info = torch.load(trajectory_path, weights_only=False)

    # 构造量测
    # 噪声标准差 [径向距离(m), 角度(degree)]
    measurement_noise_std = np.array([3.0, 0.01])

    measurement_plots_each_frame, measurement_plots_ra_each_frame, zone_x, zone_y = (
        generateMeasurements(trajectory_info['trajectory'], trajectory_info['born_dead_set'], trajectory_info['target_num'],
                         trajectory_info['frame_max'], trajectory_info['delta_T'], trajectory_info['radar_azi_resolution'],
                         trajectory_info['radar_range_resolution'], measurement_noise_std))

    # 保存数据
    # measurement_plots = []
    # measurement_plots_ra = []
    # zone_x = np.array([np.inf, -np.inf])
    # zone_y = np.array([np.inf, -np.inf])
    # np.random.seed(0)
    #
    # target_num = len(trajectory_info['trajectory'])
    # trajectory = trajectory_info['trajectory']
    # # --- 1. 为每个目标生成带噪声的量测轨迹 ---
    # for tar_index in range(target_num):
    #     single_trajectory = trajectory[tar_index]
    #     single_trajectory_length = single_trajectory.shape[1]
    #
    #     # 获取真实位置 (x, y)
    #     x_true = single_trajectory[0, :]
    #     y_true = single_trajectory[2, :]
    #     z_true = np.zeros(single_trajectory_length)
    #
    #     # 真实 ENU -> AER 坐标转换
    #     az_true, elev_true, slantRange_true = enu2aer(x_true, y_true, z_true)
    #
    #     # 添加噪声 (径向距离 + 角度)
    #     slantRange_noisy = slantRange_true + measurement_noise_std[0] * np.random.randn(single_trajectory_length)
    #     az_noisy = az_true + measurement_noise_std[1] * np.random.randn(single_trajectory_length)
    #
    #     single_target_measurement_plots_ra = np.vstack([slantRange_noisy, az_noisy])
    #
    #     # AER -> ENU 转换
    #     xEast, yNorth, zUp = aer2enu(az_noisy, np.zeros(single_trajectory_length), slantRange_noisy)
    #     single_target_measurement_plots = np.vstack([xEast, yNorth])
    #
    #     measurement_plots.append(single_target_measurement_plots)  # ENU (2xN)
    #     measurement_plots_ra.append(single_target_measurement_plots_ra)  # RA (2xN)
    #
    #
    # trajectory_gt = np.stack(trajectory_info['trajectory']).transpose(0, 2, 1)
    # trajectory_measurement = np.stack(measurement_plots).transpose(0, 2, 1)
    #
    # for idx in range(trajectory_gt.shape[0]):
    #     torch.save({'trajectory': trajectory_gt[idx,:],
    #                 'measurement': trajectory_measurement[idx,:]}, f'./dataset_for_GNN_paper/trajectory{int(idx):02d}.pt')


    # 算法配置
    radar = param_config(trajectory_info, measurement_noise_std)

    all_trajectory_cnt = 0
    all_trajectory_history = []
    ospa_error_set = []
    target_set_num = []

    elapsed_ms_set = []
    for frame_index in range(trajectory_info['frame_max']):
        print('frame_index={}'.format(frame_index))
        # 产生当前时刻的量测
        radar = simulate_measurements(radar, measurement_plots_each_frame[frame_index])

        # 开始计时
        start_time = time.perf_counter()

        # 预测步骤
        radar = prediction_step(radar)

        if radar.plot_track_num > 0:
            # 量测评估与迭代数据关联
            radar, association_info, posterior_parameters = measurement_evaluation(radar)
            message_aq, message_bv = loopy_belief_propagation(radar, association_info)

            # 更新后验
            radar = compute_posterior_spatial_distributions(radar, association_info, message_aq, message_bv, posterior_parameters)
        else:
            # 更新存在概率
            for track in radar.legacy_PT_set:
                track.r = track.r * (1 - radar.Pd) / (1 - radar.Pd * track.r)

        # 根据存在概率对目标剪枝
        radar, all_trajectory_cnt, all_trajectory_history = pure_PT(radar, all_trajectory_cnt, all_trajectory_history)

        # 结束计时
        end_time = time.perf_counter()

        # 计算耗时（转换为毫秒）
        elapsed_ms = (end_time - start_time) * 1000

        elapsed_ms_set.append(elapsed_ms)

        # 计算误差
        cal_error(radar, trajectory_info, frame_index, ospa_error_set, target_set_num)

    print(f'平均每一帧耗时{np.mean(elapsed_ms_set)} ms')

    # 绘制结果图
    target_set_num = np.hstack(target_set_num)
    ospa_error_set = np.hstack(ospa_error_set)

    ospa_error = np.mean(ospa_error_set**2)**0.5
    print(f'OSPA error: {ospa_error} m')

    # 绘制目标的势
    plt.figure()
    plt.title('Target Num')
    plt.plot(target_set_num)
    plt.plot(trajectory_info['labels_set_num'], linewidth=2.5)

    # 绘制目标的OSPA
    plt.figure()
    plt.title('OSPA')
    plt.plot(ospa_error_set)

    # 绘制目标的轨迹 与 真实轨迹
    plt.figure()
    for trajectory in trajectory_info['trajectory']:
        plt.plot(trajectory[0,:],trajectory[2,:])
    for trajectory in all_trajectory_history:
        trajectory_stack = np.hstack(trajectory)
        plt.plot(trajectory_stack[0,:],trajectory_stack[2,:])
        plt.plot(trajectory_stack[0, 1], trajectory_stack[2, 1],'o')
        plt.plot(trajectory_stack[0, -1], trajectory_stack[2, -1],'>')

    # 绘制 群的数量
    plt.figure()
    plt.plot(trajectory_info['group_num_truth'],'k-')
    plt.plot(radar.total_group_num)
    plt.ylim([0,10])
    plt.show()




