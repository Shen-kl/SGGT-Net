import numpy as np
import matplotlib.pyplot as plt
import os
from utils.common.generateLeaderFollowerTransMat import generateLeaderFollowerTransMat
import torch
# 假设 generateLeaderFollowerTransMat 函数已在 leader_follower_model.py 中定义和导入
# from leader_follower_model import generateLeaderFollowerTransMat

# ----------------------------------------------------------------------
# 辅助函数: compute_group_move (对应 MATLAB 内部函数)
# ----------------------------------------------------------------------

def compute_group_move(group, trajectory, group_param, born_dead_set, current_time, group_center_vel):
    """
    计算群组的运动学和状态外推。

    参数:
        group: 群组成员列表的列表 (e.g., [[1, 2], [3, 4]])
        trajectory: 存储每个目标轨迹的列表 (初始状态为 4x1 向量，后续为 4xN 矩阵)
        F, F_follower, E_follower: 转移矩阵和输入矩阵
        born_dead_set: 目标出生/消亡的帧数
        current_time: 当前帧数
        group_center_vel: 群组质心的速度向量列表 (2x1)
        R1, R2: 斥力模型参数

    返回:
        trajectory: 更新后的目标轨迹列表
        group_num: 当前时刻存活的群组数量
    """
    V_MAX = 50 # 允许的最大速度
    V_MIN = 0 # 允许的最小速度

    group_num = 0

    # 遍历每一个群
    for group_index, group_members in enumerate(group):
        group_state = []

        # 收集当前群组中存活成员的最新状态
        for group_mem_index in group_members:
            # 目标索引在 Python 中是从 0 开始，所以需要减 1
            idx = group_mem_index - 1

            born_frame, dead_frame = born_dead_set[idx]
            if born_frame <= current_time and dead_frame >= current_time:
                # 假设 trajectory{idx} 是 (4, N) 矩阵
                # MATLAB trajectory{index}(:,end) 对应 Python trajectory[idx][:, -1]
                group_state.append(trajectory[idx][:, -1])

        if not group_state:
            # 如果群组中没有存活的目标，跳过
            continue

        group_state = np.column_stack(group_state)  # 转换为 4xN 矩阵
        center = np.mean(group_state, axis=1, keepdims=True)  # 群质心 (4x1)

        # 处理群组只包含一个存活目标的情况 (简化为匀加速/匀速运动 + 群心速度)
        if group_state.shape[1] == 1:
            # group_members 列表中的第一个成员
            idx = group_members[0] - 1
            born_frame, dead_frame = born_dead_set[idx]

            # MATLAB: F * trajectory{group{group_index}(1)}(:,end) + [0 0 ;1 0; 0 0 ;0 1] * group_center_vel{group_index}
            # F_CA 结构体: [1 0; 0 1; 0 0; 0 0] @ vel + [0 0 ;1 0; 0 0 ;0 1] @ vel

            # 速度输入矩阵 [0 0; 1 0; 0 0; 0 1]
            vel_input_mat = np.array([
                [0, 0],
                [1, 0],
                [0, 0],
                [0, 1]
            ])
            F = group_param[group_index]['F']
            next_state = F @ trajectory[idx][:, -1].reshape(-1, 1)

            # 提取期望的速度
            V_des_x = next_state[1]
            V_des_y = next_state[3]
            V_des_vec = np.array([V_des_x, V_des_y])

            # 计算模长
            V_mag = np.linalg.norm(V_des_vec)

            # 实施速度限制
            if V_mag > V_MAX:
                V_new_vec = V_MAX * (V_des_vec / V_mag)
            elif  V_mag < V_MIN:
                V_new_vec = V_MIN * (V_des_vec / V_mag)
            else:
                V_new_vec = V_des_vec

            #  更新下一个状态
            next_state[1] = V_new_vec[0]
            next_state[3] = V_new_vec[1]

            trajectory[idx] = np.hstack((trajectory[idx], next_state))
        elif group_state.shape[1] == 0:
            continue
        else:
            # 计算斥力模型 (Repulsion Model)
            r_list = []  # 存储每个目标的斥力 (2x1)

            for i in range(group_state.shape[1]):
                current_pos = group_state[[0, 2], i].reshape(2, 1)
                other_pos = np.delete(group_state[[0, 2], :], i, axis=1)

                # 计算目标 i 与其他目标的距离向量 d_xy
                d_xy = current_pos - other_pos  # 2 x (N-1)

                # 计算距离范数 d_norm
                d_norm = np.linalg.norm(d_xy, axis=0)  # 1 x (N-1)

                # 计算斥力幅值 f = R1 / (d_norm + R2)
                f = R1 / (d_norm + R2)  # 1 x (N-1)

                # 斥力方向向量，并缩放
                # f./d_norm .* d_xy
                scaled_d_xy = (f / d_norm) * d_xy  # 2 x (N-1)

                # 总斥力 r_i = sum(scaled_d_xy, axis=1)
                r_i = np.sum(scaled_d_xy, axis=1, keepdims=True)  # 2 x 1
                r_list.append(r_i)

            # 判断类型
            match group_param[group_index]['type']:
                case 'VL':
                    # 虚拟领导者模型
                    F_follower = group_param[group_index]['F_follower']
                    E_follower = group_param[group_index]['E_follower']
                    F_leader = group_param[group_index]['F_leader']

                    # 状态外推 (使用 Leader-Follower 和 Repulsion)
                    # index 计数器对应 r_list 的索引
                    r_index = 0
                    for group_mem_index in group_members:
                        idx = group_mem_index - 1
                        born_frame, dead_frame = born_dead_set[idx]

                        if born_frame <= current_time and dead_frame >= current_time:
                            current_state = trajectory[idx][:, -1].reshape(-1, 1)  # 4x1
                            current_r = r_list[r_index]  # 2x1

                            # MATLAB: F_follower * current_state + E_follower * (center_state) + F_follower * repulsion_input

                            # 领导者/群心输入 (4x1)
                            # MATLAB: (center + [0 0 ;1 0; 0 0 ;0 1] * group_center_vel{group_index})
                            # 这里的 center 是 4x1，它本身就包含了位置和速度信息
                            center_pos_vel = center + np.array(
                                [[0], [0], [0], [0]])  # 4x1 (简化，因为 group_center_vel 是 2x1)

                            # 速度输入矩阵 [0 0; 1 0; 0 0; 0 1] 仅适用于 F * xk

                            # 修正: MATLAB E_follower * center 的 center 应该是 4x1
                            # 这里的 center 是群质心 [x_c, v_x_c, y_c, v_y_c]'

                            # 修正: MATLAB E_follower 输入是 4x1 的状态向量
                            # E_follower * (center + [0 0 ;1 0; 0 0 ;0 1] * group_center_vel{group_index})

                            # 领导者输入 (4x1)
                            leader_input = center.copy()
                            # 附加群中心速度（仅在速度项上加）
                            # leader_input[[1, 3], 0] += group_center_vel[group_index]

                            # 斥力输入项 (4x1)
                            # MATLAB: [0 ;r(1,index);0;r(2,index)]
                            repulsion_input = np.array([
                                [0],
                                [current_r[0, 0]],
                                [0],
                                [current_r[1, 0]]
                            ])

                            # 外推公式
                            next_state = (F_follower @ current_state) + \
                                         (E_follower @ F_leader @ leader_input) + \
                                         (F_follower @ repulsion_input)

                            # 提取期望的速度
                            V_des_x = next_state[1]
                            V_des_y = next_state[3]
                            V_des_vec = np.array([V_des_x, V_des_y])

                            # 计算模长
                            V_mag = np.linalg.norm(V_des_vec)

                            # 实施速度限制
                            if V_mag > V_MAX:
                                V_new_vec = V_MAX * (V_des_vec / V_mag)
                            elif V_mag < V_MIN:
                                V_new_vec = V_MIN * (V_des_vec / V_mag)
                            else:
                                V_new_vec = V_des_vec

                            #  更新下一个状态
                            next_state[1] = V_new_vec[0]
                            next_state[3] = V_new_vec[1]

                            trajectory[idx] = np.hstack((trajectory[idx], next_state))
                            r_index += 1

                case 'LF':
                    # 领导者跟随者模型
                    F_follower = group_param[group_index]['F_follower']
                    E_follower = group_param[group_index]['E_follower']
                    F_leader = group_param[group_index]['F_leader']
                    leader_ID = group_param[group_index]['leader_ID']


                    # 状态外推 (使用 Leader-Follower 和 Repulsion)
                    # index 计数器对应 r_list 的索引
                    r_index = 0
                    # 先预测leader
                    # 外推公式
                    current_state = trajectory[leader_ID-1][:, -1].reshape(-1, 1)  # 4x1
                    leader_next_state = (F_leader @ current_state)
                    trajectory[leader_ID-1] = np.hstack((trajectory[leader_ID-1], leader_next_state))


                    for group_mem_index in group_members:
                        idx = group_mem_index - 1
                        born_frame, dead_frame = born_dead_set[idx]

                        if group_mem_index != leader_ID and  born_frame <= current_time and dead_frame >= current_time:
                            current_state = trajectory[idx][:, -1].reshape(-1, 1)  # 4x1
                            current_r = r_list[r_index]  # 2x1

                            # MATLAB: F_follower * current_state + E_follower * (center_state) + F_follower * repulsion_input

                            # 领导者/群心输入 (4x1)
                            # MATLAB: (center + [0 0 ;1 0; 0 0 ;0 1] * group_center_vel{group_index})
                            # 这里的 center 是 4x1，它本身就包含了位置和速度信息
                            center_pos_vel = center + np.array(
                                [[0], [0], [0], [0]])  # 4x1 (简化，因为 group_center_vel 是 2x1)

                            # 速度输入矩阵 [0 0; 1 0; 0 0; 0 1] 仅适用于 F * xk

                            # 修正: MATLAB E_follower * center 的 center 应该是 4x1
                            # 这里的 center 是群质心 [x_c, v_x_c, y_c, v_y_c]'

                            # 修正: MATLAB E_follower 输入是 4x1 的状态向量
                            # E_follower * (center + [0 0 ;1 0; 0 0 ;0 1] * group_center_vel{group_index})

                            # 领导者输入 (4x1)
                            leader_input = center.copy()
                            # 附加群中心速度（仅在速度项上加）
                            # leader_input[[1, 3], 0] += group_center_vel[group_index]

                            # 斥力输入项 (4x1)
                            # MATLAB: [0 ;r(1,index);0;r(2,index)]
                            repulsion_input = np.array([
                                [0],
                                [current_r[0, 0]],
                                [0],
                                [current_r[1, 0]]
                            ])

                            # 外推公式
                            next_state = (F_follower @ current_state) + \
                                         (E_follower @ leader_next_state) + \
                                         (F_follower @ repulsion_input)

                            # 提取期望的速度
                            V_des_x = next_state[1]
                            V_des_y = next_state[3]
                            V_des_vec = np.array([V_des_x, V_des_y])

                            # 计算模长
                            V_mag = np.linalg.norm(V_des_vec)

                            # 实施速度限制
                            if V_mag > V_MAX:
                                V_new_vec = V_MAX * (V_des_vec / V_mag)
                            elif V_mag < V_MIN:
                                V_new_vec = V_MIN * (V_des_vec / V_mag)
                            else:
                                V_new_vec = V_des_vec

                            #  更新下一个状态
                            next_state[1] = V_new_vec[0]
                            next_state[3] = V_new_vec[1]

                            trajectory[idx] = np.hstack((trajectory[idx], next_state))
                            r_index += 1
                case 'indep':
                    # 独立模型
                    F = group_param[group_index]['F']

                    # 状态外推 (使用 Leader-Follower 和 Repulsion)
                    # index 计数器对应 r_list 的索引
                    r_index = 0

                    for group_mem_index in group_members:
                        idx = group_mem_index - 1
                        born_frame, dead_frame = born_dead_set[idx]

                        if born_frame <= current_time and dead_frame >= current_time:
                            current_state = trajectory[idx][:, -1].reshape(-1, 1)  # 4x1

                            # 速度输入矩阵 [0 0; 1 0; 0 0; 0 1] 仅适用于 F * xk

                            # 修正: MATLAB E_follower * center 的 center 应该是 4x1
                            # 这里的 center 是群质心 [x_c, v_x_c, y_c, v_y_c]'

                            # 外推公式
                            next_state = (F @ current_state)

                            # 提取期望的速度
                            V_des_x = next_state[1]
                            V_des_y = next_state[3]
                            V_des_vec = np.array([V_des_x, V_des_y])

                            # 计算模长
                            V_mag = np.linalg.norm(V_des_vec)

                            # 实施速度限制
                            if V_mag > V_MAX:
                                V_new_vec = V_MAX * (V_des_vec / V_mag)
                            elif V_mag < V_MIN:
                                V_new_vec = V_MIN * (V_des_vec / V_mag)
                            else:
                                V_new_vec = V_des_vec

                            #  更新下一个状态
                            next_state[1] = V_new_vec[0]
                            next_state[3] = V_new_vec[1]

                            trajectory[idx] = np.hstack((trajectory[idx], next_state))
                            r_index += 1
        group_num += 1

    return trajectory, group_num


# ----------------------------------------------------------------------
# 主脚本: 轨迹生成 (Generate Tracjectories)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print('=======generateTrajectory==========')
    trajectory_id = 'trajectory1'
    delta_T = 2
    radar_range_resolution = 1e-5
    radar_azi_resolution = 1e-5
    group_num_truth = [3]  # 初始群组数

    frame_max = int(200 / delta_T)  # 总帧计数

    # --- 1. 初始化目标状态和时间参数 ---

    # 初始状态 (4x1 向量，[x, vx, y, vy]')
    trajectory_init = [
        np.array([-200, 8, 1680, -2]).reshape(-1, 1),  # T1
        np.array([-200, 8, 1700, -2]).reshape(-1, 1),  # T2
        np.array([-200, 8, 1420, 2]).reshape(-1, 1),  # T3
        np.array([-200, 8, 1400, 2]).reshape(-1, 1),  # T4
        np.array([1500, 28, 2020, 0]).reshape(-1, 1),  # T5
        np.array([1500, 28, 2005, 0]).reshape(-1, 1),  # T6
        np.array([1500, 28, 1970, 0]).reshape(-1, 1),  # T7
        np.array([1500, 28, 1955, 0]).reshape(-1, 1),  # T8
        np.array([0, 10, -950, 2]).reshape(-1, 1),  # T9
        np.array([270, 18, -1045, 0]).reshape(-1, 1),  # T10
        np.array([320, 16, -980, 0]).reshape(-1, 1),  # T11
        np.array([200, 13, -845, 0]).reshape(-1, 1),  # T12
        np.array([800, 0, -1505, 15]).reshape(-1, 1)  # T13
    ]
    target_num = len(trajectory_init)
    trajectory = [t.copy() for t in trajectory_init]  # 存储轨迹的列表

    # 常加速度模型 (CA) 外推参数 (未使用，但保留)
    F_CA = np.array([[1, delta_T], [0, 1]])
    F = np.block([
        [F_CA, np.zeros((2, 2))],
        [np.zeros((2, 2)), F_CA]
    ])  # 状态转移矩阵 (4x4)
    G = np.array([delta_T ** 2 / 2, delta_T, delta_T ** 2 / 2, delta_T]).reshape(-1, 1)
    Q_CA_unused = G @ G.T  # 过程噪声矩阵 (未使用)

    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    measurement_dim = H.shape[0]

    # 出生/消亡时间 (转换为帧数)
    born_dead_time = [
        [delta_T, 160], [delta_T, 160], [delta_T, 160], [delta_T, 160],
        [delta_T, 200], [delta_T, 200], [delta_T, 200], [delta_T, 200], [5 * delta_T, 180],
        [5 * delta_T, 200], [5 * delta_T, 200], [5 * delta_T, 200], [5 * delta_T, 200]
    ]
    born_dead_set = [[int(t[0] / delta_T), int(t[1] / delta_T)] for t in born_dead_time]


    # --- 2. 计算状态转移矩阵 (调用上一个脚本的函数) ---
    alpha = 0.01
    beta = 0.4
    gamma = 1e-4
    w = 0
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数

    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    # 构造一个“模板”字典
    template = {
        'F': F,
        'F_follower': F_follower,
        'E_follower': E_follower,
        'F_leader': F_leader,
        'leader_ID': 0,
        'repulsive': [R1, R2],
        'type': 'VL',
        'w': w,
        'obsPos': []
    }

    # --- 3. 轨迹生成 (分阶段处理群组动态) ---

    # MATLAB 的 cell 数组转换为 Python list
    group_center_vel = [
        np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
    ]

    # **阶段 1: 帧 2 到 48/delta_T** (T1 T2), (T3 T4), (T5 T6 T7 T8), (T9)
    group = [[1, 2], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13]]
    frame_end_1 = int(32 / delta_T)

    # 为每个群计算群参数
    group_param = [ template.copy() for _ in range(5)] # 复制多份

    # 群1
    alpha = 0.01
    beta = 0.4
    gamma = 1e-4
    w = 0
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[0]['F'] = F
    group_param[0]['F_follower'] = F_follower
    group_param[0]['E_follower'] = E_follower
    group_param[0]['F_leader'] = F_leader
    group_param[0]['repulsive'] = np.array([R1, R2])
    group_param[0]['type'] = 'VL'
    group_param[0]['w'] = 'w'

    group_param[1]['F'] = F
    group_param[1]['F_follower'] = F_follower
    group_param[1]['E_follower'] = E_follower
    group_param[1]['F_leader'] = F_leader
    group_param[1]['repulsive'] = np.array([R1, R2])
    group_param[1]['type'] = 'VL'
    group_param[1]['w'] = 'w'

    group_param[2]['F'] = F
    group_param[2]['F_follower'] = F_follower
    group_param[2]['E_follower'] = E_follower
    group_param[2]['F_leader'] = F_leader
    group_param[2]['repulsive'] = np.array([R1, R2])
    group_param[2]['type'] = 'VL'
    group_param[2]['w'] = 'w'

    alpha = 0.01
    beta = 0.4
    gamma = 1e-4
    w = 2 * np.pi/180
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[3]['F'] = F
    group_param[3]['F_follower'] = F_follower
    group_param[3]['E_follower'] = E_follower
    group_param[3]['F_leader'] = F_leader
    group_param[3]['leader_ID'] = 10
    group_param[3]['repulsive'] = np.array([R1, R2])
    group_param[3]['type'] = 'LF'
    group_param[3]['w'] = 'w'


    group_param[4]['F'] = F
    group_param[4]['type'] = 'VL'
    group_param[4]['w'] = 'w'

    for frame_index in range(2, frame_end_1 + 1):
        trajectory, group_num = compute_group_move(group, trajectory, group_param, born_dead_set, frame_index,
                                                   group_center_vel)
        group_num_truth.append(group_num)

    # **阶段 2: 帧 48/delta_T + 1 到 138/delta_T** (T1 T2 T3 T4 合并)
    group = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]]
    group_center_vel = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
    frame_start_2 = frame_end_1 + 1
    frame_end_2 = int(92 / delta_T)

    # 为每个群计算群参数
    group_param = [ template.copy() for _ in range(3)] # 复制多份

    # 群1
    alpha = 0.01
    beta = 0.4
    gamma = 1e-4
    w = 2*np.pi/180
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[0]['F'] = F
    group_param[0]['F_follower'] = F_follower
    group_param[0]['E_follower'] = E_follower
    group_param[0]['F_leader'] = F_leader
    group_param[0]['repulsive'] = np.array([R1, R2])
    group_param[0]['type'] = 'VL'
    group_param[0]['w'] = 'w'

    alpha = 0.01
    beta = 0.4
    gamma = 1e-4
    w = 4*np.pi/180
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    F = np.array([
        [1, np.sin(w * delta_T) / w, 0, -(1 - np.cos(w * delta_T)) / w],
        [0, np.cos(w * delta_T), 0, -np.sin(w * delta_T)],
        [0, (1 - np.cos(w * delta_T)) / w, 1, np.sin(w * delta_T) / w],
        [0, np.sin(w * delta_T), 0, np.cos(w * delta_T)]
    ])
    group_param[1]['F'] = F
    group_param[1]['F_follower'] = F_follower
    group_param[1]['E_follower'] = E_follower
    group_param[1]['F_leader'] = F_leader
    group_param[1]['repulsive'] = np.array([R1, R2])
    group_param[1]['type'] = 'indep'
    group_param[1]['w'] = 'w'

    alpha = 0.01
    beta = 0.4
    gamma = 1e-4
    w = 0.5*np.pi/180
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    F = np.array([
        [1, np.sin(w * delta_T) / w, 0, -(1 - np.cos(w * delta_T)) / w],
        [0, np.cos(w * delta_T), 0, -np.sin(w * delta_T)],
        [0, (1 - np.cos(w * delta_T)) / w, 1, np.sin(w * delta_T) / w],
        [0, np.sin(w * delta_T), 0, np.cos(w * delta_T)]
    ])
    group_param[2]['F'] = F
    group_param[2]['F_follower'] = F_follower
    group_param[2]['E_follower'] = E_follower
    group_param[2]['F_leader'] = F_leader
    group_param[2]['leader_ID'] = 10
    group_param[2]['repulsive'] = np.array([R1, R2])
    group_param[2]['type'] = 'LF'
    group_param[2]['w'] = 'w'


    for frame_index in range(frame_start_2, frame_end_2 + 1):
        trajectory, group_num = compute_group_move(group, trajectory, group_param, born_dead_set, frame_index,
                                                   group_center_vel)
        group_num_truth.append(group_num)

    # **阶段 3: 帧 138/delta_T + 1 到 180/delta_T** (T5 T6 T7 T8 分裂)
    group = [[1, 2, 3, 4], [5, 6, 7, 8], [9], [10, 11, 12, 13]]
    frame_start_3 = frame_end_2 + 1
    frame_end_3 = int(120 / delta_T)

    # 为每个群计算群参数
    group_param = [ template.copy() for _ in range(4)] # 复制多份

    # 群1
    alpha = 0.02
    beta = 0.4
    gamma = 1e-4
    w = 0*np.pi/180
    R1 = 20  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[0]['F'] = F
    group_param[0]['F_follower'] = F_follower
    group_param[0]['E_follower'] = E_follower
    group_param[0]['F_leader'] = F_leader
    group_param[0]['repulsive'] = np.array([R1, R2])
    group_param[0]['type'] = 'VL'
    group_param[0]['w'] = 'w'

    group_param[1]['F'] = F
    group_param[1]['F_follower'] = F_follower
    group_param[1]['E_follower'] = E_follower
    group_param[1]['F_leader'] = F_leader
    group_param[1]['repulsive'] = np.array([R1, R2])
    group_param[1]['type'] = 'VL'
    group_param[1]['w'] = 'w'

    F_CA = np.array([[1, delta_T], [0, 1]])
    F = np.block([
        [F_CA, np.zeros((2, 2))],
        [np.zeros((2, 2)), F_CA]
    ])  # 状态转移矩阵 (4x4)
    group_param[2]['F'] = F
    group_param[2]['type'] = 'indep'
    group_param[2]['w'] = 'w'

    alpha = -0.01
    beta = 0.4
    gamma = 1e-4
    w = -2*np.pi/180
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[3]['F'] = F
    group_param[3]['F_follower'] = F_follower
    group_param[3]['E_follower'] = E_follower
    group_param[3]['F_leader'] = F_leader
    group_param[3]['leader_ID'] = 12
    group_param[3]['repulsive'] = np.array([R1, R2])
    group_param[3]['type'] = 'LF'
    group_param[3]['w'] = 'w'


    for frame_index in range(frame_start_3, frame_end_3 + 1):
        trajectory, group_num = compute_group_move(group, trajectory, group_param, born_dead_set, frame_index,
                                                   group_center_vel)
        group_num_truth.append(group_num)

    # **阶段 4: 帧 138/delta_T + 1 到 180/delta_T** (T5 T6 T7 T8 分裂)
    group = [[1, 3], [2, 4], [5, 6, 7, 8], [9], [10, 11, 12, 13]]
    frame_start_4 = frame_end_3 + 1
    frame_end_4 = int(180 / delta_T)

    # 为每个群计算群参数
    group_param = [ template.copy() for _ in range(5)] # 复制多份

    # 群1
    alpha = 0.02
    beta = 0.4
    gamma = 1e-4
    w = -2*np.pi/180
    R1 = 20  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[0]['F'] = F
    group_param[0]['F_follower'] = F_follower
    group_param[0]['E_follower'] = E_follower
    group_param[0]['F_leader'] = F_leader
    group_param[0]['repulsive'] = np.array([R1, R2])
    group_param[0]['type'] = 'VL'
    group_param[0]['w'] = 'w'

    alpha = 0.02
    beta = 0.4
    gamma = 1e-4
    w = 2*np.pi/180
    R1 = 20  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)
    group_param[1]['F'] = F
    group_param[1]['F_follower'] = F_follower
    group_param[1]['E_follower'] = E_follower
    group_param[1]['F_leader'] = F_leader
    group_param[1]['repulsive'] = np.array([R1, R2])
    group_param[1]['type'] = 'VL'
    group_param[1]['w'] = 'w'


    alpha = 0.02
    beta = 0.4
    gamma = 1e-4
    w = -4*np.pi/180
    R1 = 20  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)
    F = np.array([
        [1, np.sin(w * delta_T) / w, 0, -(1 - np.cos(w * delta_T)) / w],
        [0, np.cos(w * delta_T), 0, -np.sin(w * delta_T)],
        [0, (1 - np.cos(w * delta_T)) / w, 1, np.sin(w * delta_T) / w],
        [0, np.sin(w * delta_T), 0, np.cos(w * delta_T)]
    ])
    group_param[2]['F'] = F
    group_param[2]['F_follower'] = F_follower
    group_param[2]['E_follower'] = E_follower
    group_param[2]['F_leader'] = F_leader
    group_param[2]['repulsive'] = np.array([R1, R2])
    group_param[2]['type'] = 'indep'
    group_param[2]['w'] = 'w'

    w = 2 * np.pi / 180
    F = np.array([
        [1, np.sin(w * delta_T) / w, 0, -(1 - np.cos(w * delta_T)) / w],
        [0, np.cos(w * delta_T), 0, -np.sin(w * delta_T)],
        [0, (1 - np.cos(w * delta_T)) / w, 1, np.sin(w * delta_T) / w],
        [0, np.sin(w * delta_T), 0, np.cos(w * delta_T)]
    ])
    group_param[3]['F'] = F
    group_param[3]['type'] = 'indep'
    group_param[3]['w'] = 'w'

    alpha = 0.01
    beta = 0.4
    gamma = 1e-4
    w = -3*np.pi/180
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[4]['F'] = F
    group_param[4]['F_follower'] = F_follower
    group_param[4]['E_follower'] = E_follower
    group_param[4]['F_leader'] = F_leader
    group_param[4]['leader_ID'] = 12
    group_param[4]['repulsive'] = np.array([R1, R2])
    group_param[4]['type'] = 'LF'
    group_param[4]['w'] = 'w'


    for frame_index in range(frame_start_4, frame_end_4 + 1):
        trajectory, group_num = compute_group_move(group, trajectory, group_param, born_dead_set, frame_index,
                                                   group_center_vel)
        group_num_truth.append(group_num)


    # **阶段 5: 帧 138/delta_T + 1 到 180/delta_T** (T5 T6 T7 T8 分裂)
    group = [[5, 6, 7, 8], [9], [10, 11, 12, 13]]
    frame_start_5 = frame_end_4 + 1
    frame_end_5 = int(200 / delta_T)

    # 为每个群计算群参数
    group_param = [ template.copy() for _ in range(3)] # 复制多份

    # 群1
    alpha = 0.02
    beta = 0.4
    gamma = 1e-4
    w = 0*np.pi/180
    R1 = 20  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[0]['F'] = F
    group_param[0]['F_follower'] = F_follower
    group_param[0]['E_follower'] = E_follower
    group_param[0]['F_leader'] = F_leader
    group_param[0]['repulsive'] = np.array([R1, R2])
    group_param[0]['type'] = 'VL'
    group_param[0]['w'] = 'w'


    w = 2 * np.pi / 180
    F = np.array([
        [1, np.sin(w * delta_T) / w, 0, -(1 - np.cos(w * delta_T)) / w],
        [0, np.cos(w * delta_T), 0, -np.sin(w * delta_T)],
        [0, (1 - np.cos(w * delta_T)) / w, 1, np.sin(w * delta_T) / w],
        [0, np.sin(w * delta_T), 0, np.cos(w * delta_T)]
    ])
    group_param[1]['F'] = F
    group_param[1]['type'] = 'indep'
    group_param[1]['w'] = 'w'

    alpha = -0.01
    beta = 0.4
    gamma = 1e-4
    w = 0*np.pi/180
    R1 = 10  # 斥力参数
    R2 = 8  # 斥力参数
    F_follower, E_follower, F_leader, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)

    group_param[2]['F'] = F
    group_param[2]['F_follower'] = F_follower
    group_param[2]['E_follower'] = E_follower
    group_param[2]['F_leader'] = F_leader
    group_param[2]['leader_ID'] = 12
    group_param[2]['repulsive'] = np.array([R1, R2])
    group_param[2]['type'] = 'LF'
    group_param[2]['w'] = 'w'


    for frame_index in range(frame_start_5, frame_end_5 + 1):
        trajectory, group_num = compute_group_move(group, trajectory, group_param, born_dead_set, frame_index,
                                                   group_center_vel)
        group_num_truth.append(group_num)

    # --- 4. 绘图 (Plotting) ---

    plt.figure()
    plt.title('Generated Group Trajectories')
    plt.xlabel('X axis (m)')
    plt.ylabel('Y axis (m)')
    plt.grid(True)
    plots = []

    for tar_index in range(target_num):
        single_trajectory = trajectory[tar_index]

        # 提取 X 和 Y 坐标 (第 0 行和第 2 行)
        X = single_trajectory[0, :]
        Y = single_trajectory[2, :]

        born_frame, dead_frame = born_dead_set[tar_index]

        # 绘制轨迹 (黑色实线)
        p1, = plt.plot(X, Y, 'k-')

        # 绘制起点 (黑色圆圈)
        p2, = plt.plot(X[0], Y[0], 'ko')

        # 绘制终点 (黑色三角形)
        p3, = plt.plot(X[-1], Y[-1], 'k^')

    # MATLAB 的 legend 只需要其中一个句柄来表示 'ground truth'
    plt.legend([p1, p2, p3], ['ground truth', 'start', 'end'])
    plt.show()

    # --- 5. 数据保存 (Data Saving) ---
    scene_dir = "scene02"
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)


    labels_set = []  # 存储每一帧的真值信息
    labels_set_num = np.zeros(frame_max, dtype=int)

    for frame in range(1, frame_max + 1):
        labels_single_frame = []  # [[target_id, x, vx, y, vy], ...]

        for tar_index in range(target_num):
            single_trajectory = trajectory[tar_index]
            born_frame, dead_frame = born_dead_set[tar_index]

            if born_frame <= frame and dead_frame >= frame:
                # 如果当前帧存活
                # MATLAB 的 loc = frame - born_dead(1) + 1 (目标轨迹中的帧索引)
                # Python 的 loc 是从 0 开始的
                loc = frame - born_frame

                # 真值: [tar_index; state_vector]
                state_vector = single_trajectory[:, loc].reshape(-1, 1)  # 4x1
                labels_single_frame.append(np.vstack([[tar_index + 1], state_vector]))

                # 观测值: [x, y]' + noise
                # (single_trajectory([1,3],loc) + measurement_noise_std * randn(2,1))
                position_true = single_trajectory[[0, 2], loc].reshape(-1, 1)  # 2x1

        if labels_single_frame:
            # 合并所有真值 (5xN)
            labels_single_frame_mat = np.hstack(labels_single_frame)
        else:
            labels_single_frame_mat = np.zeros((5, 0))

        labels_set.append(labels_single_frame_mat)
        labels_set_num[frame - 1] = labels_single_frame_mat.shape[1]


    torch.save({'trajectory':trajectory,
             'target_num':target_num,
             'frame_max':frame_end_5,
             'born_dead_set':born_dead_set,
             'delta_T':delta_T,
             'radar_azi_resolution':radar_azi_resolution,
            'radar_range_resolution':radar_range_resolution,
            'labels_set':labels_set,
            'labels_set_num':labels_set_num,
    'group_num_truth':group_num_truth},f'{scene_dir}/trajectory01.pt')




    '''
    measurement_noise_std = 5  # 观测噪声标准差
    num_clutter = 3  # 杂波数量 (未在后续代码中使用)
    scene_dir = "scene01"

    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)

    labels_set = {}  # 存储每一帧的真值信息
    labels_set_num = np.zeros(frame_max, dtype=int)

    for frame in range(1, frame_max + 1):
        labels_single_frame = []  # [[target_id, x, vx, y, vy], ...]
        detections_single_frame = []  # [[x_obs, y_obs, doppler], ...]

        for tar_index in range(target_num):
            single_trajectory = trajectory[tar_index]
            born_frame, dead_frame = born_dead_set[tar_index]

            if born_frame <= frame and dead_frame >= frame:
                # 如果当前帧存活
                # MATLAB 的 loc = frame - born_dead(1) + 1 (目标轨迹中的帧索引)
                # Python 的 loc 是从 0 开始的
                loc = frame - born_frame

                # 真值: [tar_index; state_vector]
                state_vector = single_trajectory[:, loc].reshape(-1, 1)  # 4x1
                labels_single_frame.append(np.vstack([[tar_index + 1], state_vector]))

                # 观测值: [x, y]' + noise
                # (single_trajectory([1,3],loc) + measurement_noise_std * randn(2,1))
                position_true = single_trajectory[[0, 2], loc].reshape(-1, 1)  # 2x1
                noise = measurement_noise_std * np.random.randn(2, 1)
                detection_pos = position_true + noise
                detections_single_frame.append(detection_pos)

        if labels_single_frame:
            # 合并所有真值 (5xN)
            labels_single_frame_mat = np.hstack(labels_single_frame)
        else:
            labels_single_frame_mat = np.zeros((5, 0))

        if detections_single_frame:
            # 合并所有观测值 (2xN)
            detections_single_frame_mat = np.hstack(detections_single_frame)
        else:
            detections_single_frame_mat = np.zeros((2, 0))

        labels_set[frame] = labels_single_frame_mat
        labels_set_num[frame - 1] = labels_single_frame_mat.shape[1]

        # 增加多普勒信息 (用 1 填充，因为没有实际计算多普勒)
        # detections_single_frame = [detections_single_frame ;ones(1,size(detections_single_frame,2))];
        if detections_single_frame_mat.shape[1] > 0:
            detections_single_frame_mat = np.vstack([
                detections_single_frame_mat,
                np.ones((1, detections_single_frame_mat.shape[1]))
            ])
        else:
            detections_single_frame_mat = np.zeros((3, 0))
    '''

        # 存储数据 (如果需要 MATLAB 格式，可以使用 scipy.io.savemat)
        # 这里只做了数据结构生成，没有保存到文件，你可以根据需要使用 np.save 或 scipy.io.savemat
        # print(f"Frame {frame}: Targets {labels_single_frame_mat.shape[1]}, Detections {detections_single_frame_mat.shape[1]}")

