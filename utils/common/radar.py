import numpy as np
from utils.common.DBSCAN import DBSCAN
from utils.common.set_partitions import set_partitions
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from utils.common.coordinate_transformation import *
import torch
import torch.nn as nn
from torch_scatter import scatter_mean # PyG常用的依赖库
from scipy.sparse import coo_matrix
from typing import Callable
from utils.common.track import Track
from utils.common.Partition import *
# ----------------------------------------------------------------------
# 核心类定义：Radar
# ----------------------------------------------------------------------

class Radar:
    """
    雷达系统和群目标跟踪的封装类。

    属性主要用于存储雷达硬件参数、滤波器参数和跟踪状态。
    方法实现了波门判断、代价矩阵计算和群管理等核心逻辑。
    """

    def __init__(self, external_input):
        # --- 硬件参数/环境参数 ---
        self.location_enu = external_input.get('location_enu')
        self.location_geography = external_input.get('location_geography')
        self.detection_distance_range = external_input.get('detection_distance_range')
        self.detection_azimuth_range = external_input.get('detection_azimuth_range')
        self.detection_elevation_range = external_input.get('detection_elevation_range')
        self.T = external_input.get('T')
        self.distance_measurement_error = external_input.get('distance_measurement_error')
        self.azimuth_measurement_error = external_input.get('azimuth_measurement_error')
        self.elevation_measurement_error = external_input.get('elevation_measurement_error')

        # --- 跟踪器/滤波器参数 ---
        self.tracker_method = external_input.get('tracker_method')
        self.tracker_model = external_input.get('tracker_model')
        self.associate_method = external_input.get('associate_method')
        self.lambda_NT = external_input.get('lambda_NT')
        self.lambda_FA = external_input.get('lambda_FA')
        self.lambda_FA_PerUnitVolume = external_input.get('lambda_FA_PerUnitVolume')
        self.Pd = external_input.get('Pd')
        self.Pf = external_input.get('Pf')
        self.trackProbabilityThreshold = external_input.get('trackProbabilityThreshold')
        self.Vmin = external_input.get('Vmin')
        self.Vmax = external_input.get('Vmax')
        self.PG = external_input.get('PG')
        self.Chi_large = external_input.get('Chi_large')
        self.Chi_small = external_input.get('Chi_small')
        self.distance_step = external_input.get('distance_step')
        self.azi_step = external_input.get('azi_step')
        self.Ps = external_input.get('Ps')
        self.Pb = external_input.get('Pb')
        self.r_init = external_input.get('r_init')
        self.existence_threshold_legacyPT = external_input.get('existence_threshold_legacyPT')
        self.existence_threshold_newPT = external_input.get('existence_threshold_newPT')
        self.x_dimension = external_input.get('x_dimension')
        self.z_dimension = external_input.get('z_dimension')
        self.maximumNumberOfGmComponents = external_input.get('maximumNumberOfGmComponents')
        self.gmWeightThreshold = external_input.get('gmWeightThreshold')
        self.confirm_trajectory_len = external_input.get('confirm_trajectory_len')

        # --- 群目标/分辨率参数 ---
        self.radar_range_resolution = 0.0  # 初始值在 set_resoultion 中设置
        self.radar_azi_resolution = 0.0  # 初始值在 set_resoultion 中设置
        self.group_extend_space = external_input.get('group_extend_space')  # m
        self.group_sigma_r = external_input.get('group_sigma_r')  # 距离相关参数
        self.group_sigma_v = external_input.get('group_sigma_v')  # 速度相关参数
        self.max_group_partition = external_input.get('max_group_partition')

        # --- 内部状态/数据结构 ---
        self.tracker = None  # 跟踪器对象 (需要外部赋值)
        self.group_manager = {}  # 群管理相关的参数和状态
        self.neural_network = {}  # 神经网络 (需要外部赋值)
        self.total_group_num = []
        # --- 神经网络 ---
        self.use_neural_network = external_input.get('use_neural_network') # 是否启用神经网络
        self.track_len_for_nn = external_input.get('track_len_for_nn')  # 用于神经网络的航迹最短长度
        self.track_len_for_adj = external_input.get('track_len_for_adj')  # 基于建立节点之间关系的航迹最短长度



        self.plot_track_set = []
        self.track_set = []  # 航迹集合 (list of dict/object)
        self.track_header_set = []  # 航迹头集合 (list of dict/object)
        self.legacy_PT_set = []  # 遗留潜在目标集合 (list of dict/object)
        self.new_PT_set = []  # 新潜在目标集合 (list of dict/object)

        self.track_header_num = 0
        self.track_num = 0
        self.legacy_PT_num = 0
        self.new_PT_num = 0
        self.plot_track_num = 0

        # 可用索引池 (初始化 1 到 10000)
        self.track_index_set = list(range(1, 10001))
        self.head_index_set = list(range(1, 10001))

        # 全局假设 (假设 MTA/JIPDA 等需要)
        self.global_hypo_num = 1
        self.global_hypo = []
        self.global_hypo_weight = 1
        self.hypo_num_max = external_input.get('hypo_num_max', 10)

        self.report = []  # 上报航迹

        self.group_history = [] # 存储历史的群划分情况
        self.graph_builder = None
    def set_resoultion(self, radar_range_resolution, radar_azi_resolution):
        """设置雷达分辨率。"""
        self.radar_range_resolution = radar_range_resolution
        self.radar_azi_resolution = radar_azi_resolution

    def find_head(self, num):
        """根据航迹号找对应的航迹头。"""
        for i, header in enumerate(self.track_header_set):
            if header.get('header_index') == num:
                return i, 1  # 返回索引和成功标志
        return -1, 0

    def find_track(self, num):
        """根据航迹号找对应的航迹。"""
        for i, track in enumerate(self.track_set):
            if track.get('track_index') == num:
                return i, 1  # 返回索引和成功标志
        return -1, 0

    # --- 关联波门判断方法 (Related Gate Checks) ---

    def Related_gate_track2ob(self, track_index, plot_track_index, Chi):
        """航迹与观测点的相关波门判断 (用于单假设或 PTA)。"""
        # track_index 和 plot_track_index 是在 self.track_set 和 self.plot_track_set 中的数组索引
        track = self.track_set[track_index]
        plot_track = self.plot_track_set[plot_track_index]

        # S = H * P * H' + R
        H = self.tracker.H
        R = track.R  # 假设 R 存储在 track 对象中
        P = track.P

        S = H @ P @ H.T + R

        # d = X_pred([1,3],end) - Z_ob
        # X 维度 (4, N), Z 维度 (2, 1)
        d = track.X[[0, 2], -1].reshape(-1, 1) - plot_track.X  # X[[0, 2], -1] 是预测的位置

        # D = d' * inv(S) * d
        D = d.T @ np.linalg.inv(S) @ d

        D = D.item()  # 转换为标量

        flag = 1 if D <= Chi else 0
        return D, flag

    def Related_gate_track2ob_hypo(self, track_index, plot_track_index, Chi, hypo_index):
        """假设航迹与观测点的相关波门判断 (用于 IMM/PDA 等)。"""
        track = self.track_set[track_index]
        plot_track = self.plot_track_set[plot_track_index]

        # P_hypo 和 X_hypo 是列表或字典，需要根据 hypo_index 访问
        P_hypo = track.P_hypo[hypo_index]
        X_hypo = track.X_hypo[hypo_index]

        H = self.tracker.H
        R = track.R

        S = H @ P_hypo @ H.T + R

        # d = X_pred([1,3],end) - Z_ob
        d = X_hypo[[0, 2], -1].reshape(-1, 1) - plot_track.X

        D = d.T @ np.linalg.inv(S) @ d
        D = D.item()

        flag = 1 if D <= Chi else 0
        return D, flag

    def Related_gate_head2ob(self, track_index, plot_track_index, Chi):
        """航迹头与观测点的相关波门判断 (用于航迹起始)。"""
        track_header = self.track_header_set[track_index]
        plot_track = self.plot_track_set[plot_track_index]

        # S = 2 * R (假设航迹头使用的初始协方差 P0 = R)
        R = self.tracker.R  # 假设 R 在 self.tracker 中
        S = 2 * R

        # d = X_pred([1,3],end) - Z_ob (航迹头 X 假设只存储位置)
        # 航迹头 X(:,end) 假设是 2x1 位置向量
        # MATLAB X(:,end) 是 2x1 (位置) 还是 4x1 (位置+速度) 不清楚，但波门计算只使用了位置。
        d = track_header.X[:, -1].reshape(-1, 1) - plot_track.X  # 2x1

        D = d.T @ np.linalg.inv(S) @ d
        D = D.item()

        # 速度波门判断
        # V_ob = d / T
        V_ob_norm = np.linalg.norm(d) / self.T
        Vmin_norm = np.linalg.norm(self.Vmin)
        Vmax_norm = np.linalg.norm(self.Vmax)

        # MATLAB 中的 vel_flag 逻辑似乎是反的，我们按照字面意思翻译：
        # if norm(d)/obj.T > norm(obj.Vmin) && norm(d)/obj.T < norm(obj.Vmax) -> vel_flag = 0
        # else -> vel_flag = 1
        # 最后判断 if vel_flag: flag = 1 else: flag = 0 (即只用 vel_flag)

        # 简化为：只有当速度在 Vmin 和 Vmax 之间时，才算通过。
        # 重新理解 MATLAB 逻辑：当速度 *不在* 期望范围内时，vel_flag = 1 (失败/不相关)。
        # 最后 if vel_flag -> flag=1，这意味着速度 *不* 在范围内时，判定为相关？
        # 这段 MATLAB 逻辑似乎是错误的或有特殊含义。我们暂时保持原 MATLAB 逻辑：

        if V_ob_norm > Vmin_norm and V_ob_norm < Vmax_norm:
            vel_flag = 0
        else:
            vel_flag = 1

        # 最终判断只看 vel_flag (忽略了 D <= Chi 的判断!)
        flag = 1 if vel_flag == 1 else 0

        return D, flag

    # --- 代价矩阵计算 ---

    def calCostMatrix(self, clusterSet, obIndexSet):
        """计算航迹与观测点的统计距离代价矩阵。"""
        num_tracks = len(clusterSet)
        num_obs = len(obIndexSet)
        costMatrix = np.zeros((num_tracks, num_obs))
        flagMatrix = np.zeros((num_tracks, num_obs), dtype=int)

        for i in range(num_tracks):
            track_idx = clusterSet[i]
            for j in range(num_obs):
                ob_idx = obIndexSet[j]
                # 假设 clusterSet 和 obIndexSet 存储的是数组索引 (0-based)
                # 而不是 MATLAB 中的 track_index 号。
                # 如果它们是 track_index 号，则需要先调用 find_track/find_head 转换。
                D, flag = self.Related_gate_track2ob(track_idx, ob_idx, self.Chi_large)
                costMatrix[i, j] = D
                flagMatrix[i, j] = flag

        return costMatrix, flagMatrix

    def calCostMatrixHead2Ob(self, clusterSet, obIndexSet):
        """计算航迹头与观测点的统计距离代价矩阵。"""
        num_heads = len(clusterSet)
        num_obs = len(obIndexSet)
        costMatrix = np.zeros((num_heads, num_obs))
        flagMatrix = np.zeros((num_heads, num_obs), dtype=int)

        for i in range(num_heads):
            head_idx = clusterSet[i]
            for j in range(num_obs):
                ob_idx = obIndexSet[j]
                D, flag = self.Related_gate_head2ob(head_idx, ob_idx, self.Chi_large)
                costMatrix[i, j] = D
                flagMatrix[i, j] = flag

        return costMatrix, flagMatrix

    # --- OSPA 距离计算 ---

    def ospa(self, track_estimation, ground_truth, c, p):
        """OSPA (Optimal Subpattern Assignment) 距离计算。"""
        # track_estimation 和 ground_truth 假设是 N x state_dim 矩阵
        n = track_estimation.shape[0]
        m = ground_truth.shape[0]

        # 计算欧氏距离 (D)
        # 使用 pdist2 库函数 (scipy.spatial.distance.cdist 更常用，但这里用 pdist2 占位)
        # 假设 pdist2 的行为与 MATLAB 相似，需要手动实现或使用 cdist
        from scipy.spatial.distance import cdist
        D = cdist(track_estimation, ground_truth, metric='euclidean')  # n x m

        # 截断和 p 阶幂
        D = np.minimum(c, D) ** p

        # 使用 Hungarian 算法计算最优指派和代价
        # linear_sum_assignment 找到最小化代价的和的行和列索引
        row_ind, col_ind = linear_sum_assignment(D)

        # 计算最优指派下的总代价 (d)
        d = D[row_ind, col_ind].sum()

        # 计算最终 OSPA 距离
        # dist= ( 1/max(m,n)*( c^p*abs(m-n)+ d ) ) ^(1/p)
        dist = (1 / max(m, n) * (c ** p * np.abs(m - n) + d)) ** (1 / p)

        return dist

    # --- 群管理逻辑 (Compute Group Manager) ---

    def compute_group_manager(self, K_best, candidate_time_len):
        """
        在 VL 模型下计算群管理需要的参数，包括可能的群分区、对应的概率以及每个群的斥力。

        注意：这是一个非常复杂的函数，涉及到 DBSCAN、集合分区、概率更新和消息传递。
        此处仅提供结构转换，并假设依赖的属性 (legacy_PT_set, tracker.compute_repulsion, etc.) 已正确定义。
        """
        group_const_no_exist_PT = 1e-3

        # 1. DBSCAN 聚类
        param = {
            'H': self.tracker.H,
            'R': self.tracker.R,
            'H_vel': np.array([[0, 1, 0, 0], [0, 0, 0, 1]]),
            'dbscan_param': {'distance_gate': 150}
        }
        # IDX: 目标的簇索引 (1-based)
        IDX, C, adjacent_matrix = DBSCAN(self.legacy_PT_set, self.legacy_PT_num, param)

        # 2. 计算目标对马氏距离和映射矩阵
        Sigma_group = np.eye(self.x_dimension) * 30 ** 2
        pair_cnt = 0
        pair_map = np.zeros((self.legacy_PT_num, self.legacy_PT_num), dtype=int)
        target_pair = [None] * (self.legacy_PT_num * (self.legacy_PT_num - 1) // 2)

        meaasge_pair_match = []
        meaasge_pair_no_match = []
        score_pair_match = []
        score_pair_no_match = []

        for idx1 in range(self.legacy_PT_num - 1):
            for idx2 in range(idx1 + 1, self.legacy_PT_num):
                target_idx = pair_cnt
                target_pair[target_idx] = {'pair_index': [idx1, idx2]}

                # 计算马氏距离相关参数
                X1 = self.legacy_PT_set[idx1].X[-1].reshape(-1, 1)
                X2 = self.legacy_PT_set[idx2].X[-1].reshape(-1, 1)
                P1 = self.legacy_PT_set[idx1].P
                P2 = self.legacy_PT_set[idx2].P

                # 位置距离（考虑扩展空间）
                d_pos = self.tracker.H @ (X1 - X2)
                d = max(0, np.linalg.norm(d_pos) - self.group_extend_space)

                # 速度距离
                d_v = np.linalg.norm(X1[1::2] - X2[1::2])

                # 计算马氏距离得分
                score_r = np.exp(-0.5 * d ** 2 / self.group_sigma_r ** 2)
                score_v = np.exp(-0.5 * d_v ** 2 / self.group_sigma_v ** 2)
                target_pair[target_idx]['d'] = score_r * score_v

                # 填充映射矩阵
                pair_map[idx1, idx2] = pair_cnt
                pair_map[idx2, idx1] = pair_cnt

                # 计算概率相关项
                r1 = self.legacy_PT_set[idx1].r
                r2 = self.legacy_PT_set[idx2].r

                const_value = group_const_no_exist_PT * r1 * (1 - r2) + \
                              group_const_no_exist_PT * (1 - r1) * r2 + \
                              group_const_no_exist_PT * (1 - r1) * (1 - r2)

                meaasge_pair_match.append(np.log(target_pair[target_idx]['d'] * r1 * r2 + const_value + 1e-12))
                meaasge_pair_no_match.append(np.log((1 - target_pair[target_idx]['d']) * r1 * r2 + const_value + 1e-12))

                score_pair_match.append(np.log(target_pair[target_idx]['d'] * r1 * r2 + 1e-12))
                score_pair_no_match.append(np.log((1 - target_pair[target_idx]['d']) * r1 * r2 + 1e-12))

                pair_cnt += 1

        meaasge_pair_match = np.array(meaasge_pair_match)
        meaasge_pair_no_match = np.array(meaasge_pair_no_match)
        score_pair_match = np.array(score_pair_match)
        score_pair_no_match = np.array(score_pair_no_match)

        # 3. 对每个簇生成群分区
        group_partition = [[] for _ in range(C)]

        for C_index in range(C):
            # 找到属于当前簇的目标索引
            tar_index = np.where(IDX == C_index + 1)[0]  # IDX是1-based

            unconfirmed_index_set = []
            confirmed_index_set = []

            for idx in tar_index:
                if len(self.legacy_PT_set[idx].X) <= candidate_time_len:
                    unconfirmed_index_set.append(idx)
                else:
                    confirmed_index_set.append(idx)

            if confirmed_index_set:
                # 对确认目标生成分区
                partitions = limited_set_partitions(confirmed_index_set, pair_map, pair_cnt, self.max_group_partition)

                # 添加将所有目标放在一个群的分区
                current_parts = confirmed_index_set.copy()
                new_mask = np.ones(pair_cnt, dtype=bool)
                group_partition_tmp = Partition(groups=[current_parts], pair_mask=new_mask)

                # MATLAB中是将这个分区添加到末尾
                partitions.append(group_partition_tmp)

                # 将未确认目标添加到每个分区的每个群中
                for part in partitions:
                    part.groups.extend([[u] for u in unconfirmed_index_set])

                group_partition[C_index] = partitions
            else:
                # 只有未确认目标
                group_partition[C_index] = [Partition(groups=[[u] for u in unconfirmed_index_set],
                                                      pair_mask=np.zeros(pair_cnt, dtype=bool))]

        # 4. 计算群状态、群质心和斥力
        group_state = [[] for _ in range(len(group_partition))]

        for cluster_idx in range(len(group_partition)):
            for partition_idx in range(len(group_partition[cluster_idx])):
                partition_states = []
                for group_members in group_partition[cluster_idx][partition_idx].groups:
                    # 收集群内所有目标的状态
                    states = []
                    for pt_idx in group_members:
                        states.append(self.legacy_PT_set[pt_idx].X[-1].reshape(-1, 1))

                    if states:
                        state = np.hstack(states)
                        center = np.mean(state, axis=1, keepdims=True)

                        group_info = {'center': center, 'state': state}

                        # 计算斥力
                        if state.shape[1] == 1:
                            group_info['repulsion'] = 0
                        else:
                            # 假设tracker有compute_repulsion方法
                            if hasattr(self.tracker, 'compute_repulsion'):
                                group_info['repulsion'] = self.tracker.compute_repulsion(state)
                            else:
                                group_info['repulsion'] = 0

                        partition_states.append(group_info)
                    else:
                        # 空群（理论上不会发生）
                        partition_states.append({'center': np.zeros((self.x_dimension, 1)),
                                                 'state': np.zeros((self.x_dimension, 0)),
                                                 'repulsion': 0})

                if partition_idx >= len(group_state[cluster_idx]):
                    group_state[cluster_idx].append(partition_states)
                else:
                    group_state[cluster_idx][partition_idx] = partition_states

        # 5. 计算每个群的得分和概率
        group_prob = []
        group_message = []

        for cluster_idx in range(len(group_partition)):
            num_partitions = len(group_partition[cluster_idx])
            group_score_log = np.zeros(num_partitions)
            group_message_log = np.zeros(num_partitions)

            for part_idx in range(num_partitions):
                partition = group_partition[cluster_idx][part_idx]

                # 使用pair_mask计算得分（MATLAB方式）
                group_score_log[part_idx] = np.sum(partition.pair_mask * score_pair_match) + \
                                            np.sum(~partition.pair_mask * score_pair_no_match)
                group_message_log[part_idx] = np.sum(partition.pair_mask * meaasge_pair_match) + \
                                              np.sum(~partition.pair_mask * meaasge_pair_no_match)

            # 归一化群概率
            maxLog_score = np.max(group_score_log) if len(group_score_log) > 0 else 0
            prob_norm = np.exp(group_score_log - maxLog_score)
            eps = 1e-5
            prob_norm = prob_norm + eps
            cluster_prob = prob_norm / np.sum(prob_norm)

            # 归一化群消息
            maxLog_msg = np.max(group_message_log) if len(group_message_log) > 0 else 0
            msg_norm = np.exp(group_message_log - maxLog_msg)
            eps_msg = 1e-10
            msg_norm = msg_norm + eps_msg
            cluster_msg = msg_norm / np.sum(msg_norm)

            # K-BEST截断
            K = max(K_best, np.sum(cluster_prob == np.max(cluster_prob)))

            # 降序排序
            sorted_indices = np.argsort(cluster_prob)[::-1]
            keep_indices = sorted_indices[:min(len(sorted_indices), K)]

            # 截断
            group_prob.append(cluster_prob[keep_indices])
            group_message.append(cluster_msg[keep_indices])

            # 更新group_state和group_partition
            group_state[cluster_idx] = [group_state[cluster_idx][i] for i in keep_indices]
            group_partition[cluster_idx] = [group_partition[cluster_idx][i] for i in keep_indices]

            # 再次归一化
            group_prob[-1] = group_prob[-1] / np.sum(group_prob[-1])
            group_message[-1] = group_message[-1] / np.sum(group_message[-1])

        # 6. 转换group_partition格式（只保留groups）
        for cluster_idx in range(len(group_partition)):
            for part_idx in range(len(group_partition[cluster_idx])):
                group_partition[cluster_idx][part_idx] = group_partition[cluster_idx][part_idx].groups

        # 7. 存储结果
        self.group_manager = {
            'group_partition': group_partition,
            'group_state': group_state,
            'group_prob': group_prob,
            'group_message': group_message,
            'IDX': IDX
        }

        return group_partition, group_state, group_prob, IDX

    def state_to_adj(self, states, survival_duration, duration_threshold, max_radius, directed=False, metric='euclidean'):
        """
        状态向量 → 邻接矩阵（稀疏 CSR）
        参数
        ----
        states : (N, d) 数组，每行一个 d 维状态向量
        max_radius : 标量，两目标距离 ≤ max_radius 才连边
        directed : False 则生成对称无向图；True 则保持有向
        metric : 距离度量，同 scipy.spatial.distance.cdist
        返回
        ----
        adj : scipy.sparse.csr_matrix, shape=(N,N), 0/1 矩阵
        """
        # 1. 计算 pairwise 距离
        dist = cdist(states, states, metric=metric)  # (N,N)
        # 2. 根据阈值生成边
        adj = (dist <= max_radius).astype(np.uint8)
        if not directed:  # 强制对称（无向图）
            adj = np.maximum(adj, adj.T)
        # 2) 隔离低 SNR 目标
        if survival_duration is not None:
            mask = (survival_duration >= duration_threshold)  # True 表示合格
            # 合格目标之间才保留边；不合格目标整行整列置 0
            adj = adj * mask[:, None] * mask[None, :]
        # 3. 转稀疏，去掉对角 self-loop
        adj = csr_matrix(adj)
        adj.setdiag(0)
        adj.eliminate_zeros()
        return adj

    def max_connected_component(self, adj, directed=False):
        """
        返回最大连通分量里的节点索引
        若还想要所有分量，可额外返回 labels
        """
        n_comp, labels = connected_components(adj, directed=directed, connection='strong' if directed else 'weak')
        # 找最大
        max_cid = np.argmax(np.bincount(labels))
        nodes = np.where(labels == max_cid)[0]
        return nodes, labels

    def cluster_targets(self, states, survival_duration, duration_threshold, max_radius, directed=False, metric='euclidean'):
        """
        一站式：状态向量 → 所有连通分量（群）列表
        返回
        ----
        clusters : list[array] , 每个 array 是同一群的目标索引
        """
        adj = self.state_to_adj(states, survival_duration, duration_threshold, max_radius, directed, metric)
        n_comp, labels = connected_components(adj, directed=directed,
                                              connection='strong' if directed else 'weak')
        clusters = [np.where(labels == cid)[0] for cid in range(n_comp)]
        # 按大小降序，方便看最大群
        clusters.sort(key=len, reverse=True)
        return clusters

    def minmax_scaler(self, graph_data, config):
        '''
        基于固定尺度的 Min-Max 归一化函数，用于群目标追踪数据。
        :input
        graph_data 包含 x [node_num, T, state_dim]
        y [node_num, 1, state_dim]
        edge_feature_target [node_num, 1, space_dim]
        edge_feature_input [node_num, T, space_dim]
        measurement [node_num, T, space_dim]
        residual [node_num, T, space_dim]
        :return:
        '''

        # 深度复制数据，避免修改原始输入
        graph_data_norm = graph_data.clone()
        x = graph_data.x.clone()
        edge_features = [t.clone() for t in graph_data.edge_features]
        measurement = graph_data.measurement.clone()
        residual = graph_data.residual.clone()

        # 根据历史时刻的信息 构建邻接矩阵
        node_num, _, _ = graph_data.x.shape
        adj = torch.zeros((node_num,node_num))
        adj_matrix_single_frame = []
        for egge_idx in graph_data.edge_index:
            if len(egge_idx) > 0:
                row = egge_idx[0, :]
                col = egge_idx[1, :]
                data = np.ones(len(row))
                # 构建无向图的邻接矩阵
                adj_matrix_single_frame.append(coo_matrix((data, (row, col)), shape=(node_num, node_num)))
            else:
                # 没有边，说明所有点都是孤立的群
                adj_matrix_single_frame.append(coo_matrix((node_num, node_num)))

        # 检查所有矩阵形状是否相同
        shape = adj_matrix_single_frame[0].shape
        for mat in adj_matrix_single_frame:
            if mat.shape != shape:
                raise ValueError("所有矩阵必须具有相同的形状")

        result = sum((mat.astype(bool) for mat in adj_matrix_single_frame)) > 0
        result = result.astype(np.int8).tocoo()
        # 计算连通分量
        # n_components: 群的数量
        # labels: Shape (N,), labels[i] 表示第 i 行属于哪个群 ID
        n_components, labels = connected_components(result, directed=False)

        # 取出最新时刻的位置信息 (N, 2)
        latest_positions = x[:, -1, [0,2]]

        # 准备一个数组存储每个节点应该减去的偏移量 (N, 2)
        centroids_offset = np.zeros((node_num, 2), dtype=np.float32)

        # 遍历每个群，计算质心
        row_to_tid = {v: k for k, v in self.graph_builder.track_id_to_row.items()}
        active_id_list = [row_to_tid.get(i, -1) for i in range(len(row_to_tid))]
        active_row_indices = np.array(list(self.graph_builder.track_id_to_row.values())) # 取出被占用的行号
        unique_labels = np.unique(labels[active_row_indices]) # 群的编号

        for label_id in unique_labels:
            # 找到属于该群且当前存活的节点索引
            group_member_mask = (labels == label_id)

            # 获取这些成员的位置
            member_positions = latest_positions[group_member_mask]

            # 计算质心 (Mean)
            if len(member_positions) > 0:
                centroid = torch.mean(member_positions, axis=0)

                # 将质心赋值给该群的所有成员 (包括历史数据中的对应行)
                # 注意：这里我们对该群所有行都赋值，即使某些行在 T-1 时刻可能不属于这个群
                # (但在基于最新时刻归一化的定义下，我们按最新关系回溯)
                group_row_indices = np.where(labels == label_id)[0]
                centroids_offset[group_row_indices] = centroid


        # 执行归一化
        # 减去质心
        x[:,:,[0,2]]  -= centroids_offset[:, None, :]
        measurement -= centroids_offset[:, None, :]
        # 位置缩放
        x[:, :, [0, 2]] = x[:, :, [0, 2]] / config['S_POS']
        measurement = measurement / config['S_POS']
        # 速度缩放
        x[:, :, [1, 3]] = x[:, :, [1, 3]] / config['S_VEL']

        # 边特征归一化
        edge_features = [t / torch.tensor([config['S_EDGE_P_DIST'], config['S_EDGE_V_DIST']]) for t in edge_features]

        residual = residual / config['S_RES']

        graph_data_norm.x = x

        graph_data_norm.edge_features = edge_features
        graph_data_norm.measurement = measurement
        graph_data_norm.residual = residual

        graph_data_norm.x = graph_data_norm.x[..., [0, 2, 1, 3]]

        return graph_data_norm, centroids_offset


    def rescaler(self, states_input, centroids_offset, config, track_len_for_nn, graph_input, P_scaler):
        states = states_input.clone().detach().numpy()
        states = states[...,[0,2,1,3]]
        states[:,0,:,::2] = (states[:,0,:,::2]) * config['S_POS']  + centroids_offset[:,None,:]
        states[:,0,:,1::2] = (states[:,0,:,1::2]) * config['S_VEL']

        for track in self.legacy_PT_set:
            if len(track.X) <= track_len_for_nn and track.predict_flag is True:
                continue
            row_idx = self.graph_builder.track_id_to_row[track.track_index] # 获取行号

            X_predict = states[row_idx,0,0,:].reshape(-1,1)

            _, P_pred = self.tracker.KalmanPredict(np.zeros_like(X_predict), track.gmComponents[0].P, track.Q_coe)

            track.gmComponents[0].X = X_predict
            track.gmComponents[0].P = P_pred.copy()
            track.gmComponents[0].X_group_partition = np.expand_dims(X_predict, axis=-1)
            track.gmComponents[0].P_group_partition = np.expand_dims(P_pred, axis=-1)

            track.P_scaler = P_scaler[row_idx, :, :, :]
            track.X.append(X_predict)
            track.P = P_pred
            track.X_predict = X_predict
            track.P_predict = P_pred
            az, elev, slantRange = enu2aer(X_predict[0], X_predict[2], 0)
            if az > 180:
                az = az - 360
            track.Polar_X = np.array([slantRange, az, elev]).T

            # 判断是否超出探测范围
            in_dist = (self.detection_distance_range[0] <= slantRange <=
                       self.detection_distance_range[1])
            in_azi = (self.detection_azimuth_range[0] <= az <=
                      self.detection_azimuth_range[1])
            in_elev = (self.detection_elevation_range[0] <= elev <=
                       self.detection_elevation_range[1])
            if in_dist and in_azi and in_elev:
                track.track_property = track.track_property
            else:
                track.track_property = 2  # 超出范围 不再继续跟踪

    def minmax_scaler_update_step(self, state_input, measurement_input):
        '''
        在更新步骤中，实现对输入的归一化操作
        :param self:
        :param state_input:
        :param measurement_input:
        :return:
        '''
        state_input_clone = state_input.clone()
        measurement_input_clone = measurement_input.clone()
        centroids_x =  state_input_clone[:, -2, [0, 2]]
        state_input_clone[:, :, [0, 2]] = state_input_clone[:, :, [0, 2]] - centroids_x[:,None,:]
        state_input_clone[:, :, [0, 2]] = state_input_clone[:, :, [0, 2]] / self.neural_network['config']['S_POS']
        state_input_clone[:, :, [1, 3]] = state_input_clone[:, :, [1, 3]] / self.neural_network['config']['S_VEL']
        state_input_clone = state_input_clone[..., [0, 2, 1, 3]]

        measurement_input_clone = (measurement_input_clone - centroids_x[:,None,:]) / self.neural_network['config']['S_POS']

        return state_input_clone.to(torch.float32), measurement_input_clone.to(torch.float32), centroids_x.to(torch.float32)

    def rescaler_update_step(self, state_input, centroids_x):
        '''
        更新步骤中，对输出结果反归一化
        :param state_input:
        :param centroids_x:
        :return:
        '''
        state_input_renorm = state_input.clone().detach().numpy()
        state_input_renorm = state_input_renorm[...,[0,2,1,3]]
        state_input_renorm[:,:,::2] = (state_input_renorm[:,:,::2]) * self.neural_network['config']['S_POS']  + centroids_x[:,None,:].detach().numpy()
        state_input_renorm[:,:,1::2] = (state_input_renorm[:,:,1::2]) * self.neural_network['config']['S_VEL']

        return state_input_renorm

    def track_mcst_param_init(self):
        self.new_PT_set[-1].avbt_predictor_h = self.neural_network['model'].predictor.init_hidden(self.neural_network['mcst_args'].predictor_sampling_num)
        self.new_PT_set[-1].avbt_predictor_c = self.neural_network['model'].predictor.init_cell(self.neural_network['mcst_args'].predictor_sampling_num)
        self.new_PT_set[-1].avbt_sigma = torch.zeros([1, len(self.new_PT_set[-1].X), 4])

    def perform_mcst_predict(self, track):
        # 执行预测
        # 准备输入MCST的数据
        trajectory_len = len(track.X)

        detections_history = np.hstack(track.associate_plot_track)
        estimation_history = np.hstack(track.X)

        detections_history = torch.tensor(
            np.transpose(detections_history)).unsqueeze(
            dim=0).unsqueeze(dim=2).to(torch.float32)
        estimation_history = torch.tensor(
            np.transpose(estimation_history)).unsqueeze(
            dim=0).unsqueeze(dim=2).to(torch.float32)
        input_sigma = track.avbt_sigma.to(torch.float32)

        # 归一化
        normalized_update_history, normalized_detections, input_sigma = (
            self.mcst_normalize(estimation_history, detections_history, input_sigma,
                            self.neural_network['minMaxScaler'], self.neural_network['mcst_args'].predictor_time_series_len))

        normalized_update_history_MCU, normalized_detections_MCU, _ = (
            self.mcst_normalize(estimation_history, detections_history, input_sigma,
                            self.neural_network['minMaxScaler_MCU'], self.neural_network['mcst_args'].predictor_MCU_len))

        # 预测
        normalized_update_history = normalized_update_history.squeeze(dim=2)
        normalized_detections_MCU = normalized_detections_MCU.squeeze(dim=2)
        normalized_update_history_MCU = normalized_update_history_MCU.squeeze(dim=2)

        output_normalized_predict, output_predict_sigma, (track.avbt_predictor_h, track.avbt_predictor_c) = \
            self.neural_network['model'].predict(input_sigma, normalized_update_history, normalized_detections_MCU,
                                           normalized_update_history_MCU,
                                           (track.avbt_predictor_h, track.avbt_predictor_c))

        # 反归一化
        predict_output_data = self.neural_network['minMaxScaler'].deMinMaxScaler(
            output_normalized_predict.unsqueeze(dim=2)).squeeze(
            dim=2)

        X_predict = predict_output_data[0,:,:].transpose(-2,-1).detach().numpy()

        _, P_pred = self.tracker.KalmanPredict(np.zeros([4,1]), track.gmComponents[0].P, track.Q_coe)

        track.gmComponents[0].X = X_predict
        track.gmComponents[0].P = P_pred.copy()
        track.gmComponents[0].X_group_partition = np.expand_dims(X_predict, axis=-1)
        track.gmComponents[0].P_group_partition = np.expand_dims(P_pred, axis=-1)

        track.X.append(X_predict)
        track.P = P_pred
        track.X_predict = X_predict
        track.P_predict = P_pred
        az, elev, slantRange = enu2aer(X_predict[0], X_predict[2], 0)
        if az > 180:
            az = az - 360
        track.Polar_X = np.array([slantRange, az, elev]).T

        # 判断是否超出探测范围
        in_dist = (self.detection_distance_range[0] <= slantRange <=
                   self.detection_distance_range[1])
        in_azi = (self.detection_azimuth_range[0] <= az <=
                  self.detection_azimuth_range[1])
        in_elev = (self.detection_elevation_range[0] <= elev <=
                   self.detection_elevation_range[1])
        if in_dist and in_azi and in_elev:
            track.track_property = track.track_property
        else:
            track.track_property = 2  # 超出范围 不再继续跟踪

        track.avbt_sigma = torch.cat([track.avbt_sigma, output_predict_sigma.cpu()], dim=1)

        track.numberOfGmComponents = 1
        track.gmComponents = track.gmComponents[:1]
        track.gmWeight = track.gmWeight[:1]

    def perform_mcst_update(self, track, detection):
        # 执行更新
        # 准备输入MCST的数据
        trajectory_len = len(track.X)

        detections_history = np.hstack([np.hstack(track.associate_plot_track), detection])
        estimation_history = np.hstack(track.X)

        detections_history = torch.tensor(
            np.transpose(detections_history)).unsqueeze(
            dim=0).unsqueeze(dim=2).to(torch.float32)
        estimation_history = torch.tensor(
            np.transpose(estimation_history)).unsqueeze(
            dim=0).unsqueeze(dim=2).to(torch.float32)
        input_sigma = track.avbt_sigma.unsqueeze(1).to(torch.float32)

        # 归一化
        normalized_update_history, normalized_detections, _ = (
            self.mcst_normalize(estimation_history, detections_history, input_sigma,
                            self.neural_network['minMaxScaler'], self.neural_network['mcst_args'].predictor_time_series_len))

        input_sigma = track.avbt_sigma[:, -1, :].unsqueeze(1).to(torch.float32)
        # 更新
        normalized_detections = normalized_detections.squeeze(dim=2)
        normalized_update_history = normalized_update_history.squeeze(dim=2)

        output_normalized_update, output_update_sigma = \
            self.neural_network['model'].update(normalized_update_history, normalized_detections, input_sigma)

        # 反归一化
        update_output_data = self.neural_network['minMaxScaler'].deMinMaxScaler(
            output_normalized_update.unsqueeze(dim=2)).squeeze(
            dim=2)

        update_output_data = update_output_data[0,:,:].cpu().detach().numpy().reshape(-1, 1)

        return update_output_data


    def mcst_normalize(self, estimation_history, detections_history, input_sigma, minMaxScaler: Callable, normalize_len:int):
        # 函数功能： 归一化
        update_history = estimation_history.clone()
        detections = detections_history.clone()
        trajectory_len = update_history.shape[1]

        def _pad_sequence(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
            """对长度不足的序列补零"""
            pad_len = target_len - tensor.shape[1]
            if pad_len > 0:
                pad_shape = [tensor.shape[0], pad_len] + list(tensor.shape[2:])
                padding = torch.zeros(pad_shape, device=tensor.device)
                tensor = torch.cat([padding, tensor], dim=1)
            return tensor

        # 对LSTM 输入归一化
        begin_index = (trajectory_len - normalize_len) if (
                 trajectory_len - normalize_len) > 0 else 0
        input_sigma = input_sigma[:, begin_index:, :]
        (normalized_detections, normalized_update_history,
         min_vals, max_vals) = \
            minMaxScaler(detections[:, begin_index:, :, :],
                         update_history[:, begin_index:, :], self.T, (self.neural_network['mcst_args'].max_velocity,
                                                                      self.neural_network['mcst_args'].max_acceleration),
                         mode="-1_1")

        # 对不足长度序列进行补零
        normalized_detections = _pad_sequence(normalized_detections, normalize_len)
        normalized_update_history = _pad_sequence(normalized_update_history, normalize_len)
        input_sigma = _pad_sequence(input_sigma, normalize_len)

        # 移除冗余维度
        normalized_detections = normalized_detections.squeeze(dim=2)

        return normalized_update_history, normalized_detections, \
            input_sigma
