import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Set
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data, Dataset
import torch
import traceback

@dataclass
class GraphInputData:
    """
    封装 GNN 输入数据 (与之前保持接口一致)
    """
    node_features: np.ndarray        # (N, T, D)
    node_masks: np.ndarray           # (N, T)
    meas_features: np.ndarray        # (N, T, D1)
    edge_indices_list: List[np.ndarray] # T 个时刻的边索引
    edge_attrs_list: List[np.ndarray]   # T 个时刻的边特征
    active_track_ids: List[int]      # 当前时刻 N 行对应的 ID (用于调试或输出映射)
    residual_features: np.ndarray  # (N, T1, D) 残差特征
    P_scaler: torch.tensor  # (N, n_mixtures, D, D)

class GraphInputBuilder:
    def __init__(self,
                 window_size: int = 5,
                 feature_dim: int = 4,
                 meas_dim: int = 2,
                 max_targets: int  = 80,
                 group_connect_threshold: float = 200.0,
                 residual_window_size: int = 32,
                 n_mixtures: int = 1):

        self.T = window_size
        self.feature_dim = feature_dim
        self.meas_dim = meas_dim
        self.N = max_targets
        self.threshold = group_connect_threshold
        self.residual_T = residual_window_size
        self.n_mixtures = n_mixtures
        # --- 状态缓冲区 (Persistent Buffers) ---
        # 初始化为全 0
        self.node_buffer = np.zeros((self.N, self.T, self.feature_dim), dtype=np.float32)
        self.mask_buffer = np.zeros((self.N, self.T, 1), dtype=bool)
        self.meas_buffer = np.zeros((self.N, self.T, self.meas_dim), dtype=np.float32)
        self.residual_buffer = np.zeros((self.N, self.residual_T, self.meas_dim), dtype=np.float32)
        self.node_error_cov_matrix_scaler = torch.zeros((self.N, self.n_mixtures, self.feature_dim, self.feature_dim), dtype=torch.float32)

        # --- 边历史队列 ---
        # 使用 deque 自动处理滑窗，maxlen=T 会自动挤出旧元素
        # 初始化时填充空数据，保证刚开始运行输出长度也为 T
        self.edge_indices_queue = deque([np.empty((0, 2), dtype=int) for _ in range(self.T)], maxlen=self.T)
        self.edge_attrs_queue = deque([np.empty((0, 1), dtype=np.float32) for _ in range(self.T)], maxlen=self.T)

        # --- 行/槽位管理 (Row Management) ---
        # 记录 track_id -> row_index 的映射
        self.track_id_to_row: Dict[int, int] = {}
        # 记录哪些行是空闲的，可以使用
        self.free_rows: List[int] = list(range(self.N-1,-1,-1))
        # 使用堆或排序列表优化 free_rows 可以进一步提升，但对于小 N (如20) 列表足够
        # 记录每一时刻存在的目标的ID
        self.track_id_survive_history = [set() for _ in range(self.T)]

        # 记录执行过的次数
        self.execute_cnt = 0
    def update(self, radar, duration_threshold=2) -> GraphInputData:
        """
        每帧调用一次， 更新状态并返回当前窗口的 Tensor
        :param radar:
        :return:
        """
        # 0. 执行次数加1
        self.execute_cnt += 1

        # 1. 滚动缓冲区
        self.node_buffer[:, :-1, :] = self.node_buffer[:, 1:, :]
        self.meas_buffer[:, :-1, :] = self.meas_buffer[:, 1:, :]
        self.mask_buffer[:, :-1, :] = self.mask_buffer[:, 1:, :]
        self.residual_buffer[:, :-1, :] = self.residual_buffer[:, 1:, :]
        self.node_buffer[:, :-1, :] = self.node_buffer[:, 1:, :]
        self.node_buffer[:, :-1, :] = self.node_buffer[:, 1:, :]

        # 2. 清空最新时刻的数据 (Mask 置为 False)
        # 这一步很重要，防止上一帧在此行的数据残留（如果该行对应的目标这一帧消失了）
        self.node_buffer[:, -1, :] = 0
        self.meas_buffer[:, -1, :] = 0
        self.mask_buffer[:, -1, :] = False
        self.residual_buffer[:, -1, :] = 0

        # 3. 目标管理与槽位分配
        current_tracks = radar.legacy_PT_set

        # 提取当前所有存活目标的 ID 集合 并更新历史
        current_track_ids = set(t.track_index for t in current_tracks)
        self.track_id_survive_history[:-1] = self.track_id_survive_history[1:]
        self.track_id_survive_history[-1] = current_track_ids
        # [清理]：检查已消失的目标，释放其行索引
        # 需要复制 keys() 因为我们在迭代中修改字典
        for tid in list(self.track_id_to_row.keys()):
            flag = False
            for t in range(self.T):
                if tid in self.track_id_survive_history[t]:
                    flag = True
                    break
            if not flag:
                # 说明该ID 在当前的窗口内都不存在了 记录该ID 对应的行号 并回收
                row_index = self.track_id_to_row.pop(tid)
                try:
                    self.free_rows.append(row_index)
                except AttributeError as e:
                    print(f"Error: {e}")
                    print(f"Object type: {type(self)}")
                    print(f"Object: {self}")
                    traceback.print_stack()
                    raise

        active_rows_this_frame = []  # 记录当前帧哪些行被激活了 (用于算边)

        # 处理当前帧的目标
        for track in current_tracks:
            tid = track.track_index

            # 确定行索引
            if tid in self.track_id_to_row:
                row_index = self.track_id_to_row[tid]
            else:
                if not self.free_rows:
                    # 没有空闲行了 (超过 Max N)，忽略此目标
                    continue
                # 如果航迹为确认航迹 分配新行
                if len(track.X) > 0:
                    row_index = self.free_rows.pop()
                    self.track_id_to_row[tid] = row_index

                    # 注意：新分配的行，其 buffer 里的历史数据虽然还是旧的（如果不清空），
                    # 但因为之前的帧 Mask 都是 False (或者我们可以显式清空整行)，所以是安全的。
                    # 更严谨的做法是：分配新行时，清空该行的历史 buffer
                    self.node_buffer[row_index, :, :] = 0
                    self.meas_buffer[row_index, :, :] = 0
                    self.mask_buffer[row_index, :, :] = False
                    self.residual_buffer[row_index, :, :] = 0
                    self.node_error_cov_matrix_scaler[row_index, :, :, :] = 0
            if len(track.X) > 0:
                curr_state = np.array(track.X[-1]).flatten()
                curr_meas = np.array(track.associate_plot_track[-1]).flatten()

                # 填充 Buffer 的最后一位 [-1]
                self.node_buffer[row_index, -1, :] = curr_state
                self.meas_buffer[row_index, -1, :] = curr_meas
                self.mask_buffer[row_index, -1, :] = True
                self.residual_buffer[row_index, -1, :] = curr_meas - curr_state[::2]
                self.node_error_cov_matrix_scaler[row_index, :, :, :] = track.P_scaler
                # 使用量测作为位置信息的输入
                ################################
                # self.node_buffer[row_index, -1, ::2] = curr_meas
                ################################
                active_rows_this_frame.append(row_index)

        # 4. 计算当前帧的图结构 (Edges)
        # 只计算 active_rows 之间的边
        survival_duration = [len(track.X) for track in radar.legacy_PT_set]
        if len(survival_duration) != 0:
            survival_duration = np.hstack(survival_duration)
        else:
            survival_duration = None
        current_edges, current_dists, current_vels = self._compute_edges(active_rows_this_frame, survival_duration,
                                                                         duration_threshold)

        # 推入队列 (自动挤出最老的)
        self.edge_indices_queue.append(current_edges)
        self.edge_attrs_queue.append(np.hstack([current_dists,current_vels]))

        # 5. 构造返回值
        # 需要将内部 buffer 复制一份或者是只读视图，防止外部修改影响内部状态
        # 为了安全，通常 copy；为了极速，返回 view。这里返回 copy 较安全。

        # 构造 track_id_mapping 列表，索引对应矩阵行号
        # map: row -> tid
        row_to_tid = {v: k for k, v in self.track_id_to_row.items()}
        id_list = [row_to_tid.get(i, -1) for i in range(self.N)]

        return GraphInputData(
            node_features=self.node_buffer.copy(),
            node_masks=self.mask_buffer.copy(),
            meas_features=self.meas_buffer.copy(),
            edge_indices_list=list(self.edge_indices_queue),
            edge_attrs_list=list(self.edge_attrs_queue),
            active_track_ids=id_list,
            residual_features = self.residual_buffer.copy(),
            P_scaler = self.node_error_cov_matrix_scaler
        )

    def _compute_edges(self, active_rows, survival_duration=None, duration_threshold=2):
        """
        仅计算当前帧激活节点之间的边
        """
        if len(active_rows) < 2:
            return np.empty((0, 2), dtype=int), np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

        # 提取当前帧有效节点的位置
        # shape: (Num_Active, 2)
        # 注意：这里取出的坐标是按照 active_rows 顺序的，
        # active_rows 里的值是绝对行号 (0...N-1)
        positions = self.node_buffer[active_rows, -1, ::2]
        velocity = self.node_buffer[active_rows, -1, 1::2]
        # 计算位置与速度的距离
        dist_vec = pdist(positions, metric='euclidean')
        dist_matrix = squareform(dist_vec)

        vel_vec = pdist(velocity, metric='euclidean')
        vel_matrix = squareform(vel_vec)

        # 邻接判断
        adj_matrix = (dist_matrix < self.threshold) & (dist_matrix >= 0)

        if survival_duration is not None:
            mask = (survival_duration >= duration_threshold)  # True 表示合格
            # 合格目标之间才保留边；不合格目标整行整列置 0 除了对角线上的元素
            adj_matrix = adj_matrix * mask[:, None] * mask[None, :]
        # 对角线(即自己跟自己)
        diagonal_indices = np.diag_indices(adj_matrix.shape[0])
        adj_matrix[diagonal_indices] = True

        # 获取相对索引 (0 ... Num_Active-1)
        rel_src, rel_dst = np.where(adj_matrix)

        if len(rel_src) > 0:
            # 将相对索引映射回 Buffer 的绝对行索引 (0 ... N-1)
            # active_rows 是一个 list 或 array，可以直接做 indexing
            active_rows_arr = np.array(active_rows)
            abs_src = active_rows_arr[rel_src]
            abs_dst = active_rows_arr[rel_dst]

            edges = np.stack([abs_src, abs_dst], axis=1)
            distances = dist_matrix[rel_src, rel_dst].reshape(-1, 1)
            velocitys = vel_matrix[rel_src, rel_dst].reshape(-1, 1)
            return edges, distances, velocitys
        else:
            return (np.empty((0, 2), dtype=int), np.empty((0, 1), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32))

    def get(self):
        return Data(
            x=torch.from_numpy(self.node_buffer.copy()),
            nan_mask=torch.from_numpy(self.mask_buffer),
            measurement=torch.from_numpy(self.meas_buffer.copy()),
            edge_index=[torch.from_numpy(t).T for t in self.edge_indices_queue],
            edge_features=[torch.from_numpy(t) for t in self.edge_attrs_queue],
            residual=torch.from_numpy(self.residual_buffer.copy()),
            P_scaler = self.node_error_cov_matrix_scaler
        )
# ==========================================
# 模拟测试代码
# ==========================================
if __name__ == "__main__":
    # 简单的 Mock 类
    class MockTrack:
        def __init__(self, idx, pos):
            self.track_index = idx
            self.r = 0.9
            # X 列表模拟历史，[-1] 是当前
            self.X = [np.array([pos[0], 0, pos[1], 0])]
            self.associate_plot_track = [np.array([pos[0],pos[1]])]

    class MockRadar:
        def __init__(self):
            self.legacy_PT_set = []

    # 初始化构建器
    builder = GraphInputBuilder(window_size=3, max_targets=5, group_connect_threshold=10.0, residual_window_size=32)
    radar = MockRadar()

    print("--- Frame 1: T1(0,0), T2(2,2) ---")
    radar.legacy_PT_set = [MockTrack(1, [1, 1]), MockTrack(2, [2, 2])]
    out1 = builder.update(radar)
    # T1 -> Row 0, T2 -> Row 1. 距离 sqrt(8) < 10, 有边.
    # 历史: [Emp, Emp, Data]
    print(f"Masks (-1):\n{out1.node_masks[:, -1]}") # 应有两个 True
    print(f"Edges (-1):\n{out1.edge_indices_list[-1]}") # 应有连接 (0,1), (1,0)

    print("\n--- Frame 2: T1(0,0), T2(2,2), T3(20,20) ---")
    # T1, T2 位置不变，新增 T3 较远
    radar.legacy_PT_set = [MockTrack(1, [1, 1]), MockTrack(2, [2, 2]), MockTrack(3, [20, 20])]
    out2 = builder.update(radar)
    # T1 -> Row 0, T2 -> Row 1, T3 -> Row 2
    # 历史: [Emp, Data, Data]
    # T3 刚出现，前几帧 Mask 应为 False (Buffer 初始化是0)
    print(f"Row 2 (T3) Mask History: {out2.node_masks[2, :]}") # [False, False, True]
    print(f"Edges (-1):\n{out2.edge_indices_list[-1]}") # 只有 0-1 连接，2 孤立

    print("\n--- Frame 3: T1 消失, T2(2,2), T3(20,20) ---")
    radar.legacy_PT_set = [MockTrack(2, [2, 2]), MockTrack(3, [20, 20])]
    out3 = builder.update(radar)
    # T1 (Row 0) 被释放. Mask 变 False.
    # T2 仍在 Row 1, T3 仍在 Row 2.
    print(f"Row 0 (Old T1) Mask History: {out3.node_masks[0, :]}") # [True, True, False] -> 历史仍保留，当前帧无效
    print(f"Active IDs: {out3.active_track_ids}") # Row 0 是 -1 (无效)