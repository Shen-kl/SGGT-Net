import numpy as np
from utils.common.coordinate_transformation import *
# ----------------------------------------------------------------------
# Plot_Track 类
# ----------------------------------------------------------------------

class Plot_Track:
    """
    表示雷达中的一个点迹 (Measurement/Detection)。
    """
    def __init__(self, X, plot_track_index):
        """
        PLOT_TRACK 构造此类的实例。

        参数:
            X (np.ndarray or list): 原始状态向量 (可以是 1xN 或 Nx1)。
            plot_track_index (int): 点迹的唯一标号。
        """
        # 1. 属性定义 (Properties)
        self.X = None  # 状态向量 (仅包含位置，e.g., [x; y])
        self.Polar_X = None  # 极坐标状态向量 [斜距; 方位角(度)]
        self.connection_status = 0  # 关联状态 0：未关联 1：关联
        self.zone_index = None  # 分区区号
        self.plot_track_index = plot_track_index  # 点迹标号
        self.inDoor = 0  # 落入相关波们标志位
        self.doppler_vel = None  # 多普勒速度
        self.R = None  # (可能用于存储观测噪声协方差，但未在 MATLAB 构造函数中初始化)
        self.group_score = None  # 配合MP_MM, 存放所有可能分组的伪似然得分
        self.candidate_groups = []  # 候选群组

        # 2. 构造函数逻辑 (Methods)

        # 确保 X 是列向量形式 (Nx1)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1 or X.shape[0] == 1:
            X = X.reshape(-1, 1)  # 转换为列向量

        # 存储前两个分量 (位置，假设是 ENU 的 E 和 N)
        self.X = X[0:2]

        # 假设状态向量 X 的维度至少为 3，X[0]=E, X[1]=N, X[2]=多普勒速度
        E = X[0, 0]
        N = X[1, 0]
        U = 0.0  # MATLAB 假设 U=0 (平面目标)

        az, elev, slantRange = enu2aer(E, N, U)

        # MATLAB 逻辑：如果 az > 180，则 az = az - 360 (将 [0, 360] 映射到 [-180, 180])
        if az > 180:
            az = az - 360

        self.Polar_X = np.array([slantRange, az]).reshape(-1, 1)  # [斜距; 方位角]

        self.connection_status = 0
        self.inDoor = 0

        # 假设 X[2] 是多普勒速度
        if len(X) > 2:
            self.doppler_vel = X[2, 0]
        else:
            self.doppler_vel = None

        self.candidate_groups = []  # 初始化为空列表

# # 示例用法 (假设输入 X = [100, 200, 5])
# X_input = np.array([100, 200, 5])
# pt = Plot_Track(X_input, 1)
# print(f"状态向量 X (位置): \n{pt.X}")
# print(f"极坐标 Polar_X [R; Az] (deg): \n{pt.Polar_X}")
# print(f"多普勒速度: {pt.doppler_vel}")