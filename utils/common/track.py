import copy

import numpy as np
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# 辅助数据结构 (用于模拟 MATLAB 的结构体/对象)
# ----------------------------------------------------------------------

@dataclass
class GmComponent:
    """高斯混合分量结构体 (用于 KF_GM/BGM)。"""
    X: np.ndarray = field(default_factory=lambda: np.array([]).reshape(-1, 1))
    P: np.ndarray = field(default_factory=lambda: np.array([]))
    hype_param: dict = field(default_factory=dict)
    # FSMM-specific
    X_model: np.ndarray = field(default_factory=lambda: np.array([]))
    P_model: np.ndarray = field(default_factory=lambda: np.array([]))
    # BGM/VL-specific
    X_group_partition: np.ndarray = field(default_factory=lambda: np.array([]).reshape(-1, 1))
    P_group_partition: np.ndarray = field(default_factory=lambda: np.array([]))
    Pd: dict = field(default_factory=dict)
    # BP_VGTT-specific
    X_last_moment: np.ndarray = field(default_factory=lambda: np.array([]).reshape(-1, 1))
    P_last_moment: np.ndarray = field(default_factory=lambda: np.array([]))
    gmWeight_last_moment: float = 0.0
    associate_plot_track: np.ndarray = field(default_factory=lambda: np.array([]).reshape(-1, 1))



@dataclass
class BetaDistributionParams:
    """Beta 分布参数 (用于 BGM 检测概率 Pd)。"""
    s: float = 0.0
    t: float = 0.0
    k: float = 0.0
    Pd: float = 0.0


@dataclass
class VLParams:
    """虚拟领导者 (VL) 模型参数 (用于 BP_VGTT)。"""
    mu_alpha0: float = 0.0
    tau_alpha0: float = 0.0
    mu_alpha_q: float = 0.0
    tau_alpha_q: float = 0.0
    mu_beta0: float = 0.0
    tau_beta0: float = 0.0
    mu_beta_q: float = 0.0
    tau_beta_q: float = 0.0
    mu_gamma0: float = 0.0
    tau_gamma0: float = 0.0
    mu_gamma_q: float = 0.0
    tau_gamma_q: float = 0.0
    alpha_set: list = field(default_factory=list)
    beta_set: list = field(default_factory=list)
    gamma_set: list = field(default_factory=list)
    tau_alpha_set: list = field(default_factory=list)
    tau_beta_set: list = field(default_factory=list)
    tau_gamma_set: list = field(default_factory=list)
    gmComponents: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Track 类
# ----------------------------------------------------------------------

class Track:
    """
    航迹类，用于存储和管理多目标跟踪中的单个航迹状态。
    """

    def __init__(self, X=None, P=None, R=None, doppler_vel=None, track_ID=None, zone_index=None, track_quality=None,
                 tracker_method=None, tracker_model=None, hype_param=None, associate_method=None, r_init=None, P_scaler=None):

        # --- 属性初始化 ---
        self.X = []  # 状态向量
        self.X_predict = np.array([]).reshape(-1, 1)  # 预测状态向量
        self.P = np.array([])  # 误差协方差矩阵
        self.P_predict = np.array([])  # 预测误差协方差矩阵
        self.Polar_X = None  # 极坐标
        self.track_quality = None  # 航迹得分
        self.connection_status = 0  # 关联状态 0：未关联 1：关联成功
        self.track_index = None  # 批号
        self.zone_index = None
        self.track_property = 0  # 0：滤波 1：外推 2：终结
        self.doppler_vel = None
        self.R = None  # 观测噪声协方差
        self.hype_param = None  # 超参数 (MHT/FSMM等)

        # FSMM/MHT specific
        self.X_model = None  # 模型状态 (FSMM)
        self.P_model = None  # 模型协方差 (FSMM)
        self.X_hypo = None  # 假设状态 (MHT)
        self.P_hypo = None  # 假设协方差 (MHT)
        self.Polar_X_hypo = None
        self.hypo_num = 0  # 假设数量 (MHT)
        self.weight_log = None  # 对数权重

        # PF specific
        self.X_particle = None  # 粒子集合

        # Existence Probability (PT)
        self.r = 0.0  # 存在概率
        self.r_history = []  # 存在概率历史

        # GM/BGM specific
        self.numberOfGmComponents = 0  # 高斯混合的个数
        self.gmWeight = None  # 高斯分量权重
        self.gmComponents = []  # 高斯混合分量 (list of GmComponent)
        self.gmWeight_prior = None  # 高斯分量先验 (BP_VGTT)
        self.VL = None  # 虚拟领导者模型参数 (VLParams)
        self.Pd = None  # 检测概率 (BGM)

        if X is None:
            return

        # --- 航迹初始化逻辑 ---
        X = np.asarray(X, dtype=float)
        P = np.asarray(P, dtype=float)
        R = np.asarray(R, dtype=float)

        # 确保 X 是列向量 (Nx1)
        if X.ndim == 1 or X.shape[0] == 1:
            X = X.reshape(-1, 1)

        self.X.append(X)
        self.P = P
        self.X_predict = X.copy()
        self.P_predict = P.copy()
        self.track_quality = track_quality
        self.connection_status = 0
        self.zone_index = zone_index
        self.track_index = track_ID
        self.track_property = 0
        self.doppler_vel = doppler_vel
        self.R = R
        self.r = r_init
        self.r_history = [r_init]
        associate_plot_track = copy.deepcopy(X[[0,2],0].reshape(-1, 1))
        self.associate_plot_track = []
        self.associate_plot_track.append(associate_plot_track)
        self.predict_flag = False # 完成预测的标志位
        self.Q_coe = 1 # 过程噪声放大系数
        self.residual_set = [] # 历史残差
        self.P_scaler = P_scaler

        # --- ---
        self.mcst_predictor_h: any = field(default=None)
        self.mcst_predictor_c: any = field(default=None)
        self.mcst_sigma: any = field(default=None)

        # --- 根据关联方法和跟踪器初始化复杂结构 ---

        if associate_method == 'MHT':
            self.X_hypo = [X.copy()]
            self.P_hypo = [P.copy()]
            self.hypo_num = 1
            self.hype_param = [hype_param]  # MATLAB 存储在 cell array 中

            if tracker_model == 'FSMM':
                self._initialize_fsmm_model(X, P, hype_param, is_mht=True)

        else:  # 非 MHT (JIPDA, GNN, BGM等)
            self.hype_param = hype_param

            if tracker_method in ['KF', 'EKF']:
                if tracker_model == 'FSMM':
                    self._initialize_fsmm_model(X, P, hype_param)

            elif tracker_method == 'PF':
                if tracker_model == 'FSMM':
                    self._initialize_fsmm_model(X, P, hype_param)
                    self._initialize_pf_particles(X, P, hype_param, fsmm=True)
                else:
                    self._initialize_pf_particles(X, P, hype_param, fsmm=False)

            elif tracker_method == 'KF_GM':
                self.gmWeight = np.array([1.0])
                self.numberOfGmComponents = 1
                associate_plot_track = copy.deepcopy(X[[0,2],0].reshape(-1,1))
                comp = GmComponent(X=X.copy(), P=P.copy(), associate_plot_track= associate_plot_track, hype_param=hype_param)
                if tracker_model == 'FSMM':
                    self._initialize_fsmm_model_in_component(comp, X, P, hype_param)

                self.gmComponents = [comp]

            elif tracker_method == 'BGM':

                self.gmWeight = np.array([1.0])
                self.numberOfGmComponents = 1
                associate_plot_track = copy.deepcopy(X[[0,2],0].reshape(-1,1))
                comp = GmComponent(X=X.copy(), P=P.copy(), associate_plot_track= associate_plot_track, hype_param=hype_param)

                if tracker_model == 'VL':
                    if associate_method == 'BP_R_RGTT':  # Beta-Gauss-Mixture, R-RGTT
                        self._initialize_beta_params(comp, s=10, t=1, k=1.2)
                        self.gmComponents = [comp]

                    elif associate_method == 'BP_VGTT':  # Beta-Gauss-Mixture, VGTT
                        self._initialize_vl_params(hype_param)
                        self._initialize_beta_params(comp, s=10, t=1, k=1.2)

                        comp.X_group_partition = X.copy()
                        comp.P_group_partition = P.copy()
                        comp.X_last_moment = X.copy()
                        comp.P_last_moment = P.copy()
                        comp.gmWeight_last_moment = 1.0
                        associate_plot_track = copy.deepcopy(X[[0, 2], 0].reshape(-1, 1))
                        comp.associate_plot_track = associate_plot_track

                        self.gmWeight_prior = np.array([1.0])
                        self.gmComponents = [comp]

                elif tracker_model == 'CV':  # Constant Velocity (CV)
                    if associate_method == 'BP_R_MTT':
                        self._initialize_beta_params(comp, s=1, t=1, k=1.2)
                        comp.X_group_partition = X.copy()
                        comp.P_group_partition = P.copy()
                        self.gmComponents = [comp]

                # BGM 追踪器最后设置 Pd
                if self.gmComponents and self.gmComponents[0].Pd:
                    self.Pd = self.gmComponents[0].Pd['Pd']

    # --- 辅助初始化方法 ---

    def _initialize_fsmm_model(self, X, P, hype_param, is_mht=False):
        """初始化 FSMM (有限状态模型) 的 X_model 和 P_model。"""
        model_num = hype_param.get('model_num', 1)
        X_model_list = []
        P_model_list = []
        for num in range(model_num):
            X_model_list.append(X.copy())
            # P_model 存储为块对角形式，每个块是 P
            P_model_list.append(P.copy())

        # X_model: (X_dim x model_num)
        self.X_model = np.hstack(X_model_list)
        # P_model: (P_dim x P_dim*model_num) (MATLAB 存储方式)
        self.P_model = np.hstack(P_model_list)

        if is_mht:
            # 在 MHT 中，X_model 和 P_model 存储在 cell 1 中
            self.X_model = [self.X_model]
            self.P_model = [self.P_model]

    def _initialize_fsmm_model_in_component(self, comp, X, P, hype_param):
        """在 GmComponent 中初始化 FSMM 模型。"""
        model_num = hype_param.get('model_num', 1)
        X_model_list = []
        P_model_list = []
        for num in range(model_num):
            X_model_list.append(X.copy())
            P_model_list.append(P.copy())

        comp.X_model = np.hstack(X_model_list)
        comp.P_model = np.hstack(P_model_list)

    def _initialize_pf_particles(self, X, P, hype_param, fsmm=False):
        """初始化粒子集合 (Particle Filter)。"""
        # 假设 X 是 (4x1) 向量
        particle_num = hype_param.get('particle_num', 100)
        X_dim = X.shape[0]

        # MATLAB 的 mvnrnd(X, P, N)' 等价于 np.random.multivariate_normal(X.flatten(), P, N).T
        try:
            # 使用 Cholesky 分解生成更可靠
            L = np.linalg.cholesky(P)
            # X + L @ (随机正态分布向量)
            self.X_particle = X + L @ np.random.randn(X_dim, particle_num)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵 P 病态或奇异，使用简单的对角线噪声
            self.X_particle = X + np.diag(P) * np.random.randn(X_dim, particle_num)

        if fsmm:
            model_num = hype_param.get('model_num', 1)
            # 初始化粒子的模型选择 (1, 2, ...)
            hype_param['X_particle_model_idx'] = np.random.randint(1, model_num + 1, particle_num)

    def _initialize_beta_params(self, comp, s, t, k):
        """初始化 Beta 分布参数 (用于 Pd)。"""
        Pd_dict = {
            's': s,
            't': t,
            'k': k,
            'Pd': s / (s + t)
        }
        comp.Pd = Pd_dict

    def _initialize_vl_params(self, hype_param):
        """初始化虚拟领导者模型 (VL) 参数 (用于 BP_VGTT)。"""
        self.VL = VLParams(
            mu_alpha0=hype_param.get('mu_alpha0', 0.0),
            tau_alpha0=hype_param.get('tau_alpha0', 0.0),
            mu_alpha_q=hype_param.get('mu_alpha_q', 0.0),
            tau_alpha_q=hype_param.get('tau_alpha_q', 0.0),
            mu_beta0=hype_param.get('mu_beta0', 0.0),
            tau_beta0=hype_param.get('tau_beta0', 0.0),
            mu_beta_q=hype_param.get('mu_beta_q', 0.0),
            tau_beta_q=hype_param.get('tau_beta_q', 0.0),
            mu_gamma0=hype_param.get('mu_gamma0', 0.0),
            tau_gamma0=hype_param.get('tau_gamma0', 0.0),
            mu_gamma_q=hype_param.get('mu_gamma_q', 0.0),
            tau_gamma_q=hype_param.get('tau_gamma_q', 0.0)
        )

    # --- 转换 MATLAB 的 method1 占位符 ---
    def method1(self, inputArg):
        """
        METHOD1 此处显示有关此方法的摘要
        """
        # 由于 Property1 未定义，我们假设它是一个实例属性，并在此处使用占位符
        if hasattr(self, 'Property1'):
            return self.Property1 + inputArg
        else:
            # 假设 Property1 应该是一个数值，如果不存在，则返回 inputArg
            return inputArg

# # 示例用法：初始化一个 BGM/VL/BP_R_RGTT 轨迹
# hype_param_example = {'model_num': 1, 'particle_num': 100}
# X_init = np.array([10, 5, 0.5, 0.1]).reshape(-1, 1) # x, y, vx, vy
# P_init = np.diag([1, 1, 0.1, 0.1])
# R_init = np.diag([0.5, 0.5])
# r_init = 0.8

# track = Track(X=X_init, P=P_init, R=R_init, doppler_vel=1.0, track_index=1,
#               zone_index=1, track_quality=0.5, tracker_method='BGM',
#               tracker_model='VL', hype_param=hype_param_example,
#               associate_method='BP_R_RGTT', r_init=r_init)

# print(f"初始化方法: {track.tracker_method}")
# print(f"存在概率 r: {track.r}")
# print(f"GM 数量: {track.numberOfGmComponents}")
# print(f"Pd: {track.Pd}")