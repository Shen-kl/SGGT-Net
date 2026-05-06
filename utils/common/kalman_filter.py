import numpy as np

class KalmanFilter:
    """
    卡尔曼滤波器类 (KalmanFilter)。

    实现了标准的卡尔曼滤波预测和更新，以及有限状态马尔可夫模型 (FSMM)
    和虚拟领导者 (VL) 相关的群跟踪方法。
    """

    def __init__(self, F, H, Q, R, type_kf, hype_param):
        """
        构造函数。

        参数:
            F (np.ndarray): 状态转移矩阵。
            H (np.ndarray): 观测矩阵。
            Q (np.ndarray): 过程噪声协方差矩阵。
            R (np.ndarray): 量测噪声协方差矩阵。
            type_kf (str): 滤波器类型 (e.g., 'KF', 'EKF', 'FSMM')。
            hype_param (dict): 模型的超参数。
        """
        # 确保输入是 NumPy 数组
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.type = type_kf
        self.hype_param = hype_param

    def KalmanPredict(self, X, P, Q_coe):
        """
        函数功能： 标准卡尔曼预测。

        参数:
            X (np.ndarray): 状态向量 (N x 1)。
            P (np.ndarray): 误差协方差矩阵 (N x N)。

        返回:
            X_predict (np.ndarray): 预测状态向量。
            P_predict (np.ndarray): 预测误差协方差矩阵。
        """
        # X_predict = F * X;
        X_predict = self.F @ X
        # P_predict = F * P * F' + Q;
        P_predict = self.F @ P @ self.F.T + Q_coe * self.Q
        return X_predict, P_predict

    def KalmanUpdate(self, X, P, Z, R_obs):
        """
        函数功能： 标准卡尔曼滤波更新。

        参数:
            X (np.ndarray): 预测状态向量 (N x 1)。
            P (np.ndarray): 预测误差协方差矩阵 (N x N)。
            Z (np.ndarray): 当前量测向量 (M x 1)。
            R_obs (np.ndarray): 当前量测噪声协方差矩阵 (M x M)。

        返回:
            X_update (np.ndarray): 更新状态向量。
            P_update (np.ndarray): 更新误差协方差矩阵。
        """
        # Z_pre = H * X;
        Z_pre = self.H @ X

        # S = H * P * H' + R
        S = self.H @ P @ self.H.T + R_obs

        # K = P * H' / S (Python中使用 @ 和 np.linalg.inv)
        K = P @ self.H.T @ np.linalg.inv(S)

        # X_update = X + K * (Z - Z_pre);
        X_update = X + K @ (Z - Z_pre)

        # P_update = (I - K * H) * P * (I - K * H)' + K * R * K'; (Joseph 形式，更稳定)
        I = np.eye(X.shape[0])
        P_update = (I - K @ self.H) @ P @ (I - K @ self.H).T + K @ R_obs @ K.T

        return X_update, P_update

    def FSMMPredict(self, X_model, P_model, hype_param):
        """
        函数功能： 有限状态马尔可夫模型 (FSMM) 预测。

        注意：假设状态维度 N=4。P_model 的维度为 (4 x 4*model_num)。

        参数:
            X_model (np.ndarray): 多个模型的状态向量 (N x model_num)。
            P_model (np.ndarray): 多个模型的协方差矩阵 (N x N*model_num)，水平堆叠。
            hype_param (dict): 超参数字典 (包含 model_num, P_markov, u_model, T, accelerate_baisc_set)。

        返回:
            X_predict (np.ndarray): 混合状态预测向量。
            P_predict (np.ndarray): 混合误差协方差矩阵。
            X_model (np.ndarray): 更新后的模型状态。
            P_model (np.ndarray): 更新后的模型协方差。
            hype_param (dict): 更新后的超参数 (包含 C_hat)。
        """
        N = X_model.shape[0]  # 状态维度，假设 N=4
        model_num = hype_param['model_num']

        X_model_lastTime = X_model.copy()
        P_model_lastTime = P_model.copy()
        u_model_lastTime = hype_param['u_model']  # 1 x model_num
        P_markov = hype_param['P_markov']  # model_num x model_num (P(j->i))
        T = hype_param['T']

        C_hat = np.zeros(model_num)

        for i in range(model_num):  # i 是目标模型 (0-based)
            i_start = i * N
            i_end = (i + 1) * N

            # 1. 状态估计交互 (Interaction)
            X_model_i = np.zeros((N, 1))
            P_model_i = np.zeros((N, N))

            # C_hat(i) = sum_j ( P_markov(j,i) * u_model_lastTime(j) )
            C_hat[i] = np.sum(P_markov[:, i] * u_model_lastTime)

            if C_hat[i] == 0:
                # 避免除以零，如果 C_hat 为 0，则该模型不可达，保持零状态和协方差
                X_model[:, i] = np.zeros(N)
                P_model[:, i_start:i_end] = np.zeros((N, N))
                continue

            for j in range(model_num):  # j 是前一时刻模型 (0-based)
                j_start = j * N
                j_end = (j + 1) * N

                # 计算交互权重 (P(j->i) * u(j) / C_hat(i))
                interaction_weight = P_markov[j, i] * u_model_lastTime[j] / C_hat[i]

                # X_model(:,i) = sum_j (X_model_lastTime(:,j) * weight)
                X_model_i += X_model_lastTime[:, j].reshape(-1, 1) * interaction_weight

                # P_model(:,i) = sum_j ((P_model_lastTime(:,j) + (X_j - X_i)(X_j - X_i)') * weight)
                X_j = X_model_lastTime[:, j].reshape(-1, 1)
                X_i = X_model_i  # 注意：这里使用累加中的当前 X_model_i

                P_model_lastTime_j = P_model_lastTime[:, j_start:j_end]

                # 在 MATLAB 代码中，X_model(:,i) 在内层循环之后才确定最终值
                # 因此，我们先累加 P 项，在最终确定 X_model_i 后再处理 P_model_i
                # **根据 MATLAB 逻辑，此处的 X_model_i 应该是最终确定的值**
                # 修正：将 X_model_i 赋值操作放在内层循环之后，以匹配 MATLAB 逻辑。

                # P_model_i 累加协方差项 (不包含 (X_j - X_i) 项，该项在 MATLAB 中似乎使用了已更新的 X_model(:,i))
                # 严格按照 MATLAB 逻辑，X_model_i 已经在 j 循环中被不断累加。
                # 协方差项 (X_j - X_i) 的计算依赖于完全累加后的 X_model_i。
                # 由于 MATLAB 允许这种动态累加，为了安全，我们先计算 X_model_i 的最终值。

            # 重新计算 X_model_i (确保精确匹配 MATLAB 的最终值)
            X_model_i = np.zeros((N, 1))
            for j in range(model_num):
                interaction_weight = P_markov[j, i] * u_model_lastTime[j] / C_hat[i]
                X_model_i += X_model_lastTime[:, j].reshape(-1, 1) * interaction_weight

            # 计算 P_model_i 的最终值
            P_model_i = np.zeros((N, N))
            for j in range(model_num):
                interaction_weight = P_markov[j, i] * u_model_lastTime[j] / C_hat[i]
                X_j = X_model_lastTime[:, j].reshape(-1, 1)

                # 协方差项：P_lastTime_j + (X_j - X_i)(X_j - X_i)'
                X_diff = X_j - X_model_i
                P_model_lastTime_j = P_model_lastTime[:, j * N:(j + 1) * N]
                P_model_i += (P_model_lastTime_j + X_diff @ X_diff.T) * interaction_weight

            # 更新 X_model 和 P_model 矩阵
            X_model[:, i] = X_model_i.flatten()
            P_model[:, i_start:i_end] = P_model_i

            # 2. 模型修正 (Model Correction)

            F_thisModel = self.F

            # accererate_vec_this_model: 假设 state 顺序是 [x, xdot, y, ydot] 或 [x, y, xdot, ydot]
            # MATLAB 假设 state 维度 N=4。accel_basic_set 是 2xmodel_num，[ax; ay]
            accel_basic_set = hype_param['accelerate_baisc_set']
            accel_x = accel_basic_set[0, i]
            accel_y = accel_basic_set[1, i]

            # 加速度输入向量：假设状态顺序是 [x, xdot, y, ydot]
            # MATLAB 的向量构造 [T^2/2 T]*ax ... [T^2/2 T]*ay]' 是 4x1
            # 假设状态顺序是 [x; x_dot; y; y_dot]
            # 输入向量为 [T^2/2 * ax; T * ax; T^2/2 * ay; T * ay]'
            # MATLAB 似乎使用了 [T^2/2; T] 在 x 和 y 轴上分别作用于 state 1, 2 和 state 3, 4。

            # 这是一个近似：[T^2/2 * ax; T * ax; T^2/2 * ay; T * ay]
            # MATLAB 代码中的向量似乎是 [ (T^2/2)*ax, T*ax, (T^2/2)*ay, T*ay ]^T
            # X_dim=4, 状态顺序: [x, vx, y, vy] (典型 4D CV 状态)
            # F_thisModel * X + input_vector
            accererate_vec_this_model = np.array([
                T ** 2 / 2 * accel_x,
                T * accel_x,
                T ** 2 / 2 * accel_y,
                T * accel_y
            ]).reshape(-1, 1)

            X_model_i = X_model[:, i].reshape(-1, 1)
            P_model_i = P_model[:, i_start:i_end]

            # X_model(:,i)=F_thisModel*X_model(:,i) + accererate_vec_this_model;%预测
            X_model[:, i] = (F_thisModel @ X_model_i + accererate_vec_this_model).flatten()

            # P_model(:,i)=F_thisModel*P_model(:,i)*F_thisModel'+Q+ accererate_vec_this_model * accererate_vec_this_model';%预测
            # 注意：这里的 accererate_vec_this_model * accererate_vec_this_model' 应该是额外的过程噪声或建模误差项
            P_model[:, i_start:i_end] = (
                    F_thisModel @ P_model_i @ F_thisModel.T +
                    self.Q +
                    accererate_vec_this_model @ accererate_vec_this_model.T
            )

        # 3. 计算混合预测 (Overall Prediction)

        # C_hat = C_hat / sum(C_hat) (规范化)
        C_hat /= np.sum(C_hat)

        # X_predict = sum(C_hat * X_model, 2);
        X_predict = np.sum(X_model * C_hat, axis=1, keepdims=True)

        # P_predict = sum_i (C_hat(i) * (P_model(:,i) + (X_pred - X_model(:,i)) * (X_pred - X_model(:,i))'))
        P_predict = np.zeros((N, N))
        for i in range(model_num):
            i_start = i * N
            i_end = (i + 1) * N

            P_i = P_model[:, i_start:i_end]
            X_i = X_model[:, i].reshape(-1, 1)

            X_diff = X_predict - X_i
            P_predict += C_hat[i] * (P_i + X_diff @ X_diff.T)

        hype_param['C_hat'] = C_hat

        return X_predict, P_predict, X_model, P_model, hype_param

    def FSMMUpdate(self, X_model, P_model, hype_param, Z, R_obs):
        """
        函数功能： 有限状态马尔可夫模型 (FSMM) 更新。

        注意：假设状态维度 N=4。P_model 的维度为 (4 x 4*model_num)。

        参数:
            X_model (np.ndarray): 预测的模型状态 (N x model_num)。
            P_model (np.ndarray): 预测的模型协方差 (N x N*model_num)。
            hype_param (dict): 超参数字典 (包含 model_num, C_hat)。
            Z (np.ndarray): 当前量测向量 (M x 1)。
            R_obs (np.ndarray): 当前量测噪声协方差矩阵 (M x M)。

        返回:
            X_update (np.ndarray): 混合状态更新向量。
            P_update (np.ndarray): 混合误差协方差矩阵。
            X_model (np.ndarray): 更新后的模型状态。
            P_model (np.ndarray): 更新后的模型协方差。
            hype_param (dict): 更新后的超参数 (包含 u_model)。
        """
        N = 4  # 状态维度
        model_num = hype_param['model_num']
        model_likelihood = np.zeros(model_num)

        for i in range(model_num):
            i_start = i * N
            i_end = (i + 1) * N

            P_i_pred = P_model[:, i_start:i_end]
            X_i_pred = X_model[:, i].reshape(-1, 1)

            # S = H * P * H' + R
            S = self.H @ P_i_pred @ self.H.T + R_obs

            # K = P * H' * inv(S)
            K = P_i_pred @ self.H.T @ np.linalg.inv(S)

            # v = Z - H * X_model(:,i)
            v = Z - self.H @ X_i_pred

            # if 1: 考虑检测概率 (MATLAB代码中的占位符，实际执行了更新)
            # X_model(:,i)=X_model(:,i)+ K * v;
            X_model[:, i] = (X_i_pred + K @ v).flatten()

            # P_model(:,i)=(I - K*H)*P_model(:,i)*(I - K*H)' + K*R*K';
            I = np.eye(N)
            P_model[:, i_start:i_end] = (I - K @ self.H) @ P_i_pred @ (I - K @ self.H).T + K @ R_obs @ K.T

            # 模型可能性计算
            # L = 1 / sqrt(det(2*pi*S)) * exp(-0.5 * v' * inv(S) * v );
            try:
                S_inv = np.linalg.inv(S)
                det_S = np.linalg.det(S)
                likelihood = (1 / np.sqrt(det_S * 2 * np.pi)) * np.exp(-0.5 * v.T @ S_inv @ v)
                model_likelihood[i] = likelihood[0, 0]  # 确保取值
            except np.linalg.LinAlgError:
                model_likelihood[i] = 0.0  # 矩阵不可逆或奇异时

        # 1. 模型可能性更新
        C_hat = hype_param['C_hat']
        # C = sum(model_likelihood * C_hat)
        C = np.sum(model_likelihood * C_hat)

        if C == 0:
            # 如果 C=0 (所有可能性为零)，则无法更新
            u_model = C_hat  # 保持先验
        else:
            # u_model = model_likelihood * C_hat / C
            u_model = model_likelihood * C_hat / C

        # 2. 计算更新后的混合输出

        # X_update = sum(X_model * u_model, 2)
        X_update = np.sum(X_model * u_model, axis=1, keepdims=True)

        # P_update = sum_i ((P_model(:,i) + (X_i - X_update)(X_i - X_update)') * u_model(i))
        P_update = np.zeros((N, N))
        for i in range(model_num):
            i_start = i * N
            i_end = (i + 1) * N

            P_i_update = P_model[:, i_start:i_end]
            X_i_update = X_model[:, i].reshape(-1, 1)

            X_diff = X_i_update - X_update
            P_update += u_model[i] * (P_i_update + X_diff @ X_diff.T)

        hype_param['u_model'] = u_model

        return X_update, P_update, X_model, P_model, hype_param

    def KalmanPredict_specifiedT(self, X, P, T):
        """
        函数功能： 针对 9D 状态向量 (x, dx, ddx, y, dy, ddy, z, dz, ddz) 的卡尔曼预测。

        参数:
            X (np.ndarray): 状态向量 (9 x 1)。
            P (np.ndarray): 误差协方差矩阵 (9 x 9)。
            T (float): 时间步长。如果 T < 0，执行逆时间预测。

        返回:
            X_predict (np.ndarray): 预测状态向量。
            P_predict (np.ndarray): 预测误差协方差矩阵。
        """
        N = X.shape[0]  # N=9

        # 构建 9x9 状态转移矩阵 F (CV/CA 模型，取决于 Q)
        if T >= 0:
            T_val = T
        else:
            T_val = abs(T)  # 即使逆向，时间步长 T 仍为正值

        # 构建 3D (x, vx, ax) 的 F 块
        F_block = np.array([
            [1, T_val, 0.5 * T_val ** 2],
            [0, 1, T_val],
            [0, 0, 1]
        ])

        # F 是 3x3 块的块对角矩阵
        F = np.zeros((N, N))
        F[0:3, 0:3] = F_block
        F[3:6, 3:6] = F_block
        F[6:9, 6:9] = F_block

        if T >= 0:
            # 正向预测: X_pred = F * X, P_pred = F * P * F' + Q
            X_predict = F @ X
            P_predict = F @ P @ F.T + self.Q
        else:
            # 逆向预测 (T < 0): X_pred = inv(F) * X, P_pred = inv(F) * P * inv(F)' + Q (注意 MATLAB 这里的 Q 应该是指 F'Q F 或是别的项)
            # 严格卡尔曼反向滤波通常是 P_pred = inv(F) * (P - Q) * inv(F)'
            # 但这里遵循 MATLAB 逻辑，使用 P_predict = inv(F) * P * inv(F)' + Q
            inv_F = np.linalg.inv(F)
            X_predict = inv_F @ X
            P_predict = inv_F @ P @ inv_F.T + self.Q

        return X_predict, P_predict

    def compute_repulsion(self, group_state):
        """
        函数功能： 计算斥力 (用于 VL 模型)。

        参数:
            group_state (np.ndarray): 群内所有目标的状态矩阵 (N x num_targets)。

        返回:
            repulsion (np.ndarray): 斥力向量 (2 x num_targets)。
        """
        num_targets = group_state.shape[1]
        repulsion = np.zeros((2, num_targets))

        R1 = self.hype_param['R1']
        R2 = self.hype_param['R2']

        # 假设状态顺序是 [x, ..., y, ...]，位置在第 0 和 2 维 (MATLAB 的 [1, 3] 索引)
        # 假设 group_state 是 N x num_targets，且 x=row 0, y=row 2

        for i in range(num_targets):
            # 排除当前目标 i
            group_state_n_i = np.delete(group_state, i, axis=1)

            # 提取当前目标和所有其他目标的 (x, y) 坐标
            pos_i = group_state[[0, 2], i].reshape(2, 1)
            pos_n_i = group_state_n_i[[0, 2], :]

            # d_xy = pos_i - pos_n_i (2 x (num_targets - 1))
            d_xy = pos_i - pos_n_i

            # d_norm = 距离范数 (1 x (num_targets - 1))
            d_norm = np.linalg.norm(d_xy, axis=0, keepdims=True)

            # f = R1 / (d_norm + R2)
            f = R1 / (d_norm + R2)

            # 斥力项 sum(f/d_norm * d_xy) (2 x 1)
            # np.divide(f, d_norm) 执行广播 (1 x num_targets-1)
            # d_xy 是 2 x num_targets-1
            # 乘法 f/d_norm * d_xy 会在每行上广播
            repulsion[:, i:i + 1] = np.sum((f / d_norm) * d_xy, axis=1, keepdims=True)

        return repulsion

    def VLPredict(self, X, P, group_center, repulsive, F_follower, E_follower, Q_vl):
        """
        函数功能： 虚拟领导者 (VL) 预测。

        参数:
            X (np.ndarray): 目标状态向量 (N x 1)。
            P (np.ndarray): 误差协方差矩阵 (N x N)。
            group_center (np.ndarray): 群中心的预测状态。
            repulsive (np.ndarray): 斥力向量 (2 x 1)。
            F_follower (np.ndarray): 跟随者状态转移矩阵。
            E_follower (np.ndarray): 跟随者输入矩阵。
            Q_vl (np.ndarray): 过程噪声协方差矩阵。

        返回:
            X_predict (np.ndarray): 预测状态向量。
            P_predict (np.ndarray): 预测误差协方差矩阵。
        """
        # 假设 N=4, 状态顺序: [x, vx, y, vy] (或 [x, y, vx, vy])
        # MATLAB input: F_follower * [0 ;repulsive(1);0;repulsive(2)]

        # 构造 4x1 斥力输入向量 (假设状态顺序是 [x, vx, y, vy] 或 [x, y, vx, vy])
        # MATLAB 构造的向量 [0; repulsive(1); 0; repulsive(2)] (4x1)
        # 假设 repulsive(1) 是 x 轴斥力，repulsive(2) 是 y 轴斥力
        # 这里的输入向量的结构非常特定，我们保持其结构

        # 假设 repulsive 是 2x1 向量
        repulsive_input = np.array([
            [0],
            [repulsive[0, 0]],
            [0],
            [repulsive[1, 0]]
        ])

        # X_predict = F_follower * X + E_follower * group_center + F_follower * [repulsive_input]
        X_predict = F_follower @ X + E_follower @ group_center + F_follower @ repulsive_input

        # P_predict = F_follower * P * F_follower' + Q
        P_predict = F_follower @ P @ F_follower.T + Q_vl

        return X_predict, P_predict