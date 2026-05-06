import numpy as np
from scipy import signal
from scipy.linalg import expm
from scipy.integrate import quad

def generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T):
    """
    产生领导者跟随者模型的转移矩阵

    参数:
        alpha, beta, gamma, w: 模型参数
        delta_T: 采样时间间隔

    返回:
        F_follower: 跟随者状态转移矩阵 (离散化 A 矩阵)
        E_follower: 跟随者输入矩阵 (离散化 B 矩阵)
        F_leader: 领导者状态转移矩阵
        Q: 跟随者过程噪声协方差矩阵
    """

    # --- 1. 跟随者系统的连续时间状态空间表示 (dX = AX + BU) ---

    A = np.array([
        [0, 1, 0, 0],
        [-alpha, -beta - gamma, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -alpha, -beta - gamma]
    ])

    B = np.array([
        [0, 0],  # 假设输入 U 是 2x1 向量，对应 x 和 y 方向的控制输入（与 MATLAB B 矩阵不同，这里只关注 E_follower 部分）
        [alpha, 0],
        [0, 0],
        [0, alpha]
    ])
    # 注意：MATLAB 代码中的 B 矩阵是 4x4，但 C 矩阵是 2x4，D=0。
    # MATLAB 的 ss(A, B, C, D) 建模与离散化结果 F_follower = sys_d.A 仅依赖于 A。
    # 这里的 B 用于 E_follower 的计算，我们保持与原 MATLAB 逻辑一致，但 B_ss 用于 c2d 离散化。

    B_ss = np.array([
        [0, 0, 0, 0],
        [alpha, beta, 0, 0],
        [0, 0, 0, 0],
        [0, 0, alpha, beta]
    ])

    C = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    D = np.zeros([2,4])

    # --- 2. 连续时间系统离散化 (c2d) ---

    # 转换为状态空间模型 (scipy.signal.StateSpace)
    sys = signal.StateSpace(A, B_ss, C, D)

    # 离散化 (scipy.signal.cont2discrete)
    # 使用 'zoh' (零阶保持) 方法，与 MATLAB c2d 默认行为一致
    sys_d_A, sys_d_B, _, _, _ = signal.cont2discrete((A, B_ss, C, D), delta_T, method='zoh')

    F_follower = sys_d_A
    # 注意: sys_d_B 对应的是 MATLAB c2d 的 B 矩阵，但原 MATLAB 代码没有用到 sys_d.B。
    # F_follower = sys_d.A;

    # --- 3. 跟随者输入矩阵 E_follower (积分项) ---

    # MATLAB A2 (用于 E_follower 的计算)
    A2 = np.array([
        [0, 0, 0, 0],
        [alpha, beta, 0, 0],
        [0, 0, 0, 0],
        [0, 0, alpha, beta]
    ])

    def integrand_E_follower(tau, A, A2, delta_T, w):
        # M_tau 是领导者的状态转移矩阵 F_leader(tau)
        if w != 0:
            M_tau = np.array([
                [1, np.sin(w * tau) / w, 0, -(1 - np.cos(w * tau)) / w],
                [0, np.cos(w * tau), 0, -np.sin(w * tau)],
                [0, (1 - np.cos(w * tau)) / w, 1, np.sin(w * tau) / w],
                [0, np.sin(w * tau), 0, np.cos(w * tau)]
            ])
        else:  # w == 0 (常速/加速度模型)
            M_tau = np.array([
                [1, tau, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, tau],
                [0, 0, 0, 1]
            ])

        # MATLAB: expm(A*(delta_T-tau))*A2*M_tau
        return expm(A * (delta_T - tau)) @ A2 @ M_tau

    # 使用 quad 进行数值积分，对矩阵的每个元素进行积分
    E_follower = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            # 将矩阵函数封装为标量函数，用于 quad
            def scalar_fun(tau):
                return integrand_E_follower(tau, A, A2, delta_T, w)[i, j]

            # quad 返回 (积分值, 估计误差)
            result, _ = quad(scalar_fun, 0, delta_T)
            E_follower[i, j] = result

    # --- 4. 领导者状态转移矩阵 F_leader ---
    # 注意: MATLAB 代码中 F_leader 被计算了两次，最后一次计算覆盖了前面的 if/else 块。
    # 我们采用最后一次计算。

    A_leader = np.array([
        [0, 1, 0, 0],
        [0, gamma, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, gamma]
    ])

    # F_leader = expm(delta_T * A_leader)
    if w != 0:
        F_leader = np.array([
                [1, np.sin(w * delta_T) / w, 0, -(1 - np.cos(w * delta_T)) / w],
                [0, np.cos(w * delta_T), 0, -np.sin(w * delta_T)],
                [0, (1 - np.cos(w * delta_T)) / w, 1, np.sin(w * delta_T) / w],
                [0, np.sin(w * delta_T), 0, np.cos(w * delta_T)]
            ])
    else:
        F_leader =  np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
    # --- 5. 跟随者过程噪声协方差矩阵 Q (积分项) ---

    # MATLAB 噪声输入矩阵 C (这里是 4x2 矩阵，与原 MATLAB C 矩阵不同，代表噪声输入位置)
    C_noise = np.array([
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 1]
    ])

    # 噪声强度对角矩阵 diag([1e-1, 1e-1])
    # 假设噪声强度 Q_c = diag([1e-1, 1e-1])
    Q_c = np.diag([1e-1, 1e-1])

    def integrand_Q(tau, A, delta_T, C_noise, Q_c):
        # MATLAB: expm(A*(delta_T-tau))*C*diag([1e-1,1e-1])*C'*expm(A*(delta_T-tau))'
        Phi = expm(A * (delta_T - tau))
        # 结果是 Phi * C_noise * Q_c * C_noise.T * Phi.T
        return Phi @ C_noise @ Q_c @ C_noise.T @ Phi.T

    # 使用 quad 对矩阵的每个元素进行积分
    Q = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            # 将矩阵函数封装为标量函数，用于 quad
            def scalar_fun_Q(tau):
                return integrand_Q(tau, A, delta_T, C_noise, Q_c)[i, j]

            # quad 返回 (积分值, 估计误差)
            result, _ = quad(scalar_fun_Q, 0, delta_T)
            Q[i, j] = result

    return F_follower, E_follower, F_leader, Q

# --- 使用示例 ---
if __name__ == '__main__':
    alpha = 1.0
    beta = 1.0
    gamma = 0.5
    w = 0.1 # 假设一个非零 w
    delta_T = 0.1
    F_f, E_f, F_l, Q = generateLeaderFollowerTransMat(alpha, beta, gamma, w, delta_T)
    print("F_follower:\n", F_f)
    print("\nE_follower:\n", E_f)
    print("\nF_leader:\n", F_l)
    print("\nQ:\n", Q)