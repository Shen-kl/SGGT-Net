import numpy as np

def loopy_belief_propagation(radar, associationInfo):
    """
    函数功能：基于因子图，使用循环置信度传播 (LBP) 计算数据关联信息。

    该函数通过迭代地在目标节点(a)和量测节点(b)之间传递消息，直到收敛，
    来估计数据关联变量的边缘概率信息。

    输入:
        radar (object): 包含跟踪系统信息的对象，特别是 radar.legacy_PT_num (旧航迹数量)。
        associationInfo (object): 包含初始消息的结构体，特别是 beta_message 和 xi_message。
            - beta_message (np.ndarray): 目标导向的关联变量信息 (I+1) x (J+1)。
            - xi_message (np.ndarray): 量测导向的关联变量信息 (I+1) x (J+1)。
            其中 I = radar.legacy_PT_num (目标数), J = radar.plot_track_num (量测数)。
            (第一行/列通常用于表示未检测/杂波的虚拟目标/量测)

    返回:
        message_aq (np.ndarray): 目标传递给联合分布 q(x,r,a,z) 的信息。
        message_bv (np.ndarray): 量测传递给联合分布 v(x,r,b,z) 的信息。
    """

    # 初始化输出
    message_aq = None
    message_bv = None

    # 检查是否存在旧航迹，只有存在旧航迹时才执行LBP
    if radar.legacy_PT_num <= 0:
        return message_aq, message_bv

    # 初始化 LBP 参数
    notConverged = True
    counter = 0
    max_counter = 100
    epsilon = 1e-6

    num_targets = radar.legacy_PT_num
    num_plots = radar.plot_track_num

    # 目标导向信息 (I+1) x (J+1)
    beta_message = associationInfo['beta_message'].copy()
    # 量测导向信息 (I+1) x (J+1)
    xi_message = associationInfo['xi_message'].copy()

    # 1. 计算未检测概率相关的 phi (用于归一化)
    # phi(tar_index) = (1 - Pd) * r


    # 2. 初始化消息矩阵
    # 目标传递至量测的信息 (message_ab, 目标->量测, 归一化)
    # MATLAB: (I+1) x J
    message_ab = np.ones((num_targets + 1, num_plots))
    # 量测传递至目标的信息 (message_ba, 量测->目标, 归一化)
    # MATLAB: I x (J+1)
    message_ba = np.ones((num_targets, num_plots + 1))

    # message_bv 和 message_aq 是最终的输出，但在 MATLAB 中它们是 message_ab 和 message_ba 的最终值
    # MATLAB: message_bv = (I+1) x J, message_aq = I x (J+1)
    # 我们将在循环结束后赋值。

    # 3. 初始消息归一化

    # 目标信息归一化：使 a_{k}=0 时的信息归一化为1
    # 提取第一列 (未检测/杂波)
    eta_1 = beta_message[1:, 0].reshape(-1, 1)  # I x 1
    # beta_message(2:end,:) = beta_message(2:end,:) ./ eta_1;
    # (目标关联到所有量测/未检测，除了杂波)
    beta_message[1:, :] = beta_message[1:, :] / eta_1

    # 量测信息归一化：使 b_{k}=0 时的信息归一化为1
    # 提取第一行 (未关联)
    eta_2 = xi_message[0, 1:].reshape(1, -1)  # 1 x J
    # xi_message(:,2:end) = xi_message(:,2:end) ./ eta_2;
    # (所有目标/未检测关联到量测，除了未关联)
    xi_message[:, 1:] = xi_message[:, 1:] / eta_2

    # LBP 迭代循环
    while notConverged:
        message_ba_old = message_ba.copy()

        # --- 目标传递至量测的信息 (message_ab) ---
        # 消息传递仅在 I x J 的关联块内 (目标 1..I, 量测 1..J)

        # P = beta_message(2:end,2:end) .* message_ba(:,2:end); (I x J)
        P = beta_message[1:, 1:] * message_ba[:, 1:]

        # message_ab(2:end,:) = beta_message(2:end,2:end) ./ (1 + sum(P, 2) - P); (I x J)
        # sum(P, 2) 是 MATLAB 语法，在 numpy 中是 sum(P, axis=1, keepdims=True) (I x 1)
        # message_ab[1:, :] 对应 I x J 的关联块
        sum_P = np.sum(P, axis=1, keepdims=True)  # I x 1

        # message_ab[1:, :] 是 I x J 矩阵
        message_ab[1:, :] = beta_message[1:, 1:] / (1 + sum_P - P)

        # --- 量测传递至目标的信息 (message_ba) ---

        # B = xi_message(2:end,2:end) .* message_ab(2:end,:); (I x J)
        B = xi_message[1:, 1:] * message_ab[1:, :]

        # message_ba(:,2:end) = xi_message(2:end,2:end) ./ (1 + sum(B, 1) - B); (I x J)
        # sum(B, 1) 是 MATLAB 语法，在 numpy 中是 sum(B, axis=0, keepdims=True) (1 x J)
        # message_ba[:, 1:] 对应 I x J 的关联块
        sum_B = np.sum(B, axis=0, keepdims=True)  # 1 x J

        # message_ba[:, 1:] 是 I x J 矩阵
        message_ba[:, 1:] = xi_message[1:, 1:] / (1 + sum_B - B)

        # --- 检查收敛 ---
        counter += 1
        # delta = abs(message_ba_old - message_ba);
        delta = abs(message_ba_old - message_ba)

        # notConverged = (max(delta(:)) > epsilon && counter < max_counter);
        if np.max(delta) < epsilon or counter >= max_counter:
            notConverged = False

    # --- 最终消息和边缘似然计算 (Post-Convergence) ---

    # message_aq = message_ba;
    message_aq = message_ba.copy()
    # message_bv = message_ab;
    message_bv = message_ab.copy()

    # 目标传递至联合分布q(x,r,a,z) - 目标节点边缘似然的两个分量
    # message_aq(:,1) = message_aq(:,1) .* eta_1; (I x 1)
    # message_aq[:, 0] = message_aq[:, 0] * eta_1.flatten() # flatten() 将Numpy数组变为行向量

    # message_aq(:,2:end) = P .* eta_1; (I x J)
    # message_aq[:, 1:] = P * eta_1  # 广播 (I x J) * (I x 1)

    # 量测传递至联合分布v(x,r,b,z) - 量测节点边缘似然的两个分量
    # message_bv(1,:) = message_bv(1,:) .* eta_2; (1 x J)
    # message_bv[0, :] = message_bv[0, :] * eta_2.flatten()

    # message_bv(2:end,:) = B .* eta_2; (I x J)
    # message_bv[1:, :] = B * eta_2  # 广播 (I x J) * (1 x J)

    return message_aq, message_bv