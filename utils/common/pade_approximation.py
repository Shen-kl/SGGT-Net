import numpy as np

import numpy as np


def choose_optimal_m(A, t):
    """
    自动选择最优 Padé 阶数 m 和对应的缩放因子 s，
    使得误差控制在机器精度范围内，并最小化计算成本。

    该实现基于 Scaling and Squaring 策略，参考了 Al-Mohy 和 Higham 的研究。

    输入:
      A (np.ndarray): 输入方阵。
      t (int): 用于计算 alpha 的 A 的幂次 (即计算 ||A^t||^(1/t))。

    输出:
      best_m (int): 最优 Padé 阶数 m。
      best_s (int): 最优缩放因子 s。
      alpha (float): 估计的 ||A|| 相关的范数 ||A^t||^(1/t)。
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('输入 A 必须是方阵')

    # theta values from paper (Al-Mohy & Higham)
    theta_list = [1.4956e-2, 2.5391e-1, 9.5042e-1, 2.0978, 4.25]
    m_list = [3, 5, 7, 9, 13]

    # --- Step 1: Estimate alpha = ||A^t||^(1/t) ---
    # Python 中矩阵乘法使用 @ 或 np.linalg.matrix_power

    # 计算 A^t
    if t == 1:
        At = A
    else:
        # np.linalg.matrix_power(M, n) 计算 M^n
        At = np.linalg.matrix_power(A, t)

    # 计算 1-范数 ||A^t||_1
    norm_At = np.linalg.norm(At, ord=1)

    # 计算 alpha = ||A^t||^(1/t)
    # 使用 np.power(base, exponent) 或 ** 运算符
    alpha = np.power(norm_At, 1.0 / t) if t > 0 else np.inf

    # --- Step 2: Try all m and compute corresponding s and cost ---
    best_cost = np.inf
    best_m = 13
    best_s = 0

    for i in range(len(m_list)):
        m = m_list[i]
        theta_m = theta_list[i]

        # 计算缩放因子 s
        # s = max(0, ceil(log2(alpha / theta_m)))
        ratio = alpha / theta_m

        # 确保 alpha/theta_m > 0
        if ratio <= 0:
            # 如果 alpha 接近 0，不需要缩放
            s = 0
        else:
            s = max(0, int(np.ceil(np.log2(ratio))))

        # 计算成本 Cost = #matrix multiplications: approx m/2 + s squarings
        # m 阶 Padé 需要 m/2 + 1/2 (如果 m 奇数) 次矩阵乘法来计算 N(A) 和 D(A)
        # 成本 = 计算 N/D 的乘法次数 + s 次平方
        cost = np.ceil(m / 2) + s

        if cost < best_cost:
            best_cost = cost
            best_m = m
            best_s = s

    return best_m, best_s, alpha


# # 示例调用 (需要一个方阵 A 和幂次 t)
# # 假设 A 是一个 4x4 矩阵，t=2
# # A_example = np.array([[1, 2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# # m, s, alpha = choose_optimal_m(A_example, t=4)
# # print(f"最优 Padé 阶数 m: {m}")
# # print(f"最优缩放因子 s: {s}")

import numpy as np


def choose_s_with_power_norm(A, m, theta_m, t):
    """
    选择最小缩放因子 s，使得 2^{-s} * alpha <= theta_m。
    其中 alpha 是矩阵 A 的 t 阶幂范数的估计值：alpha ≈ ||A^t||_1^(1/t)。

    输入:
      A (np.ndarray): 输入方阵。
      m (int): Padé 阶数 (用于函数签名，但未直接用于计算 s)。
      theta_m (float): Padé 近似阶数 m 对应的误差阈值 (来自 Higham 的研究)。
      t (int): 用于计算幂范数的 A 的幂次。

    输出:
      s (int): 最优缩放因子 s (s >= 0)。
      alpha (float): 估计的幂范数 ||A^t||_1^(1/t)。
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('输入 A 必须是方阵')
    if t <= 0:
        raise ValueError('幂次 t 必须大于 0')

    # --- Step 1: Estimate alpha = ||A^t||_1^(1/t) ---

    # 计算 A^t
    # np.linalg.matrix_power(M, n) 比循环乘法更高效
    At = np.linalg.matrix_power(A, t)

    # 计算 1-范数 ||A^t||_1
    norm_At = np.linalg.norm(At, ord=1)

    # 计算 alpha
    alpha = np.power(norm_At, 1.0 / t)

    # --- Step 2: Solve inequality for s ---
    # 我们需要满足: alpha / 2^s <= theta_m
    # 即: 2^s >= alpha / theta_m
    # 即: s >= log2(alpha / theta_m)

    ratio = alpha / theta_m

    if ratio <= 1.0:
        # 如果 alpha <= theta_m，则 s=0 (不需要缩放)
        s = 0
    else:
        # s = max(0, ceil(log2(alpha / theta_m)))
        s = int(np.ceil(np.log2(ratio)))

    return s, alpha

# ----------------------------------------------------------------------
# Padé 近似主函数
# ----------------------------------------------------------------------

def pade_approximate(A, n, theta_n=None, t=None):
    """
    手动实现任意阶[n/n] Padé 近似用于计算矩阵指数 expm(A)。

    输入:
      A (np.ndarray): 输入方阵
      n (int): Padé 近似阶数（推荐 3, 5, 7, 9, 13）
      theta_n, t: 传递给 choose_s_with_power_norm 的可选参数

    输出:
      E (np.ndarray): 近似的 expm(A)
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('输入必须是方阵')

    I = np.eye(A.shape[0])

    # Step 1: Scaling（缩放）
    s, _ = choose_s_with_power_norm(A, n, theta_n, t)

    A_scaled = A / (2 ** s)  # 缩放后的 A

    # Step 2: 计算 Padé 系数 c_k
    # c_k = (2n - k)! * n! / [(2n)! * k! * (n - k)!]

    # Python 中使用字典存储预计算的系数，键为阶数 n
    PRECALCULATED_COEFFS = {
        3: [1.0, 0.5, 0.1, 0.00833333333333333],
        5: [1.0, 0.5, 0.12, 0.01833333333333333,
            0.0019927536231884053, 1.630434782608695e-4],
        7: [1.0, 0.5, 0.115384615384615, 0.0160256410256410,
            0.00145687645687646, 8.74125874125874e-05, 3.23750323750324e-06,
            5.78125578125578e-08],
        9: [1.0, 0.5, 0.117647058823529, 0.0171568627450980,
            0.00171568627450980, 0.000122549019607843, 6.28456510809452e-06,
            2.24448753860519e-07, 5.10110804228451e-09, 5.66789782476057e-11],
        13: [1.0, 0.5, 0.12, 0.0183333333333333,
             0.00199275362318841, 0.000163043478260870, 1.03519668737060e-05,
             5.17598343685300e-07, 2.04315135665250e-08, 6.30602270571759e-10,
             1.48377004840414e-11, 2.52915349159797e-13, 2.81017054621996e-15,
             1.54404975067031e-17]
    }

    if n in PRECALCULATED_COEFFS:
        c = np.array(PRECALCULATED_COEFFS[n])
    else:
        # 使用 numpy 的 lgamma 函数计算阶乘，更稳定
        from scipy.special import comb
        c = np.zeros(n + 1)

        # c_k = (comb(2n, k) * comb(2n-k, n-k) / comb(2n, n)) / comb(n, k)
        # Padé 系数 c_k 也可以通过递归关系或公式计算，
        # 但MATLAB代码使用阶乘公式 c_k = (2n - k)! * n! / [(2n)! * k! * (n - k)!]

        # 简化版: c_k = (2n-k)! n! / ( (2n)! k! (n-k)! ) = 1/k! * (n!/(n-k)!) / ( (2n)!/(2n-k)! )
        # c_k = 1/k! * P(n, k) / P(2n, k)

        # 保持与 MATLAB 逻辑一致，使用 np.math.factorial (需要 Python 3.9+ 或使用 np.math.gamma)
        # 注意：对于大数，np.math.factorial 可能会溢出，实际应使用 log-gamma。
        from math import factorial

        for k in range(n + 1):
            term1 = factorial(2 * n - k) * factorial(n)
            term2 = factorial(2 * n) * factorial(k) * factorial(n - k)
            c[k] = term1 / term2

    # Step 3: 构造分子 N 和分母 D
    # N(A) = sum_{k=0}^n c_k A^k
    # D(A) = sum_{k=0}^n c_k (-1)^k A^k = N(-A)

    N = np.zeros_like(A)
    D = np.zeros_like(A)
    A_power = I  # A^0

    for k in range(n + 1):
        # 注意: k+1 对应 c 的索引
        ck = c[k]

        N = N + ck * A_power
        D = D + ck * ((-1) ** k) * A_power

        # 更新 A_power
        A_power = A_power @ A_scaled  # 矩阵乘法用 @

    # Step 4: 求解 R = D^{-1} * N
    # MATLAB 的 D \ N 对应于求解线性方程组 D * R = N
    # 推荐使用 np.linalg.solve 而非 np.linalg.inv，以提高数值稳定性
    try:
        R = np.linalg.solve(D, N)
    except np.linalg.LinAlgError:
        # 如果 D 是奇异的，回退到使用伪逆或发出警告
        R = np.linalg.pinv(D) @ N
        # 实际应用中，如果 D 奇异，通常意味着 Padé 阶数或缩放有问题

    # Step 5: 还原缩放 exp(A) ≈ (D^{-1}N)^(2^s)
    # R = R * R (s 次)
    E = R
    for i in range(s):
        E = E @ E

    return E