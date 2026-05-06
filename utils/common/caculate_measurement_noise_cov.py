import numpy as np
from numpy import sin, cos, exp, sinh, cosh, pi


def calculate_measurement_noise_cov(R, A, E, rangeNoise, aziNoise, eleNoise):
    """
    函数功能：计算极坐标系转换至直角坐标系下时，观测误差矩阵的转换。

    该函数使用近似解析方法计算由极坐标系 (R, A, E) 误差 (假设为高斯分布)
    转换到直角坐标系 (X, Y, Z) 后的等效观测噪声协方差矩阵和均值偏差项。

    注意:
    1. 输入的角度 A (方位角), E (俯仰角) 必须是以度 (degree) 为单位。
    2. 输入的误差 aziNoise, eleNoise 必须是以弧度 (radian) 为单位。
    3. 假设 X 轴对应 cos(A)*cos(E)，Y 轴对应 sin(A)*cos(E)。(这是常见的雷达坐标系约定，
       与 MATLAB 原代码中的顺序相反，但我们严格遵循 MATLAB 原代码中的数学表达式。)

    输入:
        R (float): 量测径向距离。
        A (float): 量测方位角 (度)。
        E (float): 量测俯仰角 (度)。
        rangeNoise (float): 距离误差的标准差。
        aziNoise (float): 方位角误差的标准差 (弧度)。
        eleNoise (float): 俯仰角误差的标准差 (弧度)。

    返回:
        mu (np.ndarray): 均值偏差向量 (3x1)。
        measureNoiseCov (np.ndarray): 转换后的观测噪声协方差矩阵 (3x3)。
    """

    # 将输入角度从度转换为弧度
    A_rad = A * pi / 180
    E_rad = E * pi / 180

    # 1. 计算均值偏差向量 mu (3x1)

    # mu(1) - 对应 X 轴
    # mu(1) = R*sin(A)*cos(E) * [exp(-(aziNoise)^2)*exp(-(eleNoise)^2) - exp(-(eleNoise)^2/2)*exp(-(aziNoise)^2/2)]
    mu1_factor = exp(-(aziNoise) ** 2) * exp(-(eleNoise) ** 2) - exp(-(eleNoise) ** 2 / 2) * exp(-(aziNoise) ** 2 / 2)
    mu1 = R * sin(A_rad) * cos(E_rad) * mu1_factor

    # mu(2) - 对应 Y 轴
    # mu(2) = R*cos(A)*cos(E) * [exp(-(aziNoise)^2)*exp(-(eleNoise)^2) - exp(-(eleNoise)^2/2)*exp(-(aziNoise)^2/2)]
    mu2 = R * cos(A_rad) * cos(E_rad) * mu1_factor

    # mu(3) - 对应 Z 轴
    # mu(3) = R*sin(E) * [exp(-(eleNoise)^2) - exp(-(eleNoise)^2/2)]
    mu3_factor = exp(-(eleNoise) ** 2) - exp(-(eleNoise) ** 2 / 2)
    mu3 = R * sin(E_rad) * mu3_factor

    mu = np.array([[mu1], [mu2], [mu3]])

    # 2. 计算辅助变量 alpha 和 beta

    # Alpha terms (用于协方差的中间计算)
    # alpha_y = sin(A)^2*sinh(aziNoise^2) + cos(A)^2*cosh(aziNoise^2)
    alpha_y = sin(A_rad) ** 2 * sinh(aziNoise ** 2) + cos(A_rad) ** 2 * cosh(aziNoise ** 2)
    # alpha_x = sin(A)^2*cosh(aziNoise^2) + cos(A)^2*sinh(aziNoise^2)
    alpha_x = sin(A_rad) ** 2 * cosh(aziNoise ** 2) + cos(A_rad) ** 2 * sinh(aziNoise ** 2)
    # alpha_z = sin(E)^2*cosh(eleNoise^2) + cos(E)^2*sinh(eleNoise^2)
    alpha_z = sin(E_rad) ** 2 * cosh(eleNoise ** 2) + cos(E_rad) ** 2 * sinh(eleNoise ** 2)
    # alpha_xy = sin(E)^2*sinh(eleNoise^2) + cos(E)^2*cosh(eleNoise^2)
    alpha_xy = sin(E_rad) ** 2 * sinh(eleNoise ** 2) + cos(E_rad) ** 2 * cosh(eleNoise ** 2)

    # Beta terms (用于协方差的中间计算)
    # beta_y = sin(A)^2*sinh(2*aziNoise^2) + cos(A)^2*cosh(2*aziNoise^2)
    beta_y = sin(A_rad) ** 2 * sinh(2 * aziNoise ** 2) + cos(A_rad) ** 2 * cosh(2 * aziNoise ** 2)
    # beta_x = sin(A)^2*cosh(2*aziNoise^2) + cos(A)^2*sinh(2*aziNoise^2)
    beta_x = sin(A_rad) ** 2 * cosh(2 * aziNoise ** 2) + cos(A_rad) ** 2 * sinh(2 * aziNoise ** 2)
    # beta_z = sin(E)^2*cosh(2*eleNoise^2) + cos(E)^2*sinh(2*eleNoise^2)
    beta_z = sin(E_rad) ** 2 * cosh(2 * eleNoise ** 2) + cos(E_rad) ** 2 * sinh(2 * eleNoise ** 2)
    # beta_xy = sin(E)^2*sinh(2*eleNoise^2) + cos(E)^2*cosh(2*eleNoise^2)
    beta_xy = sin(E_rad) ** 2 * sinh(2 * eleNoise ** 2) + cos(E_rad) ** 2 * cosh(2 * eleNoise ** 2)

    # 3. 计算协方差矩阵的各个元素
    rangeNoise_sq = rangeNoise ** 2
    aziNoise_sq = aziNoise ** 2
    eleNoise_sq = eleNoise ** 2
    R_sq = R ** 2

    # R_xx
    # R_xx = (R^2*(beta_x*beta_xy-alpha_x*alpha_xy) + rangeNoise^2*(2*beta_x*beta_xy-alpha_x*alpha_xy)) * exp(-2*aziNoise^2) * exp(-2*eleNoise^2)
    R_xx_term1 = R_sq * (beta_x * beta_xy - alpha_x * alpha_xy)
    R_xx_term2 = rangeNoise_sq * (2 * beta_x * beta_xy - alpha_x * alpha_xy)
    R_xx = (R_xx_term1 + R_xx_term2) * exp(-2 * aziNoise_sq) * exp(-2 * eleNoise_sq)

    # R_yy
    # R_yy = (R^2*(beta_y*beta_xy-alpha_y*alpha_xy) + rangeNoise^2*(2*beta_y*beta_xy-alpha_y*alpha_xy)) * exp(-2*aziNoise^2) * exp(-2*eleNoise^2)
    R_yy_term1 = R_sq * (beta_y * beta_xy - alpha_y * alpha_xy)
    R_yy_term2 = rangeNoise_sq * (2 * beta_y * beta_xy - alpha_y * alpha_xy)
    R_yy = (R_yy_term1 + R_yy_term2) * exp(-2 * aziNoise_sq) * exp(-2 * eleNoise_sq)

    # R_zz
    # R_zz = (R^2*(beta_z-alpha_z) + rangeNoise^2*(2*beta_z-alpha_z)) * exp(-2*eleNoise^2)
    R_zz_term1 = R_sq * (beta_z - alpha_z)
    R_zz_term2 = rangeNoise_sq * (2 * beta_z - alpha_z)
    R_zz = (R_zz_term1 + R_zz_term2) * exp(-2 * eleNoise_sq)

    # R_xy
    # R_xy = (R^2*(beta_xy-alpha_xy*exp(aziNoise^2)) + rangeNoise^2*(2*beta_xy-alpha_xy*exp(aziNoise^2))) * sin(A) * cos(A) * exp(-4*aziNoise^2) * exp(-2*eleNoise^2)
    R_xy_term1 = R_sq * (beta_xy - alpha_xy * exp(aziNoise_sq))
    R_xy_term2 = rangeNoise_sq * (2 * beta_xy - alpha_xy * exp(aziNoise_sq))
    R_xy_factor = sin(A_rad) * cos(A_rad) * exp(-4 * aziNoise_sq) * exp(-2 * eleNoise_sq)
    R_xy = (R_xy_term1 + R_xy_term2) * R_xy_factor

    # R_xz
    # R_xz = (R^2*(1-exp(eleNoise^2)) + rangeNoise^2*(2-exp(eleNoise^2))) * sin(A) * sin(E) * cos(E) * exp(-aziNoise^2) * exp(-4*eleNoise^2)
    R_xz_term1 = R_sq * (1 - exp(eleNoise_sq))
    R_xz_term2 = rangeNoise_sq * (2 - exp(eleNoise_sq))
    R_xz_factor = sin(A_rad) * sin(E_rad) * cos(E_rad) * exp(-aziNoise_sq) * exp(-4 * eleNoise_sq)
    R_xz = (R_xz_term1 + R_xz_term2) * R_xz_factor

    # R_yz
    # R_yz = (R^2*(1-exp(eleNoise^2)) + rangeNoise^2*(2-exp(eleNoise^2))) * cos(A) * sin(E) * cos(E) * exp(-aziNoise^2) * exp(-4*eleNoise^2)
    R_yz_term1 = R_sq * (1 - exp(eleNoise_sq))
    R_yz_term2 = rangeNoise_sq * (2 - exp(eleNoise_sq))
    R_yz_factor = cos(A_rad) * sin(E_rad) * cos(E_rad) * exp(-aziNoise_sq) * exp(-4 * eleNoise_sq)
    R_yz = (R_yz_term1 + R_yz_term2) * R_yz_factor

    # 4. 构建对称协方差矩阵 (3x3)
    # measureNoiseCov=[R_xx R_xy R_xz; R_xy R_yy R_yz ; R_xz R_yz R_zz];
    measureNoiseCov = np.array([
        [R_xx, R_xy, R_xz],
        [R_xy, R_yy, R_yz],
        [R_xz, R_yz, R_zz]
    ])

    return mu, measureNoiseCov