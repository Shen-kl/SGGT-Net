import numpy as np

'''
 用于常规的坐标转换
'''

def enu2aer(xEast, yNorth, zUp):
    """
    将地心直角坐标系 (East-North-Up) 转换为雷达球坐标系 (Azimuth-Elevation-Range)。
    假设雷达在原点 (0, 0, 0)。

    参数:
        xEast (np.array): East 坐标 (m)
        yNorth (np.array): North 坐标 (m)
        zUp (np.array): Up 坐标 (m)

    返回:
        az (np.array): 方位角 (rad)
        elev (np.array): 俯仰角 (rad)
        slantRange (np.array): 斜距 (m)
    """
    slantRange = np.sqrt(xEast ** 2 + yNorth ** 2 + zUp ** 2)

    # 方位角 (从 North 顺时针为正，但在标准 atan2 中，从 East 逆时针为正)
    # MATLAB enu2aer 默认的 Azimuth 定义可能与标准 atan2(y, x) 有差异。
    # 标准 atan2(y, x) 返回 [-pi, pi]， East=x, North=y。
    az = np.arctan2(xEast, yNorth)  # 从 North 逆时针

    # 将 azimuth 范围转换为 [0, 2*pi]
    az = np.mod(az, 2 * np.pi)

    # 俯仰角
    elev = np.arcsin(zUp / slantRange)

    return az * 180 / np.pi, elev * 180 / np.pi, slantRange


def aer2enu(az, elev, slantRange):
    """
    将雷达球坐标系 (Azimuth-Elevation-Range) 转换为地心直角坐标系 (East-North-Up)。
    假设雷达在原点 (0, 0, 0)。
    """

    # 投影到水平面上的距离
    range_proj = slantRange * np.cos(elev * np.pi / 180)
    zUp = slantRange * np.sin(elev * np.pi / 180)

    # East/North 坐标 (注意：atan2(x, y) 对应 az，即 xEast = R_proj * sin(az), yNorth = R_proj * cos(az))
    xEast = range_proj * np.sin(az * np.pi / 180)
    yNorth = range_proj * np.cos(az * np.pi / 180)

    return xEast, yNorth, zUp