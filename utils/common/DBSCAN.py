import numpy as np

def edist(tracks, H):
    """
    计算轨迹集中每对不同轨迹在观测空间投影后的欧式距离矩阵。

    欧式距离 d 定义为 d = || H * X_i(:,end) - H * X_j(:,end) ||_2，
    其中 ||.||_2 是 L2 范数（欧式距离）。

    输入:
      tracks (list of dict/object): 轨迹对象列表。每个对象必须有：
          - 'X' (np.ndarray): 状态向量 (e.g., 4xN)。
      H (np.ndarray): 观测矩阵 (e.g., 2x4)。

    输出:
      edisVector (np.ndarray): 欧式距离向量 (长度为 N*(N-1)/2)。
    """
    num_tracks = len(tracks)

    # 计算需要存储的欧式距离数量 (N*(N-1)/2)
    num_pairs = int(num_tracks * (num_tracks - 1) / 2)
    edisVector = np.zeros(num_pairs)

    index = 0

    # 遍历所有不重复的轨迹对 (i, j)，其中 i < j
    for i in range(num_tracks):
        for j in range(i + 1, num_tracks):
            # 获取轨迹 i 和 j 的最新状态 X
            track_i = tracks[i]
            track_j = tracks[j]

            # X[:, -1] 获取最新的状态向量 (e.g., 4x1)
            X_i = track_i.X[-1].reshape(-1, 1)  # 确保是列向量
            X_j = track_j.X[-1].reshape(-1, 1)

            # 1. 计算观测空间中的残差 d_obs = H * X_i - H * X_j
            # Python 中使用 @ 进行矩阵乘法
            d_obs = H @ X_i - H @ X_j  # e.g., 2x1

            # 2. 计算欧式距离 d = ||d_obs||_2
            # MATLAB 的 norm() 默认是 L2 范数
            d = np.linalg.norm(d_obs)

            # 存储标量结果
            edisVector[index] = d

            index += 1

    return edisVector

def sdist(tracks, H, R):
    """
    计算轨迹集中每对不同轨迹之间的统计距离矩阵 (Statistical Distance)。

    统计距离 D 定义为 D = d' * inv(S) * d，其中：
      d = H * (X_i - X_j) 是观测残差。
      S = H * (P_i + P_j) * H' + R 是残差协方差。

    输入:
      tracks (list of dict/object): 轨迹对象列表。每个对象必须有：
          - 'P' (np.ndarray): 协方差矩阵 (e.g., 4x4)。
          - 'X' (np.ndarray): 状态向量 (e.g., 4xN)。
      H (np.ndarray): 观测矩阵 (e.g., 2x4)。
      R (np.ndarray): 观测噪声协方差矩阵 (e.g., 2x2)。

    输出:
      sdisVector (np.ndarray): 统计距离向量 (长度为 N*(N-1)/2)。
    """
    num_tracks = len(tracks)

    # 计算需要存储的统计距离数量 (N*(N-1)/2)
    num_pairs = int(num_tracks * (num_tracks - 1) / 2)
    sdisVector = np.zeros(num_pairs)

    index = 0

    # 遍历所有不重复的轨迹对 (i, j)，其中 i < j
    for i in range(num_tracks):
        for j in range(i + 1, num_tracks):
            # 获取轨迹 i 和 j 的协方差 P 和最新状态 X
            track_i = tracks[i]
            track_j = tracks[j]

            P_i = track_i['P']  # 假设 P 是一个键，或者使用 track_i.P
            P_j = track_j['P']

            # X[:, -1] 获取最新的状态向量 (e.g., 4x1)
            X_i = track_i['X'][:, -1].reshape(-1, 1)  # 确保是列向量
            X_j = track_j['X'][:, -1].reshape(-1, 1)

            # 1. 计算残差协方差 S = H * (P_i + P_j) * H' + R
            # Python 中使用 @ 进行矩阵乘法
            S = H @ (P_i + P_j) @ H.T + R

            # 2. 计算观测残差 d = H * (X_i - X_j)
            d = H @ (X_i - X_j)  # e.g., 2x1

            # 3. 计算统计距离 D = d' * inv(S) * d
            # 使用 np.linalg.solve(S, d) 求解 S * x = d，得到 inv(S) * d
            # 然后计算 d.T * x
            try:
                inv_S_d = np.linalg.solve(S, d)
            except np.linalg.LinAlgError:
                # S 奇异或病态，使用伪逆作为回退
                inv_S_d = np.linalg.pinv(S) @ d

            D = d.T @ inv_S_d

            # 存储标量结果
            sdisVector[index] = D.item()

            index += 1

    return sdisVector


import numpy as np


def RegionQuery_edist(edisVector, current_index, NumMax, param):
    """
    根据欧式距离向量，查找与当前要处理的航迹相邻的其他航迹的索引。
    该函数用于基于欧式距离的邻域查询，常用于 DBSCAN 等聚类算法。

    输入:
      edisVector (np.ndarray): 欧式距离向量 (按上三角矩阵顺序存储，长度为 N*(N-1)/2)。
      current_index (int): 当前要处理的航迹在数组中的 0-based 下标 (0 <= current_index < NumMax)。
      NumMax (int): 航迹的总数量 (即 edist 函数输入的 tracks 列表长度)。
      param (dict): 包含 DBSCAN 参数的字典。
          - param['dbscan_param']['distance_gate'] (float): 欧式距离门限 (epsilon)。

    输出:
      Neighbors (np.ndarray): 与当前航迹相邻的其他航迹的 0-based 下标数组。
    """

    # 距离门限 (epsilon)
    # MATLAB 代码注释中使用了 2 * distance_gate，但在实际 DBSCAN 中通常只用 distance_gate
    # 我们遵循 MATLAB 代码的逻辑：threshold = 2 * param.dbscan_param.distance_gate
    distance_threshold = 2 * param['dbscan_param']['distance_gate']

    Neighbors = []

    # MATLAB 代码的目标是找到所有 j > index 的距离对 (index, j)
    # distance_vector 存储的是 (i, j) 距离，i < j。

    # --- 1. 计算与 j > current_index 的轨迹对 (current_index, j) 相关的起始和结束索引 ---

    # 航迹索引 i 从 0 到 NumMax - 1
    # 对于给定的 current_index (i)，我们查找 j = current_index + 1 到 NumMax - 1 的距离。

    # distance_vector 中的起始位置 (0-based)
    # 累计 i = 0 到 i = current_index - 1 的距离对数量
    begin = 0
    for i in range(current_index):
        # 航迹 i 产生的距离对数量: NumMax - 1 - i
        begin += (NumMax - 1 - i)

    # 距离向量中，与 current_index 相关的距离对数量: NumMax - 1 - current_index
    num_related_pairs = NumMax - 1 - current_index
    last = begin + num_related_pairs  # 实际是 [begin, last)

    # --- 2. 提取距离并执行查询 ---

    if num_related_pairs > 0:
        # 提取与 current_index (i) 相关且 j > i 的距离向量片段
        related_distances = edisVector[begin: last]

        # 找到满足距离门限的距离的索引 k (k 从 0 到 num_related_pairs - 1)
        # 此时，第 k 个距离对应于轨迹对 (current_index, current_index + 1 + k)

        # indices_of_neighbors 是在 related_distances 中的 0-based 索引
        indices_of_neighbors = np.where(related_distances <= distance_threshold)[0]

        # 转换回实际的轨迹索引 j
        # j = current_index + 1 + k
        neighbor_indices = current_index + 1 + indices_of_neighbors

        Neighbors.extend(neighbor_indices)

    # --- 3. 额外处理：查找 i < current_index 的轨迹对 (i, current_index) ---
    # MATLAB 代码只处理了 j > index 的情况，如果需要完整的邻域，还需要反向查找。
    # MATLAB 代码只检查了 j > index 的情况，即它假设调用者会遍历所有 index 来构建邻域。
    # 我们保持与 MATLAB 代码相同的行为（只检查 j > index 的距离）。
    # 如果需要查找 i < current_index 的距离，则需要更复杂的逻辑来找到 edisVector 中对应的位置。
    # 例如：
    # for i in range(current_index):
    #     distance_vector_idx = get_edisVector_index(i, current_index, NumMax)
    #     if edisVector[distance_vector_idx] <= distance_threshold:
    #         Neighbors.append(i)

    # 由于原始 MATLAB 函数只关注 j > index 的部分，我们只返回这部分。

    return np.array(Neighbors, dtype=int)


# ----------------------------------------------------------------------
# 辅助函数：计算 edisVector 中的索引 (如果需要反向查找，会用到)
# ----------------------------------------------------------------------

def get_edisVector_index(i, j, NumMax):
    """
    计算轨迹对 (i, j) (i < j) 在 edisVector 中的 0-based 索引。
    """
    if i >= j:
        raise ValueError("i 必须小于 j")

    # 累计 i=0 到 i=i-1 的距离对数量
    index_offset = 0
    for k in range(i):
        index_offset += (NumMax - 1 - k)

    # 加上 i 产生的距离对数量 (j - (i + 1))
    # i 产生的距离对是: (i, i+1), (i, i+2), ..., (i, j), ...
    index = index_offset + (j - (i + 1))
    return index


import numpy as np


# 假设 RegionQuery_edist 函数已经按照前面的转换定义
# from .region_query import RegionQuery_edist
# 为了代码完整性，这里假设它已经被导入或定义在同文件中。

def ExpandCluster_edist(IDX, visited, Neighbors, edisVector, NumMax, param, C, adjacent_matrix):
    """
    扩展簇类的大小，向外探索与当前簇密度可达的点。
    这是 DBSCAN 算法的核心扩展步骤。

    输入:
      IDX (np.ndarray): 航迹的簇索引向量 (0 表示未分类，C 是当前簇的索引)。
      visited (np.ndarray): 航迹是否被访问过的布尔向量。
      Neighbors (list/np.ndarray): 当前簇的邻居列表 (0-based 索引)。
      edisVector (np.ndarray): 欧式距离向量。
      NumMax (int): 航迹的总数量。
      param (dict): 包含 DBSCAN 参数的字典。
      C (int): 当前簇的索引 (通常 > 0)。
      adjacent_matrix (np.ndarray): 邻接矩阵，用于记录可达关系。

    输出:
      IDX (np.ndarray): 更新后的簇索引向量。
      visited (np.ndarray): 更新后的访问状态向量。
      adjacent_matrix (np.ndarray): 更新后的邻接矩阵。
    """

    # 转换为列表，方便在循环中动态添加元素
    if isinstance(Neighbors, np.ndarray):
        Neighbors = Neighbors.tolist()

    k = 0  # Python 使用 0-based 索引

    while True:
        # k 是 Neighbors 列表的索引
        j = Neighbors[k]  # 找到距离小于epsilon的第 k 个直接密度可达点 (0-based 轨迹索引)

        # --- 1. 探索未访问的点 ---
        if not visited[j]:
            visited[j] = True  # 标记为已访问

            # 查询周围点中距离小于epsilon的个数 (Neighbors2 是 0-based 索引数组)
            # 假设 RegionQuery_edist 已经被正确定义并使用 0-based 索引
            Neighbors2 = RegionQuery_edist(edisVector, j, NumMax, param)

            # 更新邻接矩阵: j 到 Neighbors2 中的点是可达的
            # 由于 j 和 Neighbors2 都是 0-based，可以直接索引
            for neighbor_idx in Neighbors2:
                adjacent_matrix[j, neighbor_idx] = 1

            # 将新的邻居添加到 Neighbors 列表，取并集 (但 MATLAB 代码是简单拼接)
            # MATLAB: Neighbors=[Neighbors Neighbors2]; -> Python: Neighbors.extend(Neighbors2)
            # 这种简单拼接可能会导致重复计算，但在 DBSCAN 中通常可以接受，因为 visited 检查会处理重复。
            Neighbors.extend(Neighbors2.tolist())

        # --- 2. 标记簇归属 ---
        if IDX[j] == 0:  # 如果还没形成任何簇类 (0 表示未分配)
            IDX[j] = C  # 将第 j 个点归类到当前簇 C 中

        # --- 3. 移动到下一个邻居 ---
        k += 1

        if k >= len(Neighbors):  # 如果已经遍历完所有直接密度可达的点 (0-based 检查)
            break

    return IDX, visited, adjacent_matrix

# ----------------------------------------------------------------------
# 核心 DBSCAN 函数
# ----------------------------------------------------------------------

def DBSCAN(tracks, tarTracksNum, param):
    """
    对轨迹集执行基于欧式距离的 DBSCAN 聚类。

    输入:
      tracks (list of dict/object): 轨迹对象列表。
      tarTracksNum (int): 轨迹的总数量 N。
      param (dict): 包含 DBSCAN 参数的字典。
      edist_func: 欧式距离计算函数 (e.g., edist)。
      region_query_func: 邻域查询函数 (e.g., RegionQuery_edist)。
      expand_cluster_func: 簇扩展函数 (e.g., ExpandCluster_edist)。

    输出:
      IDX (np.ndarray): 航迹的簇索引向量 (1-based)。
      C (int): 最终的簇数量。
      adjacent_matrix (np.ndarray): 邻接矩阵 (对称)。
    """

    # 0. 计算距离 (虽然计算了 sdist 和 svel，但只使用了 edist)
    # [sdisVector] = sdist(tracks, param['H'], param['R'])
    # [svelVector] = svel(tracks, param['H_vel'])
    edisVector = edist(tracks, param['H'])  # 计算欧式距离向量

    # 初始化
    # IDX: 簇索引 (0-based)
    IDX = np.zeros(tarTracksNum, dtype=int)
    # visited: 访问状态 (0-based)
    visited = np.zeros(tarTracksNum, dtype=bool)

    C = 0  # 簇的个数 (Cluster Count, 1-based)

    # adjacent_matrix: 初始化为单位矩阵 (每个点都邻接自身)
    adjacent_matrix = np.eye(tarTracksNum, dtype=int)

    # --- 1. 遍历所有点 ---
    for i in range(tarTracksNum):  # i 从 0 到 tarTracksNum - 1

        if not visited[i]:  # 未被访问
            visited[i] = True  # 标记为已访问

            # 查询邻居 (Neighbors 是 0-based 索引数组)
            # RegionQuery_edist 假设 i 是 0-based
            Neighbors = RegionQuery_edist(edisVector, i, tarTracksNum, param)

            C += 1  # 簇类个数 + 1 (新的簇索引 C)
            IDX[i] = C  # 标记当前点为新的簇中心

            if Neighbors.size == 0:
                continue
            # 更新邻接矩阵: i 到 Neighbors 中的点是可达的
            for neighbor_idx in Neighbors:
                adjacent_matrix[i, neighbor_idx] = 1

            # 扩展簇 (Neighbors 在这里作为待扩展点的队列)
            # ExpandCluster_edist 将 Neighbors 中的点加入到簇 C
            IDX, visited, adjacent_matrix = ExpandCluster_edist(
                IDX, visited, Neighbors, edisVector, tarTracksNum,
                param, C, adjacent_matrix
            )

    # --- 2. 使邻接矩阵对称 ---
    # adjacent_matrix = adjacent_matrix | adjacent_matrix'
    # 使用逻辑 OR 运算符
    adjacent_matrix = adjacent_matrix | adjacent_matrix.T

    # 返回的 IDX 是 1-based 簇索引，与 MATLAB 一致
    return IDX, C, adjacent_matrix