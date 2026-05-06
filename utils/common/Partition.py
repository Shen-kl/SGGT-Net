import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union, Any


@dataclass
class Partition:
    """用于存储分区结果的数据类"""
    groups: List[List]
    pair_mask: np.ndarray


def limited_set_partitions(S, pair_map, pair_cnt, max_count=None):
    """
    根据max_count限制生成的分区数量

    参数:
        S: 在当前簇中，有效目标的编号集合
        pair_map: N*N的矩阵，存储马氏距离的索引
        pair_cnt: 索引的最大值
        max_count: 最多可以生成的分区组数

    返回:
        partitions: 分区结果列表
    """
    if max_count is None:
        max_count = float('inf')

    # 确保S是一维数组
    S = np.asarray(S).flatten()
    n = len(S)

    if n == 0:
        return [Partition(groups=[], pair_mask=np.zeros(pair_cnt, dtype=bool))]

    # 使用可变对象来存储状态（Python闭包中的nonlocal变量）
    count = 0
    stopped = False

    # 预分配结果列表（使用列表推导式）
    result = [None] * min(max_count, 1000000) if max_count != float('inf') else []

    def dfs(idx, current_parts):
        nonlocal count, stopped

        # 双重保险：检查停止标志和数量限制
        if stopped or count >= max_count:
            stopped = True
            return

        # 基准情况：完成一个完整划分
        if idx >= n:
            count += 1
            new_mask = np.zeros(pair_cnt, dtype=bool)

            for group in current_parts:
                if len(group) >= 2:
                    for i in range(len(group) - 1):
                        for j in range(i + 1, len(group)):
                            pid = pair_map[group[i], group[j]]
                            new_mask[pid] = True

            # 存储结果
            result[count - 1] = Partition(groups=current_parts.copy(), pair_mask=new_mask)

            if count >= max_count:
                stopped = True
            return

        elem = int(S[idx])

        # ========== 策略 1：elem 单独成组 ==========
        dfs(idx + 1, current_parts + [[elem]])
        if stopped:
            return

        # ========== 策略 2：elem 加入已有各组 ==========
        num_groups = len(current_parts)
        for i in range(num_groups):
            # 每次迭代前检查，严格防止超出生成
            if stopped or count >= max_count:
                return

            # 创建新划分：复制并在第 i 组添加 elem
            new_parts = [group.copy() for group in current_parts]
            new_parts[i].append(elem)

            dfs(idx + 1, new_parts)

    # 启动深度优先搜索，从第1个元素、空划分开始
    dfs(0, [])

    # 截取实际生成的有效部分
    if max_count != float('inf'):
        partitions = result[:count]
    else:
        # 对于无限情况，需要实际收集结果
        partitions = result

    return partitions


# 使用示例：
if __name__ == "__main__":
    # 示例数据
    S = [1, 2, 3]
    N = max(S) + 1
    pair_map = np.zeros((N, N), dtype=int)
    # 填充pair_map示例
    idx = 0
    for i in range(0,N-1):
        for j in range(i+1,N):
            if i < j:
                pair_map[i, j] = idx
                pair_map[j, i] = idx
                idx += 1

    pair_cnt = idx

    # 调用函数
    result = limited_set_partitions(S, pair_map, pair_cnt, max_count=10)

    # 打印结果
    for i, part in enumerate(result):
        print(f"分区 {i + 1}: {part.groups}")
        print(f"掩码: {part.pair_mask}")
        print()