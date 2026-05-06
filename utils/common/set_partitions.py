import numpy as np

def set_partitions(S):
    """
    计算集合 S 的所有可能划分。

    输入: S (list 或 np.array，例如 [1, 2, 3])
    输出: P (list of lists of lists/sets)。每个外部列表元素是一个划分，
          每个划分是包含子集的列表。

    示例:
    set_partitions([1, 2]) ->
    [
        [[1], [2]],  # 情况1: [1] 自成一组, [[2]]
        [[1, 2]]     # 情况2: [1] 加入到 [2]
    ]
    """
    # 确保输入是列表，方便索引操作
    S = list(S)

    if not S:
        # 终止条件：空集只有一个划分：空列表
        return [[]]

    # 递归步骤
    first = S[0]
    rest = S[1:]

    # 递归调用获取剩余元素的所有划分
    rest_parts = set_partitions(rest)

    P = []

    for part in rest_parts:
        # part 是 rest 的一个划分，例如 [[2], [3]]

        # 1. 情况1：first 自成一组
        # 将 [first] 和 part 合并
        # MATLAB: [{first}, part] -> Python: [[first]] + part
        P.append([[first]] + part)

        # 2. 情况2：插入到已有组
        for j in range(len(part)):
            # part[j] 是 part 中的一个子集，例如 [2]

            # 创建新的划分，避免修改原 part
            new_part = [sub.copy() for sub in part]

            # 将 first 加入到第 j 个子集
            # MATLAB: newpart{j} = [newpart{j}, first]
            new_part[j].append(first)

            # 将新的划分添加到结果列表
            P.append(new_part)

    return P

if __name__ == '__main__':
    S_example = [1, 2, 3]
    partitions = set_partitions(S_example)
    for p in partitions:
        print(p)