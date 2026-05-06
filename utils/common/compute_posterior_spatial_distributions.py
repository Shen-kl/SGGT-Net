import numpy as np
from numpy import sum, log, sort, argsort, reshape

"""
 函数功能： 计算legacy PT 与 new PT 的后验分布
"""
def compute_posterior_spatial_distributions(radar, association_info, message_aq, message_bv, posterior_parameters):
    if radar.legacy_PT_num > 0:
        # 如果当前时刻存在legacy PT ,处理
        for tar_index in range(radar.legacy_PT_num):
            match radar.tracker_method:
                case 'BGM':
                    current_target = radar.legacy_PT_set[tar_index]
            
                    # 获取 LPT Likelihood 和 message_aq
                    # LPT_likelihood{tar_index}(:,:) 形状: (J+1) x K (J=plots, K=weights)
                    LPT_likelihood = association_info['LPT_likelihood'][tar_index]
                    # message_aq(tar_index,:) 形状: 1 x (J+1)
                    message_aq_row = message_aq[tar_index, :].reshape(-1, 1)  # (J+1) x 1
            
                    # 1. 计算分子 gm_weight_A (与量测关联)

                    # 原始权重： (J+1) x K
                    gm_weight_prior = posterior_parameters[tar_index]['gmWeight']
                    # LPT_likelihood: (J+1) x K
                    # message_aq_row: (J+1) x 1
            
                    # LPT_likelihood * gm_weight_prior: (J+1) x K (Broadcast multiplication)
                    # message_aq_row * gm_weight_prior: (J+1) x K (Broadcast multiplication)
            
                    # 这里的 MATLAB 乘法是元素级的：
                    # message_aq_row 必须广播到 LPT_likelihood 的 K 维度
                    # message_aq_row * LPT_likelihood: (J+1) x K
                    # (J+1) x K * 1 x K (Broadcast) -> (J+1) x K
            
                    # gm_weight_A 的形状应该是 (J+1) x K，对应每个量测/未检测，和每个先验分量的组合
                    gm_weight_A = message_aq_row * LPT_likelihood * gm_weight_prior
            
                    # 2. 计算分母的第二个分量 gm_weight_B (未检测且未生存)
                    # gm_weight_B = (1 - Ps * r) * message_aq(tar_index,1);
                    # message_aq(tar_index, 1) 是 message_aq 的第一个元素 (对应未检测)
                    gm_weight_B = (1 - radar.Ps * current_target.r) * message_aq[tar_index, 0]
            
                    # 3. 更新高斯权重 (Normalization)
            
                    # numberOfPosteriorComponents = numel(gm_weight_A);
                    numberOfPosteriorComponents = gm_weight_A.size
            
                    # MATLAB: reshape(gm_weight_A, 1, numberOfPosteriorComponents);
                    gm_weight_A_flat = gm_weight_A.flatten()  # 1 x (J+1)*K
            
                    # gm_weight_update = gm_weight_A ./ (sum(gm_weight_A) + gm_weight_B);
                    sum_gm_weight_A = sum(gm_weight_A_flat)
                    sum_denominator = sum_gm_weight_A + gm_weight_B
            
                    gm_weight_update = gm_weight_A_flat / sum_denominator
            
                    # 4. 更新存在概率 (r)
                    # radar.legacy_PT_set(tar_index).r = sum(gm_weight_update);
                    current_target.r = sum(gm_weight_update)
                    current_target.r_history.append(current_target.r)
            
                    # 5. 归一化权重 (用于修剪)
                    # weights = gm_weight_update / sum(gm_weight_update);
                    # 注意：如果 r > 1 (理论上不可能，但应避免), sum(gm_weight_update) 会大于 1。
                    # 这里我们使用更新后的 r 作为归一化因子，因为 sum(gm_weight_update) = r
                    weights = gm_weight_update / current_target.r
            
                    # 6. 权重排序和修剪 (Pruning)
            
                    # 权重从大到小排序
                    # numpy argsort 默认升序，我们需要降序
                    sortedIndices = argsort(weights)[::-1]
                    weights_sorted = weights[sortedIndices]
            
                    # 剔除低高斯权重部分 (Thresholding)
                    # significantComponents = weights > radar.gmWeightThreshold;
                    significantComponents = weights_sorted > radar.gmWeightThreshold
                    significantWeights = weights_sorted[significantComponents]
            
                    # 重新归一化和更新权重
                    current_target.gmWeight = significantWeights / sum(significantWeights)
                    sortedIndices_pruned = sortedIndices[significantComponents]
                    current_target.numberOfGmComponents = len(current_target.gmWeight)
            
                    # 7. 最大分量限制 (Capping)
                    max_comp = radar.maximumNumberOfGmComponents
                    if current_target.numberOfGmComponents > max_comp:
                        current_target.gmWeight = current_target.gmWeight[:max_comp]
                        current_target.gmWeight = current_target.gmWeight / sum(current_target.gmWeight)  # 再次归一化
                        sortedIndices_pruned = sortedIndices_pruned[:max_comp]
                        current_target.numberOfGmComponents = max_comp
            
                        # 8. 根据被选择的权重 ID 保留最终的高斯分量
            
                        # 获取被选择的后验分量集合 (Posterior Components)
                        # MATLAB: posterior_parameters(tar_index).gmComponents = reshape(posterior_parameters(tar_index).gmComponents(sortedIndices), 1, N);
            
                        # posterior_parameters 结构: 
                        # [tar_index] -> 2D 数组 (plots+1) x (weights) -> Component Object
                        # 我们需要将其展平，然后根据 sortedIndices_pruned 选择
            
                        # 展平后验参数列表: (J+1) * K 个分量
                    posterior_flat = [item for row in posterior_parameters[tar_index]['gmComponents'] for item in row]

                    # 选择并更新到目标对象
                    # gmComponents 必须存储为列表/数组，而不是 MATLAB 的 cell array
                    current_target.gmComponents = [posterior_flat[t] for t in sortedIndices_pruned]

                    # 10. 航迹状态聚合 (仅保留权重最高的第一个分量)
                    # X, P 状态通常是所有分量按权重加权平均得到，但这段 MATLAB 代码直接使用权重最高的第一个分量作为航迹的代表状态。
                    if current_target.numberOfGmComponents > 0:
                        best_component = current_target.gmComponents[0]
            
                        # radar.legacy_PT_set(tar_index).X(:,end) = radar.legacy_PT_set(tar_index).gmComponents{1}.X;
                        '''
                        更改：使用所有高斯混合分量的加权求和
                        '''

                        # import pdb;
                        # pdb.set_trace()

                        X = np.zeros([radar.x_dimension,1])
                        for idx in range(current_target.numberOfGmComponents):
                            X += current_target.gmComponents[idx].X * current_target.gmWeight[idx]

                        current_target.X[-1] = X  # 假设 X_state 存储最终状态
            
                        # radar.legacy_PT_set(tar_index).P = radar.legacy_PT_set(tar_index).gmComponents{1}.P;
                        current_target.P = best_component.P  # 假设 P_cov 存储最终协方差

                        # 记录关联上的量测
                        current_target.associate_plot_track.append(best_component.associate_plot_track)

                        current_target.residual_set.append(best_component.residual)

                        if best_component.residual > 2:
                            current_target.Q_coe = 1e2
                        else:
                            current_target.Q_coe = 1e0
                        '''
                        如果使用神经网络 只保留一个高斯混合分量
                        '''
                        if radar.use_neural_network and radar.neural_network['model_name'] != 'MCST' and len(
                                current_target.X) > radar.track_len_for_nn:
                            current_target.numberOfGmComponents = 1
                            current_target.gmComponents = current_target.gmComponents[:1]
                            current_target.gmWeight = current_target.gmWeight[:1]
                    else:
                        # 如果所有分量都被修剪，则该航迹可能在后续步骤中被删除
                        current_target.X_state = None
                        current_target.P_cov = None
            
                        # 11. 更新目标检测概率 (Pd)
                        # Pd 是所有分量 Pd 的加权平均
                    current_target.Pd = 0.0
                    for gm_index in range(current_target.numberOfGmComponents):
                        weight = current_target.gmWeight[gm_index]
                        comp_pd = current_target.gmComponents[gm_index].Pd['Pd']
                        current_target.Pd += weight * comp_pd


        # 处理 new PT
        for tar_index, new_PT in enumerate(radar.new_PT_set):
            match radar.tracker_method:
                case 'BGM':
                    gm_weight_A = association_info['NPT_likelihood'][tar_index] * message_bv[0, tar_index]
                    gm_weight_B = sum(message_bv[:, tar_index])

                    # 更新权重
                    gm_weight_update = gm_weight_A / (sum(gm_weight_A) + gm_weight_B)

                    # 更新存在概率
                    new_PT.r = sum(gm_weight_update)
                    new_PT.r_history.append(new_PT.r)

    return radar