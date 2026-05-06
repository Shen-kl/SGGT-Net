import torch
import torch.nn as nn
from torch_scatter import scatter_mean # PyG常用的依赖库


@torch.no_grad()
def minmax_scaler(graph_data, graph_data_norm, config, type='baseline'):
    '''
    基于固定尺度的 Min-Max 归一化函数，用于群目标追踪数据。
    :input
    graph_data 包含 x [node_num, T, state_dim]
    y [node_num, 1, state_dim]
    edge_feature_target [node_num, 1, space_dim]
    edge_feature_input [node_num, T, space_dim]
    measurement [node_num, T, space_dim]
    residual [node_num, T, space_dim]
    :return:
    '''

    num_graphs = graph_data.batch.max().item() + 1

    # 深度复制数据，避免修改原始输入
    # graph_data_norm = graph_data.clone()
    x = graph_data.x.clone()
    y = graph_data.y.clone()
    edge_features = [t.clone() for t in graph_data.edge_features]
    tar_edge_features = [t.clone() for t in graph_data.tar_edge_features]
    measurement = graph_data.measurement.clone()
    if type == 'sggt_net':
        residual = graph_data.residual.clone()
        measurement_future = graph_data.measurement_future.clone()

    # 计算 x 与 y 在位置上的质心
    # 使用scatter_mean 在不同的图中找最后一个时刻的群的质心
    if config['MODE'] == 1:
        centroids_x = scatter_mean(x[:, -1, [0, 2]], graph_data.batch, dim=0, dim_size=num_graphs)
    elif config['MODE'] == 2:
        centroids_x = scatter_mean(measurement[:, -1, :], graph_data.batch, dim=0, dim_size=num_graphs)

    # 将质心广播回每一个节点 输入与真值使用同一个质心
    T_len = x.shape[1]
    centroids_per_node_x = centroids_x[graph_data.batch].unsqueeze(1).expand(-1, T_len, -1)
    centroids_per_node_y = centroids_x[graph_data.batch].unsqueeze(1)

    # 执行去质心操作(Localization)
    x[:, :, [0, 2]] = x[:, :, [0, 2]] - centroids_per_node_x
    y[:, :, [0, 2]] = y[:, :, [0, 2]] - centroids_per_node_y
    measurement = measurement - centroids_per_node_x
    if type == 'sggt_net':
        measurement_future = measurement_future - centroids_per_node_y
    # 位置缩放
    x[:, :, [0, 2]] = x[:, :, [0, 2]] / config['S_POS']
    y[:, :, [0, 2]] = y[:, :, [0, 2]] / config['S_POS']
    measurement = measurement / config['S_POS']
    if type == 'sggt_net':
        measurement_future = measurement_future / config['S_POS']
    # 速度缩放
    x[:, :, [1, 3]] = x[:, :, [1, 3]] / config['S_VEL']
    y[:, :, [1, 3]] = y[:, :, [1, 3]] / config['S_VEL']

    # 边特征归一化
    scale_edge = torch.tensor([config['S_EDGE_P_DIST'], config['S_EDGE_V_DIST']], device=x.device)
    edge_features = [t/ scale_edge for t in edge_features]
    tar_edge_features = [t / scale_edge for t in tar_edge_features]

    if type == 'sggt_net':
        residual = residual / config['S_RES']

    if config['MODE'] == 1:
        graph_data_norm.x = x
    elif config['MODE'] == 2: # 模式2 使用量测代替输入的位置
        x[:, :, [0, 2]] = measurement
        graph_data_norm.x = x


    graph_data_norm.y = y
    graph_data_norm.edge_features = edge_features
    graph_data_norm.tar_edge_features = tar_edge_features
    graph_data_norm.measurement = measurement

    if type == 'sggt_net':
        graph_data_norm.residual = residual
        graph_data_norm.measurement_future = measurement_future

    # [x,vx,y,vy] -> [x,y,vx,vy]
    graph_data_norm.y = graph_data_norm.y[..., [0, 2, 1, 3]]
    graph_data_norm.x = graph_data_norm.x[..., [0, 2, 1, 3]]
    graph_data_norm.nan_mask = graph_data.nan_mask
    graph_data_norm.real_mask = graph_data.real_mask
    graph_data_norm.param_change_time = graph_data.param_change_time
    graph_data_norm.edge_index = graph_data.edge_index
    graph_data_norm.tar_edge_index = graph_data.tar_edge_index
    return graph_data_norm, centroids_x



