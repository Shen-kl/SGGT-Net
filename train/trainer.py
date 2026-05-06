import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
from utils.minmax_scaler import *
from utils.losses import *
import time
from config import CONFIG
import numpy as np
import psutil, os, torch
from scipy.special import expit  # 就是 sigmoid
from utils.common.caculate_measurement_noise_cov import calculate_measurement_noise_cov

def kalman_nll(innovation, S):
    """
    innovation: (N,T,k)
    S: (N,T,k,k)
    """
    S = S.unsqueeze(1)
    L = torch.linalg.cholesky(S)

    # Mahalanobis
    solve = torch.cholesky_solve(
        innovation.unsqueeze(-1), L
    ).squeeze(-1)

    maha = (innovation * solve).sum(-1)

    logdet = 2 * torch.log(
        torch.diagonal(L, dim1=-2, dim2=-1)
    ).sum(-1)

    return torch.mean(0.5 * (maha + logdet))

def enu2aer(enu_x, enu_y, enu_z):
    azi = torch.atan2(enu_x,enu_y) * 180 / torch.pi
    elev = torch.atan2(enu_z, (enu_x**2+enu_y**2)**0.5)* 180 / torch.pi
    range = (enu_x**2+enu_y**2+enu_z**2)**0.5

    return azi, elev, range

def graph_data_input(graph_data, CONFIG, args):
    # 初始化图数据
    # 根据输入窗长度大小 拷贝一个输入
    batch = graph_data.x.shape[0]
    graph_data_tmp = graph_data.clone()
    graph_data_tmp.x = graph_data_tmp.x[:, :CONFIG['input_graph_window_len'], :]
    graph_data_tmp.y = graph_data_tmp.y[:, CONFIG['input_graph_window_len']:CONFIG['input_graph_window_len'] + CONFIG[
        'output_graph_window_len'], :]
    graph_data_tmp.measurement = graph_data_tmp.measurement[:,
                                 :CONFIG['input_graph_window_len'], :]

    graph_data_tmp.measurement_future = graph_data.measurement[:,
                                        CONFIG['input_graph_window_len']:CONFIG['input_graph_window_len'] + CONFIG[
                                            'output_graph_window_len'], :]

    # graph_data_tmp.measurement_future = graph_data.y[:,
    #                                     CONFIG['input_graph_window_len']:CONFIG['input_graph_window_len'] + CONFIG[
    #                                         'output_graph_window_len'], [0,2]]

    graph_data_tmp.residual = torch.cat([torch.zeros([batch, np.max(args.decoder_residual_length) -
                                                      CONFIG['input_graph_window_len'], CONFIG['z_dimension']]),
                                          graph_data_tmp.measurement - graph_data_tmp.x[:,:,[0,2]]],dim=1)

    graph_data_tmp.edge_index = graph_data_tmp.edge_index[:CONFIG['input_graph_window_len']]
    graph_data_tmp.edge_features = graph_data_tmp.edge_features[:CONFIG['input_graph_window_len']]

    graph_data_tmp.tar_edge_index = graph_data_tmp.tar_edge_index[CONFIG['input_graph_window_len']:
                                                                  CONFIG['input_graph_window_len'] +
                                                                  CONFIG['output_graph_window_len']]
    graph_data_tmp.tar_edge_features = graph_data_tmp.tar_edge_features[CONFIG['input_graph_window_len']:
                                                                        CONFIG['input_graph_window_len'] +
                                                                        CONFIG['output_graph_window_len']]
    graph_data_tmp.nan_mask = graph_data_tmp.nan_mask[:, :CONFIG['input_graph_window_len'], :]
    graph_data_tmp.real_mask = graph_data_tmp.real_mask[:, CONFIG['input_graph_window_len']:
                                                        CONFIG['input_graph_window_len'] +
                                                        CONFIG['output_graph_window_len'],:]
    return graph_data_tmp

def graph_data_update(graph_data_tmp, graph_data, CONFIG, x_update_renorm, bias, args,use_ground_truth=False):
    batch = graph_data.x.shape[0]
    graph_data.x[:,bias + CONFIG['input_graph_window_len'] - 1,:] = x_update_renorm.squeeze()

    graph_data_tmp.x[:,:-1,:] = graph_data_tmp.x[:,1:,:]
    graph_data_tmp.x[:,-1,:] = x_update_renorm.squeeze(dim=1)
    # graph_data_tmp.x = torch.cat([graph_data_tmp.x[:, :-1, :], x_update_renorm], dim=1)

    # graph_data_tmp.x = graph_data.x[:, bias:CONFIG['input_graph_window_len'] + bias, :]
    graph_data_tmp.y = graph_data.y[:, bias +
                                           CONFIG['input_graph_window_len']:bias + CONFIG['input_graph_window_len'] +
                                                                            CONFIG[
                                                                                'output_graph_window_len'], :]

    # graph_data_tmp.x =  graph_data.y[:, bias :bias + CONFIG['input_graph_window_len'], :]  # 测试用 将输入改为真值

    graph_data_tmp.measurement = graph_data.measurement[:,
                                 bias:CONFIG['input_graph_window_len'] + bias, :]
    graph_data_tmp.measurement_future = graph_data.measurement[:,bias +
                                           CONFIG['input_graph_window_len']:bias + CONFIG['input_graph_window_len'] +
                                                                            CONFIG[
                                                                                'output_graph_window_len'], :]

    # graph_data_tmp.measurement_future = graph_data.y[:,bias +
    #                                        CONFIG['input_graph_window_len']:bias + CONFIG['input_graph_window_len'] +
    #                                                                         CONFIG[
    #                                                                             'output_graph_window_len'], [0,2]]

    if bias + CONFIG['input_graph_window_len'] < np.max(args.decoder_residual_length):
        # graph_data_tmp.residual = torch.cat([torch.zeros([batch, np.max(args.decoder_residual_length) -
        #                                                   CONFIG['input_graph_window_len'] - bias, CONFIG['z_dimension']]),
        #                                      graph_data.measurement[:,:CONFIG['input_graph_window_len'] + bias,:] -
        #                                      graph_data.x[:, :CONFIG['input_graph_window_len'] + bias, [0, 2]] ], dim=1)

        zeros_len = np.max(args.decoder_residual_length) - CONFIG['input_graph_window_len'] - bias
        graph_data_tmp.residual[:, :zeros_len, :].zero_()
        graph_data_tmp.residual[:, zeros_len:, :] = graph_data.measurement[:,:CONFIG['input_graph_window_len'] + bias,:] - \
                                             graph_data.x[:, :CONFIG['input_graph_window_len'] + bias, [0, 2]]
    else:
        begin_idx = bias + CONFIG['input_graph_window_len'] -  np.max(args.decoder_residual_length)
        data = (graph_data.measurement[:,begin_idx:CONFIG['input_graph_window_len'] + bias,:] -
                                   graph_data.x[:, begin_idx:CONFIG['input_graph_window_len'] + bias, [0, 2]])
        graph_data_tmp.residual[:, -data.shape[1]:, :] = data

    graph_data_tmp.edge_index = graph_data.edge_index[bias:CONFIG['input_graph_window_len'] + bias]
    graph_data_tmp.edge_features = graph_data.edge_features[bias:CONFIG['input_graph_window_len'] + bias]

    graph_data_tmp.tar_edge_index = graph_data.tar_edge_index[bias + CONFIG['input_graph_window_len']:
                                                              bias + CONFIG['input_graph_window_len'] +
                                                              CONFIG['output_graph_window_len']]
    graph_data_tmp.tar_edge_features = graph_data.tar_edge_features[
                                       bias + CONFIG['input_graph_window_len']:
                                       bias + CONFIG['input_graph_window_len'] +
                                       CONFIG['output_graph_window_len']]
    graph_data_tmp.nan_mask = graph_data.nan_mask[:, bias:CONFIG['input_graph_window_len'] + bias, :]
    graph_data_tmp.real_mask = graph_data.real_mask[:, bias + CONFIG['input_graph_window_len']:
                                                       bias + CONFIG['input_graph_window_len'] +
                                                       CONFIG['output_graph_window_len'], :]


    return graph_data_tmp

def init_P_from_R(measurement_noise_matrix, delta_T, device=None):

    R = measurement_noise_matrix

    P_matlab = torch.zeros(measurement_noise_matrix.shape[0], 4, 4, device=device)

    P_matlab[...,0,0] = R[...,0,0]
    P_matlab[...,0,1] = R[...,0,0] / delta_T
    P_matlab[...,0,2] = R[...,0,1]
    P_matlab[...,0,3] = R[...,0,1] / delta_T

    P_matlab[...,1,0] = R[...,0,0] / delta_T
    P_matlab[...,1,1] = 2 * R[...,0,0] / delta_T**2
    P_matlab[...,1,2] = R[...,0,1] / delta_T
    P_matlab[...,1,3] = 2 * R[...,0,1] / delta_T**2

    P_matlab[...,2,0] = R[...,0,1]
    P_matlab[...,2,1] = R[...,0,1] / delta_T
    P_matlab[...,2,2] = R[...,1,1]
    P_matlab[...,2,3] = R[...,1,1] / delta_T

    P_matlab[...,3,0] = R[...,0,1] / delta_T
    P_matlab[...,3,1] = 2 * R[...,0,1] / delta_T**2
    P_matlab[...,3,2] = R[...,1,1] / delta_T
    P_matlab[...,3,3] = 2 * R[...,1,1] / delta_T**2

    T = torch.tensor([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ], dtype=P_matlab.dtype, device=device)

    return T @ P_matlab @ T.T

def train_and_evaluation(model:nn.Module, train_dataloader:DataLoader, test_dataloader:DataLoader,
                         optimizer, lr_schedule, logger, config, args):
    process = psutil.Process(os.getpid())

    # 开始训练
    train_epoches = int(100) # 训练轮数
    wta_epochs = int(train_epoches //20)
    warm_epochs = int(train_epoches - (train_epoches // 8))

    P_chaneg_epochs = int(train_epoches // 3)

    update_stage1_epochs = int(train_epoches // 10)
    update_stage2_epochs = int(train_epoches // 10 * 2)
    update_stage3_epochs = int(train_epoches // 10 * 3)
    update_stage4_epochs = int(train_epoches // 10 * 4)


    WTA_LOSS = EWTALoss()
    NLL_LOSS = NLLMDNLoss()
    NLL_LOSS_single = LossCompute_NLL()
    min_evaluation = 1e9

    # model.load_state_dict(
    #     torch.load('./checkpoint/2026_03_02_13_40_.pth', map_location=torch.device('cpu'))['state_dict'])

    tf_prob_start = 1.0
    tf_prob_min = 0.0
    tf_decay_epochs = 20

    delta_epoch = 0

    D = torch.diag_embed(1.0 / torch.tensor([CONFIG['S_POS'], 1.0 / CONFIG['S_POS'], 1.0 / CONFIG['S_VEL'], 1.0 / CONFIG['S_VEL']]))

    for current_epoch in range(train_epoches):
        print("RAM:", process.memory_info().rss / 1024 ** 3, "GB")
        print("GPU:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")

        train_loss_set = []
        validation_loss_set = []
        location_rmse_prediction_set = []
        velocity_rmse_prediction_set = []

        location_rmse_update_set = []
        velocity_rmse_update_set = []

        f1_struct_set = []

        tf_prob = max(tf_prob_min,
                      tf_prob_start - current_epoch / tf_decay_epochs) # 计算教师概率

        # 训练集
        model.train()
        with tqdm(total=len(train_dataloader), desc=f'{current_epoch}/{train_epoches}') as pbar:
            for batch_idx, graph_data in enumerate(train_dataloader):
                # print(f'batch_idx:{batch_idx}')
                # 滑窗处理数据
                time_Total = graph_data.y.shape[1] # 跟踪总时长

                # 根据输入窗长度大小 拷贝一个输入
                graph_data_tmp = graph_data_input(graph_data, CONFIG, args)
                graph_data_norm = graph_data_tmp.clone()  # 准备归一化版本

                # 存储滑窗结果
                single_batch_loss_set = []
                single_batch_location_rmse_prediction_set = []
                single_batch_velocity_rmse_prediction_set = []
                single_batch_location_rmse_update_set = []
                single_batch_velocity_rmse_update_set = []

                single_batch_location_rmse_measurement_set = []

                single_batch_f1_set = []

                # 计算R矩阵
                azi, elev, distance = enu2aer(graph_data.measurement[:,0,0], graph_data.measurement[:,0,1], torch.zeros_like(graph_data.measurement[:,0,0]))
                mu, measureNoiseCov = calculate_measurement_noise_cov(distance, azi, elev, 5, 0.2*torch.pi/180, 0.2*torch.pi/180)
                measureNoiseCov = torch.from_numpy(measureNoiseCov).permute(2,0,1)[:, : CONFIG['z_dimension'],: CONFIG['z_dimension']]

                # 计算P矩阵
                P_predict = init_P_from_R(measureNoiseCov, args.T)

                # 归一化两个矩阵
                measureNoiseCov_norm = measureNoiseCov / (CONFIG['S_POS'])**2
                P_predict_norm = P_predict
                P_predict_norm[..., [0, 1], :] = P_predict_norm[..., [0, 1], :] / (CONFIG['S_POS'])
                P_predict_norm[..., :, [0, 1]] = P_predict_norm[..., :, [0, 1]] / (CONFIG['S_POS'])
                P_predict_norm[..., [2, 3], :] = P_predict_norm[..., [2, 3], :] / (CONFIG['S_VEL'])
                P_predict_norm[..., :, [2, 3]] = P_predict_norm[..., :, [2, 3]] / (CONFIG['S_VEL'])

                P_predict = P_predict_norm * 1e-1
                encoder_hidden = None
                # 开始滑窗
                for time_step in range(CONFIG['input_graph_window_len'], time_Total - CONFIG['output_graph_window_len']):
                    # 对数据进行归一化处理
                    graph_data_norm, centroids_x = minmax_scaler(graph_data_tmp, graph_data_norm, config, args.neural_net)

                    # 输入网络
                    (all_states_prediction, all_Ps_prediction, mixture_coeffs, dec_mask, target,
                     x_update, P_update, encoder_hidden, A_vec, all_A_vec_mask, gate_final) = model(
                        graph_data_norm, tf_prob, P_predict, encoder_hidden, measureNoiseCov_norm)

                    # 计算predictor 损失
                    if current_epoch + delta_epoch < wta_epochs and wta_epochs > 1:
                        # Only EWTA loss
                        #  The number of winners used for the WTA loss decreases with increasing epoch
                        n_mixtures = mixture_coeffs.shape[1]
                        loss_1 = WTA_LOSS(all_states_prediction, target, dec_mask, n_mixtures)
                        loss_predict = loss_1
                    else:
                        # Compute NLL loss
                        n_mixtures = mixture_coeffs.shape[1]
                        if current_epoch + delta_epoch < warm_epochs and warm_epochs > 1: #current_epoch < warm_epochs and warm_epochs > 1:
                            # WTA + NLL loss
                            #  Update the weight by which the warm-up criterion should be multiplied
                            # Compute WTA loss
                            wta_loss_1 = WTA_LOSS(all_states_prediction, target, dec_mask, n_mixtures)
                            wta_loss = wta_loss_1
                            # Combine losses
                            loss_predict = wta_loss
                        else:
                            loss_predict = wta_loss

                    # 更新协方差矩阵
                    if current_epoch + delta_epoch < (wta_epochs + 0) and P_chaneg_epochs > 1:
                        P_predict = P_predict_norm * 1e-1
                    else:
                         P_predict = P_update.detach()

                    # updater 损失
                    L_mse_pos = WTA_LOSS(x_update.unsqueeze(1), target[:, 0, :].unsqueeze(1),
                                             dec_mask[:, 0].unsqueeze(-1), 1)
                    loss_update = L_mse_pos

                    # 计算 分类损失
                    if time_step < graph_data.param_change_time[0, -2]:
                        use_loss_flag = (torch.rand(1) < 0.2)
                    else:
                        use_loss_flag = (torch.rand(1) < 1)
                    if use_loss_flag:
                        struct_loss = focal_loss(A_vec, graph_data.struct_feat[time_step],all_A_vec_mask)
                    else:
                        struct_loss = 0

                    loss = loss_predict + loss_update + struct_loss

                    # 计算评价指标 将估计结果反归一化 与真值做对比
                    (location_rmse_prediction, velocity_rmse_prediction, location_rmse_update, velocity_rmse_update,
                    x_update_renorm,location_rmse_measurement) = caculate_evaluation(graph_data_tmp, all_states_prediction, x_update, centroids_x, config, 1)

                    # 计算分类评价指标
                    f1_struct, precision_struct, recall_struct = edge_f1_score(A_vec.detach(), graph_data.struct_feat[time_step])

                    param_times = graph_data.param_change_time[0, :-1]
                    if not ((param_times >= time_step) & (
                            param_times < time_step + CONFIG['output_graph_window_len'])).any():
                        # 如果预测的窗口内 存在航迹机动切换帧 则 loss 不参与反向传播
                        # 梯度归零
                        optimizer.zero_grad()
                        # 反向传播
                        loss.backward()
                        # 更新
                        optimizer.step()

                    train_loss_set.append(loss.item())

                    # 将预测结果作为下一帧的输入
                    bias = time_step - CONFIG['input_graph_window_len'] + 1
                    graph_data_tmp = graph_data_update(graph_data_tmp, graph_data, CONFIG, x_update_renorm,
                                                       bias, args,None)

                    single_batch_loss_set.append(loss.item())
                    single_batch_location_rmse_prediction_set.append(location_rmse_prediction)
                    single_batch_velocity_rmse_prediction_set.append(velocity_rmse_prediction)
                    single_batch_location_rmse_update_set.append(location_rmse_update)
                    single_batch_velocity_rmse_update_set.append(velocity_rmse_update)
                    single_batch_location_rmse_measurement_set.append(location_rmse_measurement)
                    single_batch_f1_set.append(f1_struct)
                pbar.set_postfix({
                    'loss': np.mean(single_batch_loss_set),
                    'location_prediction_rmse':np.mean(single_batch_location_rmse_prediction_set),
                    'velocity_prediction_rmse':np.mean(single_batch_velocity_rmse_prediction_set),
                    'location_update_rmse': np.mean(single_batch_location_rmse_update_set),
                    'velocity_update_rmse': np.mean(single_batch_velocity_rmse_update_set),
                    'location_measurement_rmse': np.mean(single_batch_location_rmse_measurement_set),
                    'f1': np.mean(single_batch_f1_set),
                })
                pbar.update(1)

        # 测试集
        model.eval()
        with ((torch.no_grad())):
            with tqdm(total=len(test_dataloader), desc=f'{current_epoch}/{train_epoches}') as pbar:
                for graph_data in test_dataloader:
                    # 滑窗处理数据
                    time_Total = graph_data.y.shape[1]  # 跟踪总时长
                    # 根据输入窗长度大小 拷贝一个输入
                    graph_data_tmp = graph_data_input(graph_data, CONFIG, args)
                    graph_data_norm = graph_data_tmp.clone()  # 准备归一化版本
                    # 存储滑窗结果
                    single_batch_loss_set = []
                    single_batch_location_rmse_prediction_set = []
                    single_batch_velocity_rmse_prediction_set = []
                    single_batch_location_rmse_update_set = []
                    single_batch_velocity_rmse_update_set = []

                    single_batch_location_rmse_measurement_set = []

                    single_batch_f1_set = []
                    # 开始滑窗
                    # 计算R矩阵
                    azi, elev, distance = enu2aer(graph_data.measurement[:, 0, 0], graph_data.measurement[:, 0, 1],
                                                  torch.zeros_like(graph_data.measurement[:, 0, 0]))
                    mu, measureNoiseCov = calculate_measurement_noise_cov(distance, azi, elev, 5, 0.2 * torch.pi / 180,
                                                                          0.2 * torch.pi / 180)
                    measureNoiseCov = torch.from_numpy(measureNoiseCov).permute(2, 0, 1)[:, : CONFIG['z_dimension'],
                                      : CONFIG['z_dimension']]

                    # 计算P矩阵
                    P_predict = init_P_from_R(measureNoiseCov, args.T)

                    # 归一化两个矩阵
                    measureNoiseCov_norm = measureNoiseCov / (CONFIG['S_POS']) ** 2
                    P_predict_norm = P_predict
                    P_predict_norm[..., [0, 1], :] = P_predict_norm[..., [0, 1], :] / (CONFIG['S_POS'])
                    P_predict_norm[..., :, [0, 1]] = P_predict_norm[..., :, [0, 1]] / (CONFIG['S_POS'])
                    P_predict_norm[..., [2, 3], :] = P_predict_norm[..., [2, 3], :] / (CONFIG['S_VEL'])
                    P_predict_norm[..., :, [2, 3]] = P_predict_norm[..., :, [2, 3]] / (CONFIG['S_VEL'])

                    P_predict = P_predict_norm * 1e-1
                    encoder_hidden = None
                    for time_step in range(CONFIG['input_graph_window_len'], time_Total - CONFIG['output_graph_window_len']):
                        # 对数据进行归一化处理
                        graph_data_norm, centroids_x = minmax_scaler(graph_data_tmp, graph_data_norm, config, args.neural_net)

                        # 输入网络
                        (all_states_prediction, all_Ps_prediction, mixture_coeffs, dec_mask, target,
                        x_update, P_update, encoder_hidden, A_vec,all_A_vec_mask, gate_final) = \
                        model(graph_data_norm, 0.0, P_predict, encoder_hidden, measureNoiseCov_norm)

                        if current_epoch + delta_epoch < (wta_epochs + 0) and P_chaneg_epochs > 1:
                            P_predict = P_predict_norm * 1e-1
                        else:
                            P_predict = P_update.detach()

                        #P_predict = P_update.detach()
                        # 计算损失
                        loss_1 = WTA_LOSS(all_states_prediction, target, dec_mask)

                        loss_predict = loss_1

                        loss = loss_predict

                        # 计算评价指标 将估计结果反归一化 与真值做对比
                        (location_rmse_prediction, velocity_rmse_prediction, location_rmse_update, velocity_rmse_update,
                         x_update_renorm, location_rmse_measurement) = caculate_evaluation(graph_data_tmp, all_states_prediction, x_update,
                                                                centroids_x, config, 1)

                        # 计算分类评价指标
                        f1_struct, precision_struct, recall_struct = edge_f1_score(A_vec.detach(),
                                                                                   graph_data.struct_feat[time_step])


                        # 将预测结果作为下一帧的输入
                        bias = time_step - CONFIG['input_graph_window_len'] + 1
                        graph_data_tmp = graph_data_update(graph_data_tmp, graph_data, CONFIG, x_update_renorm, bias, args)

                        single_batch_loss_set.append(loss.item())
                        single_batch_location_rmse_prediction_set.append(location_rmse_prediction)
                        single_batch_velocity_rmse_prediction_set.append(velocity_rmse_prediction)
                        single_batch_location_rmse_update_set.append(location_rmse_update)
                        single_batch_velocity_rmse_update_set.append(velocity_rmse_update)
                        single_batch_location_rmse_measurement_set.append(location_rmse_measurement)
                        single_batch_f1_set.append(f1_struct)

                        validation_loss_set.append(loss.item())
                        location_rmse_prediction_set.append(location_rmse_prediction)
                        velocity_rmse_prediction_set.append(velocity_rmse_prediction)
                        location_rmse_update_set.append(location_rmse_update)
                        f1_struct_set.append(f1_struct)

                    pbar.set_postfix({
                        'loss': loss.item(),
                        'location_prediction_rmse': np.mean(single_batch_location_rmse_prediction_set),
                        'velocity_prediction_rmse': np.mean(single_batch_velocity_rmse_prediction_set),
                        'location_update_rmse': np.mean(single_batch_location_rmse_update_set),
                        'velocity_update_rmse': np.mean(single_batch_velocity_rmse_update_set),
                        'velocity_measurement_rmse': np.mean(single_batch_location_rmse_measurement_set),
                        'f1':np.mean(single_batch_f1_set)
                    })
                    pbar.update(1)
        lr_schedule.step(sum(train_loss_set) / len(train_loss_set))
        logger.debug("train_loss:{},index:{}".format(sum(train_loss_set) / len(train_loss_set),
                                                    current_epoch))
        logger.debug("validation_loss:{},index:{}".format(sum(validation_loss_set) / len(validation_loss_set),
                                                    current_epoch))
        logger.debug("location_prediction_rmse:{},index:{}".format(sum(location_rmse_prediction_set) / len(location_rmse_prediction_set),
                                                    current_epoch))
        logger.debug("velocity_prediction_rmse:{},index:{}".format(sum(velocity_rmse_prediction_set) / len(velocity_rmse_prediction_set),
                                                    current_epoch))
        logger.debug("location_update_rmse:{},index:{}".format(sum(location_rmse_update_set) / len(location_rmse_update_set),
                                                    current_epoch))
        logger.debug("velocity_update_rmse:{},index:{}".format(sum(velocity_rmse_update_set) / len(velocity_rmse_update_set),
                                                    current_epoch))
        logger.debug("f1:{},index:{}".format(sum(f1_struct_set) / len(f1_struct_set),
                                                    current_epoch))
        now = time.localtime()
        nowt = time.strftime("%Y_%m_%d_%H_%M_", now)
        evaluation = sum(location_rmse_prediction_set) / len(location_rmse_prediction_set) \
                          + sum(velocity_rmse_prediction_set) / len(velocity_rmse_prediction_set) \
                     + sum(location_rmse_update_set) / len(location_rmse_update_set) \
                     + sum(velocity_rmse_update_set) / len(velocity_rmse_update_set)

        if min_evaluation > evaluation or current_epoch % 25 == 0:
            if min_evaluation > evaluation:
                min_evaluation = evaluation
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': sum(validation_loss_set) / len(validation_loss_set),
                'lr_schedule': lr_schedule.state_dict()
            }, args.output_dir + 'GNN_groupTargetTracking/' + str(nowt) + ".pth")