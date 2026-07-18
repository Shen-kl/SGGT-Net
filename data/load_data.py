import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import os
import re
from collections import defaultdict


# 加载群目标运动数据集 加载打包好的数据集 一次性加载整条航迹
class GroupTargetTrackingDataset_load_whole_trajectory(Dataset):
    def __init__(self, dataset_path:str, mode:str):
        super().__init__()
        self.dataset_path = dataset_path
        self.mode = mode
        self.dataset_len = 0
        data_list = os.listdir(os.path.join(self.dataset_path, self.mode, 'target'))

        for name in data_list:
            self.dataset_len = self.dataset_len + 1

    def len(self):
        return self.dataset_len

    def get(self, idx):
        idx = int(idx)
        input_path = os.path.join(
            self.dataset_path,
            self.mode,
            'input',
            f'batch_data_{idx:05d}.pt'
        )

        target_path = os.path.join(
            self.dataset_path,
            self.mode,
            'target',
            f'batch_data_{idx:05d}.pt'
        )

        if not isinstance(input_path, str):
            raise TypeError(f"Invalid file path: {input_path}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)

        if not isinstance(target_path, str):
            raise TypeError(f"Invalid file path: {target_path}")

        if not os.path.exists(target_path):
            raise FileNotFoundError(target_path)

        # 加载模型输入
        input = torch.load(input_path)
        target = torch.load(target_path)

        graph_measurement_input = input['graph_measurement'].float()
        graph_estimation_input = input['graph_estimation'].float()
        nan_mask_input = input['nan_mask'].bool()
        graph_edge_idx_input = input['graph_input_edge_index']
        graph_edge_idx_input = [t.long() for t in graph_edge_idx_input]
        graph_edge_feat_input = input['graph_input_edge_feature']
        graph_edge_feat_input = [t.float() for t in graph_edge_feat_input]
        param_change_time = input['param_change_time'].float()
        # 加载真值
        graph_target = target['graph_target'].float()
        real_mask = target['real_mask'].bool()
        graph_edge_idx_target = target['graph_target_edge_index']
        graph_edge_idx_target = [t.long() for t in graph_edge_idx_target]
        graph_edge_feat_target = target['graph_target_edge_feature']
        graph_edge_feat_target = [t.float() for t in graph_edge_feat_target]
        graph_struct_feat_target = target['graph_struct_total']
        graph_struct_feat_target = [t.float().transpose(-1,-2) for t in graph_struct_feat_target]


        # 使用Data 封装
        data = Data(x=graph_estimation_input, edge_index=graph_edge_idx_input, edge_features=graph_edge_feat_input,
                    y=graph_target, tar_edge_index=graph_edge_idx_target, tar_edge_features=graph_edge_feat_target,
                    measurement=graph_measurement_input, nan_mask=nan_mask_input,
                    real_mask=real_mask, param_change_time=param_change_time,struct_feat=graph_struct_feat_target)
        return data


if __name__ == '__main__':
    dataset = GroupTargetTrackingDataset_load_whole_trajectory('D:/Dataset/GroupTargetsDataset_dataRate_2_v8_pack/','train')
    dataset.get(267)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

    max_residual = torch.tensor([0])
    for (hetero_graph) in dataloader:
        graph_target = hetero_graph.y
        graph_estimation_input = hetero_graph.x
        # residual = hetero_graph.residual
        # max_residual_tmp = torch.max(torch.abs(residual))
        # max_residual = torch.max(max_residual,max_residual_tmp)
        # print(graph_estimation_input)
        # print(graph_estimation_input.size())

    print(max_residual.item())
