from data.load_data import *
import torch
import numpy as np
from loguru import logger
from model.sggt_net_optimized import SGGT_Net
from train.trainer_optimized import *
from torch_geometric.loader import DataLoader
from config import *
from model.motion_models import *
from model.motion_models_base import *



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = Args().get_parser()
    setup_seed(args.seed)
    logger.add(args.log_dir)
    config = CONFIG
    # 数据集加载
    dataset_str = args.data_dir
    train_dataset = GroupTargetTrackingDataset_load_whole_trajectory(dataset_str,'train')
    test_dataset = GroupTargetTrackingDataset_load_whole_trajectory(dataset_str,'test')

    # 按照batch size 划分
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0)

    # Pure integrators
    if args.motion_model == '1Xint':
        m_model = SingleIntegrator(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)
    elif args.motion_model == '2Xint':
        m_model = DoubleIntegrator(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)
    elif args.motion_model == '3Xint':
        m_model = TripleIntegrator(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)

    # Orientation-based
    elif args.motion_model == 'singletrack':
        m_model = KinematicSingleTrack(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)
        if args.init_static:
            static_f_dim = 2
    elif args.motion_model == 'unicycle':
        m_model = Unicycle(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)
    elif args.motion_model == 'curvature':
        m_model = Curvature(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures)
    elif args.motion_model == 'curvilinear':
        m_model = CurviLinear(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures, u1_lim=args.u1_lim)

    # Neural ODEs
    elif args.motion_model == 'neuralode':
        m_model = FirstOrderNeuralODE(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures,
                                      static_f_dim=0, n_hidden=args.n_ode_hidden,
                                      n_layers=args.n_ode_layers)
    elif args.motion_model == '2Xnode':
        m_model = SecondOrderNeuralODE(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures,
                                       static_f_dim=0, n_hidden=args.n_ode_hidden,
                                       n_layers=args.n_ode_layers)
    elif args.motion_model == 'group_residual_without_MCU':
        m_model = SecondOrderNeuralODE_groupTrack_without_MCU(solver=args.ode_solver, dt=args.T,
                                                              mixtures=args.n_mixtures,
                                                              static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                              n_layers=args.n_ode_layers)
    elif args.motion_model == 'group_residual_without_struct':
        m_model = SecondOrderNeuralODE_groupTrack_without_struct(solver=args.ode_solver, dt=args.T,
                                                                 mixtures=args.n_mixtures,
                                                                 static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                                 n_layers=args.n_ode_layers)
    elif args.motion_model == 'group_residual_without_struct_MCU':
        m_model = SecondOrderNeuralODE_groupTrack_without_struct_MCU(solver=args.ode_solver, dt=args.T,
                                                                     mixtures=args.n_mixtures,
                                                                     static_f_dim=0, n_hidden=args.n_ode_hidden,
                                                                     n_layers=args.n_ode_layers)
    else:
        m_model = SecondOrderNeuralODE_groupTrack(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures,
                                       static_f_dim=0, n_hidden=args.n_ode_hidden,
                                       n_layers=args.n_ode_layers)
    # 模型
    track_model = SGGT_Net(args.encoder_input_size, args.encoder_hidden_size,
                                 args.encoder_n_heads,
                                 args.encoder_n_layers, m_model.mixtures, args.encoder_dropout,
                                 args.encoder_gnn_layer, args.encoder_use_edge_features,
                                 m_model, args.decoder_max_length, args.decoder_hidden_size,
                                 args.decoder_n_heads, args.decoder_n_layers, args.decoder_alpha,
                                 args.decoder_dropout, args.decoder_residual_length,
                                 args.decoder_z_dimension, args.decoder_gnn_layer,
                                 args.decoder_use_MCU, args.decoder_use_struct, args.T)



    track_model_optimizer = torch.optim.Adam(track_model.parameters(), lr=args.lr)

    track_model_lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(track_model_optimizer,
                                                                         factor=args.optimizer_factor,
                                                                         patience=args.optimizer_patience)
    total_params = sum(p.numel() for p in track_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (Method 1): {total_params}")

    # 训练
    train_and_evaluation(track_model, train_dataloader, test_dataloader, track_model_optimizer,
                    track_model_lr_schedule, logger, config, args)

