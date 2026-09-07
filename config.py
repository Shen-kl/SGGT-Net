import torch.nn as nn
import torch
import numpy as np

CONFIG = {
    # 1. 位置缩放 (用于去质心后的局部坐标)
    # 目的：保留宏观几何结构，范围 [-1, 1]。例如：500米
    'S_POS': 1500.0,

    # 2. 速度缩放 (用于速度 Vxy)
    # 目的：保留速度物理意义。例如：50 m/s
    'S_VEL': 300.0,

    # 3. 残差缩放 (用于残差 Rxy)
    # 目的：保证高灵敏度，范围 [-1, 1]。例如：10 米
    'S_RES': 110.0,

    # 4. 边特征缩放 (欧氏距离是正数，范围 [0, 1])
    # 连接距离阈值。例如：50米
    'S_EDGE_P_DIST': 450.0,  # 150
    # 最大速度差。例如：20 m/s
    'S_EDGE_V_DIST': 150.0,  # 20

    # 5. 模式 1:位置信息使用预测值  2: 位置信息使用量测值(模型初步搭建时期的模式)
    'MODE': 1,

    # 6. 量测维度
    'z_dimension': 2,

    # 7. 状态向量维度
    'x_dimension': 4,

    # 8. 输入的历史窗口长度
    'input_graph_window_len': 8,

    # 9. 输出的窗口长度
    'output_graph_window_len': 1,

    # 10. 归一化偏移量
    'centroids_offset': 0,

    # 11. 量测噪声标准差
    'measurement_noise_std': [5, 0.2 * np.pi / 180, 0.2 * np.pi / 180],
}

activations = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'lrelu': nn.LeakyReLU,
}


import argparse
import torch

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(description='GNN')
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoint/',
                            help='the output dir for model checkpoints')
        parser.add_argument('--checkpoint', default='../checkpoint/gnn/2026_03_18_19_45_.pth', type=str,
                            help='Path to model checkpoint')
        parser.add_argument('--resume_checkpoint', default='', type=str,
                            help='Resume model, optimizer, scheduler, and epoch from a training checkpoint')
        parser.add_argument('--resume_reset_scheduler', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='Resume model/optimizer state but start a new scheduler from the requested LR')
        parser.add_argument('--init_checkpoint', default='', type=str,
                            help='Initialize compatible model weights without optimizer state')
        parser.add_argument('--data_dir', default='E:/Dataset/GroupTargetsDataset_dataRate_2_v8_pack/', type=str,
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='./log/training.log', type=str,
                            help='log dir for uer')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')
        parser.add_argument('--T', type=float, default=2,help='data rate')
        parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), #
                            help='cpu or gpu')
        parser.add_argument('--train_batch_size', default=4, type=int)
        parser.add_argument('--train_epochs', default=100, type=int,
                            help='Max training epoch')
        parser.add_argument('--test_epochs', default=10, type=int,
                            help='Max training epoch')
        parser.add_argument('--eval_batch_size', default=1, type=int)
        parser.add_argument('--loader_workers', default=0, type=int,
                            help='DataLoader workers; 0 minimizes Windows host-memory duplication')
        parser.add_argument('--loader_prefetch_factor', default=2, type=int)
        parser.add_argument('--pin_memory', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--debug_train_samples', default=0, type=int,
                            help='Limit training samples for smoke runs; 0 uses all samples')
        parser.add_argument('--debug_train_start', default=0, type=int,
                            help='Starting trajectory index for a limited training subset')
        parser.add_argument('--debug_val_samples', default=0, type=int,
                            help='Limit validation samples for smoke runs; 0 uses all samples')
        parser.add_argument('--debug_val_start', default=0, type=int,
                            help='Starting trajectory index for a limited validation subset')
        parser.add_argument('--skip_final_tests', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='Skip IID/OOD evaluation during diagnostic training')
        parser.add_argument('--debug_time_steps', default=0, type=int,
                            help='Limit sliding-window steps per trajectory; 0 uses all steps')
        parser.add_argument('--v3_windows_per_trajectory', default=20, type=int,
                            help='Scored causal windows sampled per v3 trajectory and epoch')
        parser.add_argument('--v3_residual_warmup', default=16, type=int,
                            help='Unscored causal steps used to fill the online residual buffer')
        parser.add_argument('--v3_window_counts', default=[8, 6, 6], nargs=3, type=int,
                            metavar=('STEADY', 'SWITCH', 'STRONG'))
        parser.add_argument('--full_time_supervision', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='Score every legal causal step during v3 training')
        parser.add_argument('--encoder_execution_mode', default='stream',
                            choices=['full', 'stream'],
                            help='Re-encode each window or update only the newest frame')
        parser.add_argument('--encoder_stream_reset_interval', default=0, type=int,
                            help='Rebuild the streaming encoder every N steps; 0 disables resets')
        parser.add_argument('--train_metric_interval', default=8, type=int,
                            help='Compute non-differentiable train metrics every N steps')
        parser.add_argument('--report_trajectory_quantiles', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='Report per-trajectory RMSE quantiles; requires eval batch size 1')
        parser.add_argument('--windowed_training', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='Train on sampled contiguous rollout windows')
        parser.add_argument('--rollout_steps', default=8, type=int,
                            help='Consecutive prediction/update steps per sampled window')
        parser.add_argument('--validation_burn_in_steps', default=4, type=int,
                            help='Unscored recurrent warm-up before fixed-window validation')
        parser.add_argument('--event_window_fraction', default=0.5, type=float,
                            help='Fraction of training windows sampled near parameter changes')
        parser.add_argument('--grad_clip_norm', default=10.0, type=float,
                            help='Clip the gradient norm after each rollout; 0 disables clipping')
        parser.add_argument('--optimizer_step_mode', default='trajectory_batch',
                            choices=['time_step', 'trajectory_batch'],
                            help='Update per supervised step or once after each trajectory batch')
        parser.add_argument('--finite_check_interval', default=0, type=int,
                            help='Full tensor/parameter finite scan interval; 0 disables costly scans')
        parser.add_argument('--initial_covariance_scale', default=1.0, type=float,
                            help='Scale physically initialized state covariance')
        parser.add_argument('--covariance_reset_epochs', default=0, type=int,
                            help='Legacy per-step covariance reset curriculum; 0 disables it')
        parser.add_argument('--early_stopping_patience', default=10, type=int,
                            help='Stop after this many epochs without validation-score improvement')
        parser.add_argument('--early_stopping_min_delta', default=1e-3, type=float,
                            help='Minimum composite validation-score improvement')
        parser.add_argument('--checkpoint_metric', default='composite',
                            choices=['composite', 'position', 'association_position',
                                     'prediction', 'expert',
                                     'expert_strong', 'gate'],
                            help='Metric used for early stopping and final checkpoint loading')
        parser.add_argument('--optimizer_factor', default=0.8, type=float)
        parser.add_argument('--optimizer_patience', default=5, type=int)
        parser.add_argument('--lr', default=6e-4, type=float,
                            help='maximum learning rate')
        parser.add_argument('--finetune_lr', default=1e-4, type=float,
                            help='learning rate after fusion-path adaptation')
        parser.add_argument('--fusion_adapt_epochs', default=0, type=int,
                            help='optional pretrained-backbone fusion adaptation; scratch spectral-SEAN training uses 0')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='AdamW weight decay')
        parser.add_argument('--warmup_epochs', default=5, type=int,
                            help='Linear learning-rate warmup epochs')
        parser.add_argument('--warmup_start_factor', default=0.1, type=float,
                            help='Warmup starts at this fraction of --lr')
        parser.add_argument('--min_lr', default=1e-5, type=float,
                            help='Minimum learning rate for cosine annealing')


        # train args
        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')
        parser.add_argument('--u1-lim', type=float, default=10,
                            help='u1 control limit')
        parser.add_argument('--u2-lim', type=float, default=10,
                            help='u2 control limit')
        parser.add_argument('--ode-solver', type=str, default='rk4',
                            help='choose explicit solver [ef | mp | heun | rk3 | ssprk3 | rk4 | dopri5 | impl_adam]')
        parser.add_argument('--n-mixtures', type=int, default=1,
                            help='number of mixtures (default: 8)')
        parser.add_argument('--motion-model', type=str, default='group',
                            help='choice of motion model (default: 2Xnode)')  
        parser.add_argument('--n-ode-hidden', type=int, default=64,
                            help='n graph layers (default: 16)') # residula: 64
        parser.add_argument('--n-ode-layers', type=int, default=3,
                            help='n ode layers (default: 1)') # residula: 3

        parser.add_argument('--encoder_input_size', default=4, type=int)
        parser.add_argument('--encoder_hidden_size', default=64, type=int)
        parser.add_argument('--encoder_n_heads', default=3, type=int,)
        parser.add_argument('--encoder_n_layers', default=2, type=int,)
        parser.add_argument('--encoder_n_mixtures', default=4, type=int)
        parser.add_argument('--encoder_dropout', default=0.1, type=float)
        parser.add_argument('--encoder_gnn_layer', default="graphconv", type=str)
        parser.add_argument('--encoder_use_edge_features', default=True,
                            action=argparse.BooleanOptionalAction)


        parser.add_argument('--decoder_motion_model', default='neuralode', type=str)
        parser.add_argument('--decoder_max_length', default=9, type=int) # 输入历史帧的长度加1
        parser.add_argument('--decoder_hidden_size', default=64, type=int,
                            help='(default: 64)')
        parser.add_argument('--decoder_n_heads', default=3, type=int)
        parser.add_argument('--decoder_n_layers', default=2, type=int)
        parser.add_argument('--decoder_alpha', default=0.2, type=float)  #
        parser.add_argument('--decoder_dropout', default=0.1, type=float)  #
        parser.add_argument('--decoder_gnn_layer', default="graphconv", type=str)  #
        parser.add_argument('--decoder_residual_length', default=[8, 16], nargs='+', type=int)  #
        parser.add_argument('--decoder_z_dimension', default=2, type=int)  #
        parser.add_argument('--decoder_node_num', default=200, type=int) # 允许跟踪的最大目标数量
        parser.add_argument('--decoder_use_GASF', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--decoder_use_SEAN', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--gasf_filter_mode', default='abc',
                            choices=['abc', 'learned'],
                            help='Parametric low-pass or learned graph-polynomial spectral filter')
        parser.add_argument('--gasf_graph_mode', default='graph',
                            choices=['graph', 'temporal_only'],
                            help='Use graph propagation after temporal filtering, or temporal filtering only')
        parser.add_argument('--gasf_recurrent_filter', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='use previous per-scale eta and gated ABC parameter recurrence')
        parser.add_argument('--gasf_recurrent_gate_initial_update', default=0.1,
                            type=float,
                            help='initial candidate fraction of the recurrent ABC update gate')
        parser.add_argument('--innovation_decomposition', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='use EKF-whitened, SEAN soft-graph common/individual innovation features')
        parser.add_argument('--innovation_whiten_clip', default=20.0, type=float,
                            help='robust upper bound for whitened innovation norm')
        parser.add_argument('--causal_innovation_correction', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='apply an 8-frame own/leave-one-out peer innovation prior correction')
        parser.add_argument('--causal_innovation_history', default=8, type=int)
        parser.add_argument('--causal_innovation_hidden_size', default=32, type=int)
        parser.add_argument('--causal_innovation_max_position_m', default=50.0, type=float)
        parser.add_argument('--causal_innovation_max_velocity_mps', default=5.0, type=float)
        parser.add_argument('--causal_innovation_dynamic_activation', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='select the frozen causal innovation expert with a learned strictly-causal activation')
        parser.add_argument('--causal_innovation_activation_floor', default=0.5, type=float)
        parser.add_argument('--causal_innovation_activation_nis_low', default=1.38629436112, type=float)
        parser.add_argument('--causal_innovation_activation_nis_high', default=5.99146454711, type=float)
        parser.add_argument('--causal_innovation_minimum_valid_history', default=0, type=int,
                            help='minimum paired own/peer innovation frames; 0 preserves the legacy all-valid rule')
        parser.add_argument('--causal_innovation_recent_valid_window', default=3, type=int,
                            help='require at least one valid paired innovation in this recent suffix')
        parser.add_argument('--causal_innovation_association_context', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='condition CGIC on detection mass, association entropy and candidate spread')
        parser.add_argument('--causal_innovation_use_peer', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='use leave-one-out peer innovation; disable for the own-only ablation')
        parser.add_argument('--association_augmentation', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='training-only causal soft-association augmentation with misses and clutter')
        parser.add_argument('--association_validation', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='also evaluate a deterministic missed-detection/local-clutter validation rollout')
        parser.add_argument('--association_validation_seed', default=20260813, type=int)
        parser.add_argument('--association_pd_min', default=0.75, type=float)
        parser.add_argument('--association_pd_max', default=0.98, type=float)
        parser.add_argument('--association_clutter_mu_min', default=0.0, type=float,
                            help='training-only global clutter; keep 0 for the identity-consistent protocol')
        parser.add_argument('--association_clutter_mu_max', default=0.0, type=float)
        parser.add_argument('--association_surveillance_side_m', default=3000.0, type=float)
        parser.add_argument('--association_max_clutter', default=32, type=int)
        parser.add_argument('--association_local_clutter_rate', default=0.25, type=float)
        parser.add_argument('--association_max_local_clutter', default=2, type=int)
        parser.add_argument('--association_local_clutter_scale', default=2.5, type=float)
        parser.add_argument('--association_gate_threshold', default=9.21034037198, type=float,
                            help='chi-square gating threshold for training-time candidate associations')
        parser.add_argument('--association_minimum_detection_mass', default=0.05, type=float)
        parser.add_argument('--association_include_target_competitors', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='include other targets as training association candidates; disabled for identity-consistent supervision')
        parser.add_argument('--association_clutter_requires_detection', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='do not create identity-breaking clutter-only updates on simulated missed detections')
        parser.add_argument('--lambda_causal_velocity_guard', default=0.0, type=float,
                            help='one-sided counterfactual protection against velocity-error regression')
        parser.add_argument('--causal_velocity_guard_margin', default=0.0, type=float,
                            help='allowed task-scaled velocity-error increase before the guard activates')
        parser.add_argument('--gasf_fusion_mode', default='concat',
                            choices=['concat', 'add', 'film'],
                            help='Direct concatenation/addition fusion or legacy gated FiLM')
        parser.add_argument('--gasf_aux_source', default='gasf',
                            choices=['gasf', 'fusion'],
                            help='Predict auxiliary delta-v from raw GASF features or the residual actually fused into the ODE')
        parser.add_argument('--model_variant', default='sggt_v2',
                            choices=['sggt_v1', 'sggt_v2'])
        parser.add_argument('--covariance_transition', default='ode_jacobian',
                            choices=['ode_jacobian', 'cv'])
        parser.add_argument('--lambda_struct', default=0.0, type=float,
                            help='deprecated: formal spectral-SEAN training uses no group labels')
        parser.add_argument('--lambda_temporal', default=0.0, type=float,
                            help='deprecated: no adjacency consistency loss in label-free training')
        parser.add_argument('--lambda_aux', default=0.1, type=float)
        parser.add_argument('--lambda_nll', default=1e-4, type=float,
                            help='weight for dimension-normalized Gaussian NLL')
        parser.add_argument('--adaptive_q_mode', default='legacy',
                            choices=['legacy', 'innovation', 'causal_nis'],
                            help='legacy hidden-only Q or causal innovation-conditioned Q')
        parser.add_argument('--adaptive_q_max_multiplier', default=5.0, type=float)
        parser.add_argument('--adaptive_q_max_position_variance', default=400.0, type=float,
                            help='per-axis physical position variance ceiling in m^2')
        parser.add_argument('--adaptive_q_initial_multiplier', default=1.0, type=float,
                            help='constant acceleration-Q multiplier at causal adapter initialization')
        parser.add_argument('--adaptive_q_initial_position_variance', default=0.01, type=float,
                            help='initial direct position-Q variance per axis in m^2')
        parser.add_argument('--adaptive_q_acceleration', default=True,
                            action=argparse.BooleanOptionalAction,
                            help='allow Q-v2 to rescale acceleration Q; disable for position-only adaptation')
        parser.add_argument('--adaptive_q_tail_boost', default=0.0, type=float,
                            help='fixed monotone boost for causal residual tails; 0 disables it')
        parser.add_argument('--adaptive_q_tail_threshold', default=0.8, type=float)
        parser.add_argument('--lambda_q_rank', default=0.0, type=float)
        parser.add_argument('--q_rank_margin', default=0.25, type=float,
                            help='log-Q separation required between maneuver and steady nodes')
        parser.add_argument('--lambda_q_smooth', default=0.0, type=float)
        parser.add_argument('--lambda_q_baseline', default=0.0, type=float)
        parser.add_argument('--lambda_q_surprise', default=0.0, type=float,
                            help='rank Q by strictly causal residual-history surprise')
        parser.add_argument('--q_surprise_margin', default=0.25, type=float)
        parser.add_argument('--q_surprise_quantile', default=0.25, type=float)
        parser.add_argument('--q_surprise_min_gap', default=0.1, type=float,
                            help='minimum log-residual gap required to activate ranking')
        parser.add_argument('--q_surprise_high_target', default=0.0, type=float,
                            help='optional lower anchor for high-surprise effective Q score')
        parser.add_argument('--q_surprise_low_target', default=0.0, type=float,
                            help='optional upper anchor for low-surprise effective Q score')
        parser.add_argument('--q_surprise_anchor_weight', default=1.0, type=float)
        parser.add_argument('--q_surprise_high_threshold', default=0.0, type=float,
                            help='absolute causal-surprise threshold; 0 uses quantiles')
        parser.add_argument('--q_surprise_low_threshold', default=0.0, type=float)
        parser.add_argument('--lambda_cv_guard', default=0.0, type=float,
                            help='weight for one-sided regret relative to the causal CV prior')
        parser.add_argument('--cv_guard_margin', default=0.0, type=float,
                            help='allowed normalized error increase over CV before guard activates')
        parser.add_argument('--cv_guard_position_weight', default=1.0, type=float)
        parser.add_argument('--cv_guard_velocity_weight', default=1.0, type=float)
        parser.add_argument('--velocity_loss_weight', default=1.0, type=float,
                            help='velocity-coordinate weight in trajectory losses')
        parser.add_argument('--position_loss_weight', default=1.0, type=float,
                            help='position-coordinate weight in trajectory losses')
        parser.add_argument('--counterfactual_training', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--lambda_interaction_expert', default=0.25, type=float,
                            help='trajectory loss weight for the fully active interaction expert')
        parser.add_argument('--lambda_interaction_base', default=0.05, type=float,
                            help='trajectory loss weight for the temporal-only counterfactual')
        parser.add_argument('--lambda_interaction_gate', default=0.0, type=float,
                            help='dead-zone binary counterfactual supervision weight for spectral SEAN activation')
        parser.add_argument('--lambda_interaction_regret', default=0.1, type=float,
                            help='protect mixed prediction against the better counterfactual branch')
        parser.add_argument('--lambda_interaction_aux_expert', default=0.25, type=float,
                            help='form a useful full-interaction expert through auxiliary velocity supervision')
        parser.add_argument('--lambda_interaction_aux_rank', default=0.1, type=float,
                            help='make the full-interaction auxiliary expert outperform the temporal branch')
        parser.add_argument('--lambda_interaction_state_alignment', default=0.0, type=float,
                            help='align graph-expert state correction with the temporal prediction residual')
        parser.add_argument('--lambda_interaction_state_improvement', default=0.0, type=float,
                            help='one-sided loss requiring graph-expert state error below temporal-only')
        parser.add_argument('--interaction_expert_steady_weight', default=1.0, type=float)
        parser.add_argument('--interaction_expert_switch_weight', default=1.0, type=float)
        parser.add_argument('--interaction_expert_strong_weight', default=1.0, type=float)
        parser.add_argument('--interaction_alignment_position_weight', default=1.0, type=float)
        parser.add_argument('--interaction_alignment_velocity_weight', default=1.0, type=float)
        parser.add_argument('--interaction_expert_position_scale', default=25.0, type=float,
                            help='physical position scale (m) for graph-expert task loss')
        parser.add_argument('--interaction_expert_velocity_scale', default=5.0, type=float,
                            help='physical velocity scale (m/s) for graph-expert task loss')
        parser.add_argument('--interaction_aux_rank_margin', default=1e-4, type=float,
                            help='minimum auxiliary-error advantage requested from the interaction expert')
        parser.add_argument('--lambda_interaction_demand', default=0.01, type=float,
                            help='trajectory-derived maneuver-demand supervision weight for spectral SEAN')
        parser.add_argument('--interaction_maneuver_threshold', default=2.0, type=float,
                            help='physical velocity increment (m/s) defining maneuver demand')
        parser.add_argument('--interaction_gate_temperature', default=1e-3, type=float,
                            help='deprecated compatibility option; binary counterfactual targets do not use a temperature')
        parser.add_argument('--interaction_gate_margin', default=1e-5, type=float,
                            help='ignore counterfactual gate labels inside this normalized error margin')
        parser.add_argument('--interaction_gate_position_margin_m', default=1e-4, type=float,
                            help='gate-stage dead zone for paired position-error advantage, in metres')
        parser.add_argument('--interaction_gate_steady_weight', default=1.0, type=float)
        parser.add_argument('--interaction_gate_switch_weight', default=1.5, type=float)
        parser.add_argument('--interaction_gate_strong_weight', default=2.0, type=float)
        parser.add_argument('--interaction_gate_lr_multiplier', default=10.0, type=float,
                            help='learning-rate multiplier for the embedded SEAN activation head during joint training')
        parser.add_argument('--interaction_projection_init_scale', default=1.0, type=float,
                            help='one-time scale absorbed into the graph interaction projection after init checkpoint loading')
        parser.add_argument('--interaction_physical_residual', default=False,
                            action=argparse.BooleanOptionalAction,
                            help='use bounded SEAN neighbour innovation as an ODE acceleration residual')
        parser.add_argument('--interaction_physical_max_gain', default=0.075, type=float)
        parser.add_argument('--interaction_physical_init_gain', default=0.05, type=float)
        parser.add_argument('--interaction_activation_floor', default=0.0, type=float,
                            help='minimum learned physical-interaction activation')
        parser.add_argument('--training_stage', default='joint',
                            choices=['joint', 'expert', 'innovation_expert', 'graph_expert',
                                     'graph_context',
                                     'graph_context_rollout',
                                     'graph_context_bridge',
                                     'physical_gain',
                                     'graph_fusion_adapt', 'gate', 'q_adapt',
                                     'causal_innovation_expert',
                                     'causal_innovation_gate',
                                     'finetune'])
        parser.add_argument('--lambda_expert', default=0.2, type=float)
        parser.add_argument('--lambda_gate', default=0.1, type=float)
        parser.add_argument('--lambda_gate_sparse', default=0.01, type=float)
        parser.add_argument('--gate_margin', default=0.01, type=float)
        parser.add_argument('--gate_temperature', default=0.05, type=float)


        parser.add_argument('--neural_net', default='sggt_net', type=str)  # baseline/ sggt_net/  MCST
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()


