import os
import hashlib
import json
from pathlib import Path

from data.load_data import *
import torch
from torch.utils.data import Subset
import numpy as np
from loguru import logger
from model.sggt_net import SGGT_Net
from train.trainer import *
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from train.trainer_v3 import train_and_evaluation as train_and_evaluation_v3
from torch_geometric.loader import DataLoader
from config import *
from model.motion_models import *
from model.motion_models_base import *



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_v3_evaluation_protocol(dataset, dataset_root, config, args):
    """Return an auditable manifest for the exact validation trajectories."""
    root = Path(dataset_root).resolve()
    trajectories = []
    for target_path, input_path in zip(dataset.target_files, dataset.input_files):
        trajectories.append({
            "target": str(target_path.resolve().relative_to(root)),
            "target_bytes": target_path.stat().st_size,
            "input": str(input_path.resolve().relative_to(root)),
            "input_bytes": input_path.stat().st_size,
        })
    core = {
        "version": 2,
        "dataset_root": str(root),
        "split": dataset.mode,
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
        "input_window_length": int(config["input_graph_window_len"]),
        "output_window_length": int(config["output_graph_window_len"]),
        "validation_selection": "all_legal_causal_steps",
        "validation_order": "sorted_batch_data_filename",
        "validity_mask": "real_mask_at_scored_target_step",
        "metric_reduction": "global_valid_point_sse_then_sqrt",
        "training_supervision": (
            "all_legal_causal_steps"
            if args.full_time_supervision
            else "event_stratified_sampled_windows"
        ),
        "encoder_execution_mode": args.encoder_execution_mode,
        "encoder_stream_reset_interval": int(args.encoder_stream_reset_interval),
        "covariance_transition": args.covariance_transition,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
    protocol = dict(core)
    protocol["protocol_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    path = Path(args.output_dir) / "evaluation_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol

if __name__ == '__main__':
    args = Args().get_parser()
    setup_seed(args.seed)
    logger.add(args.log_dir)
    dataset_str = args.data_dir
    is_v3_pack = os.path.isdir(os.path.join(dataset_str, 'val'))
    config = dict(CONFIG)
    if is_v3_pack:
        config.update({'S_VEL': 40.0, 'S_EDGE_V_DIST': 40.0,
                       'output_graph_window_len': 1,
                       'window_sampling_ratio': [0.4, 0.3, 0.3]})
        CONFIG.update(config)
    # 数据集加载
    if is_v3_pack:
        train_dataset = GroupTargetTrackingDatasetV3(
            dataset_str, 'train', load_structure=False
        )
        test_dataset = GroupTargetTrackingDatasetV3(
            dataset_str, 'val', load_structure=False
        )
        args.evaluation_protocol = build_v3_evaluation_protocol(
            test_dataset, dataset_str, config, args
        )
        print(
            "Evaluation protocol: "
            f"{args.evaluation_protocol['protocol_sha256']} "
            f"({args.evaluation_protocol['trajectory_count']} trajectories)"
        )
    else:
        train_dataset = GroupTargetTrackingDataset_load_whole_trajectory(dataset_str,'train')
        test_dataset = GroupTargetTrackingDataset_load_whole_trajectory(dataset_str,'test')
        args.evaluation_protocol = None
    if args.debug_train_samples > 0:
        train_start = max(0, int(args.debug_train_start))
        train_stop = min(train_start + args.debug_train_samples, len(train_dataset))
        if train_start >= train_stop:
            raise ValueError("empty debug training subset")
        train_dataset = Subset(train_dataset, range(train_start, train_stop))
    if args.debug_val_samples > 0:
        val_start = max(0, int(args.debug_val_start))
        val_stop = min(val_start + args.debug_val_samples, len(test_dataset))
        if val_start >= val_stop:
            raise ValueError("empty debug validation subset")
        test_dataset = Subset(test_dataset, range(val_start, val_stop))

    # 按照batch size 划分
    loader_options = dict(num_workers=args.loader_workers, pin_memory=args.pin_memory)
    if args.loader_workers > 0:
        loader_options['prefetch_factor'] = args.loader_prefetch_factor
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, **loader_options)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_options)

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
    else:
        m_model = SecondOrderNeuralODE_groupTrack(solver=args.ode_solver, dt=args.T, mixtures=args.n_mixtures,
                                       static_f_dim=0, n_hidden=args.n_ode_hidden,
                                       n_layers=args.n_ode_layers, use_SEAN=args.decoder_use_SEAN,
                                       use_GASF=args.decoder_use_GASF,
                                       gasf_fusion_mode=args.gasf_fusion_mode,
                                       gasf_aux_source=args.gasf_aux_source,
                                       interaction_physical_residual=args.interaction_physical_residual,
                                       interaction_physical_max_gain=args.interaction_physical_max_gain,
                                       interaction_physical_init_gain=args.interaction_physical_init_gain,
                                       interaction_activation_floor=args.interaction_activation_floor)
    # 模型
    track_model = SGGT_Net(args.encoder_input_size, args.encoder_hidden_size,
                                 args.encoder_n_heads,
                                 args.encoder_n_layers, m_model.mixtures, args.encoder_dropout,
                                 args.encoder_gnn_layer, args.encoder_use_edge_features,
                                 m_model, args.decoder_max_length, args.decoder_hidden_size,
                                 args.decoder_n_heads, args.decoder_n_layers, args.decoder_alpha,
                                 args.decoder_dropout, args.decoder_residual_length,
                                 args.decoder_z_dimension, args.decoder_gnn_layer,
                                 args.decoder_use_GASF, args.decoder_use_SEAN, args.T,
                                 model_variant=args.model_variant,
                                 covariance_transition=args.covariance_transition,
                                 gasf_filter_mode=args.gasf_filter_mode,
                                 gasf_graph_mode=args.gasf_graph_mode,
                                 gasf_recurrent_filter=args.gasf_recurrent_filter,
                                 gasf_recurrent_gate_initial_update=(
                                     args.gasf_recurrent_gate_initial_update
                                 ),
                                 gasf_abc_c_limit=args.gasf_abc_c_limit,
                                 gasf_glp_p_min=args.gasf_glp_p_min,
                                 gasf_glp_p_max=args.gasf_glp_p_max,
                                 gasf_fft_points=args.gasf_fft_points,
                                 counterfactual_training=args.counterfactual_training,
                                 adaptive_q_mode=args.adaptive_q_mode,
                                 adaptive_q_max_multiplier=args.adaptive_q_max_multiplier,
                                 adaptive_q_max_position_variance=args.adaptive_q_max_position_variance,
                                 adaptive_q_initial_multiplier=args.adaptive_q_initial_multiplier,
                                 adaptive_q_initial_position_variance=(
                                     args.adaptive_q_initial_position_variance
                                 ),
                                 adaptive_q_acceleration=args.adaptive_q_acceleration,
                                 adaptive_q_tail_boost=args.adaptive_q_tail_boost,
                                 adaptive_q_tail_threshold=args.adaptive_q_tail_threshold,
                                 innovation_decomposition=args.innovation_decomposition,
                                 innovation_whiten_clip=args.innovation_whiten_clip,
                                 causal_innovation_correction=args.causal_innovation_correction,
                                 causal_innovation_history=args.causal_innovation_history,
                                 causal_innovation_hidden_size=args.causal_innovation_hidden_size,
                                 causal_innovation_output_mode=(
                                     args.causal_innovation_output_mode
                                 ),
                                 causal_innovation_state_channels=(
                                     args.causal_innovation_state_channels
                                 ),
                                 causal_innovation_max_acceleration_mps2=(
                                     args.causal_innovation_max_acceleration_mps2
                                 ),
                                 causal_innovation_max_position_m=(
                                     args.causal_innovation_max_position_m
                                 ),
                                 causal_innovation_max_velocity_mps=(
                                     args.causal_innovation_max_velocity_mps
                                 ),
                                 causal_innovation_minimum_valid_history=(
                                     args.causal_innovation_minimum_valid_history or None
                                 ),
                                 causal_innovation_recent_valid_window=(
                                     args.causal_innovation_recent_valid_window
                                 ),
                                 causal_innovation_use_peer=args.causal_innovation_use_peer)



    track_model = track_model.to(args.device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        incompatible = track_model.load_state_dict(initial['state_dict'], strict=False)
        print(
            f"Initialized from {args.init_checkpoint}; "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )
        projection_scale = float(args.interaction_projection_init_scale)
        if projection_scale <= 0.0:
            raise ValueError("--interaction_projection_init_scale must be positive")
        if projection_scale != 1.0:
            if not args.decoder_use_GASF or not args.decoder_use_SEAN:
                raise ValueError(
                    "interaction projection scaling requires GASF and SEAN"
                )
            final_projection = track_model.decoder.GASF.interaction_projection[-1]
            with torch.no_grad():
                final_projection.weight.mul_(projection_scale)
                if final_projection.bias is not None:
                    final_projection.bias.mul_(projection_scale)
            print(
                f"Absorbed interaction projection scale {projection_scale:g} "
                "into the final projection layer"
            )

    staged_joint_finetune = (
        args.training_stage in ('joint', 'finetune')
        and args.decoder_use_GASF
        and args.gasf_fusion_mode == 'concat'
        and args.fusion_adapt_epochs > 0
    )
    args.staged_joint_finetune = staged_joint_finetune
    if args.training_stage == 'physical_gain':
        if not (
            args.decoder_use_GASF
            and args.decoder_use_SEAN
            and args.interaction_physical_residual
        ):
            raise ValueError(
                "physical_gain requires GASF, SEAN and physical interaction"
            )
        track_model.counterfactual_evaluation = True
        track_model.decoder.GASF.interaction_activation_override = 1.0
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        track_model.decoder.motion_model.interaction_physical_gain_raw.requires_grad = True
        print(
            "Physical-gain stage: one bounded gain parameter is trainable; "
            "online rollout uses the always-on physical interaction expert"
        )
    elif args.training_stage == 'innovation_expert':
        if not (
            args.decoder_use_GASF
            and args.decoder_use_SEAN
            and args.innovation_decomposition
        ):
            raise ValueError(
                "innovation_expert requires GASF, SEAN and "
                "--innovation-decomposition"
            )
        # Learn the new causal expert against the exact frozen Temporal
        # comparator. The online rollout remains on alpha=0 until the expert
        # has demonstrated a positive counterfactual contribution.
        track_model.counterfactual_training = True
        track_model.counterfactual_evaluation = True
        track_model.decoder.GASF.interaction_activation_override = 0.0
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        innovation_modules = (
            track_model.decoder.GASF.group_innovation_encoders,
            track_model.decoder.GASF.individual_innovation_encoders,
            track_model.decoder.GASF.innovation_uncertainty_encoder,
            track_model.decoder.GASF.innovation_decomposition_projection,
            track_model.decoder.SEAN.node_projection,
            track_model.decoder.SEAN.edge_network,
        )
        for module in innovation_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        track_model.decoder.SEAN.temperature_raw.requires_grad = True
        trainable = sum(
            parameter.numel() for parameter in track_model.parameters()
            if parameter.requires_grad
        )
        print(
            f"Innovation-expert stage: {trainable} trainable parameters; "
            "Temporal GASF, encoder and ODE are frozen; online rollout uses "
            "the temporal comparator"
        )
    elif args.training_stage == 'filter_recurrence':
        if not args.decoder_use_GASF or not args.gasf_recurrent_filter:
            raise ValueError(
                "filter_recurrence requires GASF with recurrent ABC enabled"
            )
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        for parameter in track_model.decoder.GASF.parameter_update_gates.parameters():
            parameter.requires_grad = True
        recurrent_parameter_count = sum(
            parameter.numel() for parameter in track_model.parameters()
            if parameter.requires_grad
        )
        if recurrent_parameter_count == 0:
            raise RuntimeError("filter_recurrence has no trainable parameters")
        print(
            "ABC recurrence adaptation stage: "
            f"{recurrent_parameter_count} trainable parameters"
        )
    elif args.training_stage == 'q_adapt':
        if track_model.decoder.Q_innovation_encoder is None:
            raise ValueError("q_adapt requires --adaptive_q_mode innovation")
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        for parameter in track_model.decoder.Q_innovation_encoder.parameters():
            parameter.requires_grad = True
    elif args.training_stage == 'causal_innovation_expert':
        if not args.causal_innovation_correction:
            raise ValueError(
                "causal_innovation_expert requires --causal_innovation_correction"
            )
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        for parameter in track_model.causal_innovation_corrector.parameters():
            parameter.requires_grad = True
        track_model.counterfactual_training = True
        track_model.counterfactual_evaluation = True
    elif args.training_stage == 'causal_innovation_gate':
        raise ValueError(
            "causal_innovation_gate was removed; acceleration CIPIC is always "
            "active when its causal history is eligible"
        )
    elif args.training_stage in (
        'graph_expert', 'graph_context', 'graph_context_rollout',
        'graph_context_bridge'
    ):
        if not args.decoder_use_GASF or not args.decoder_use_SEAN:
            raise ValueError("graph_expert requires embedded spectral SEAN")
        track_model.counterfactual_evaluation = True
        # Keep the online state, innovation buffer and covariance trajectory on
        # the temporal-only path while the alpha=1 counterfactual learns.  If
        # the mixed path drives the rollout, every graph update also changes
        # the future "baseline" history and invalidates the comparison.
        track_model.decoder.GASF.interaction_activation_override = (
            1.0 if args.training_stage == 'graph_context_rollout' else 0.0
        )
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        if args.training_stage == 'graph_context_bridge':
            graph_expert_modules = (
                track_model.decoder.motion_model.gasf_interaction_context_embedding,
            )
        elif args.training_stage in ('graph_context', 'graph_context_rollout'):
            # The already validated SEAN graph supplies a useful causal
            # neighbour-innovation signal. Adapt only its new feature
            # projection so the temporal comparator, topology, ODE and
            # covariance path remain invariant.
            graph_expert_modules = (
                track_model.decoder.GASF.interaction_innovation_projection,
            )
        else:
            graph_expert_modules = (
                track_model.decoder.GASF.interaction_projection,
                track_model.decoder.GASF.interaction_innovation_projection,
                track_model.decoder.SEAN.node_projection,
                track_model.decoder.SEAN.edge_network,
            )
        for module in graph_expert_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if args.training_stage == 'graph_expert':
            track_model.decoder.SEAN.temperature_raw.requires_grad = True
        graph_expert_parameter_count = sum(
            parameter.numel() for parameter in track_model.parameters()
            if parameter.requires_grad
        )
        if graph_expert_parameter_count == 0:
            raise RuntimeError("graph_expert stage has no trainable parameters")
        print(
            f"{args.training_stage} stage: {graph_expert_parameter_count} "
            "trainable parameters; temporal GASF, activation head and ODE are frozen; "
            "online rollout is fixed to "
            f"{'interaction expert' if args.training_stage == 'graph_context_rollout' else 'temporal-only'}"
        )
    elif args.training_stage == 'graph_fusion_adapt':
        if not args.decoder_use_GASF or not args.decoder_use_SEAN:
            raise ValueError("graph_fusion_adapt requires embedded spectral SEAN")
        if args.gasf_fusion_mode != 'concat':
            raise ValueError("graph_fusion_adapt currently requires concat fusion")
        track_model.counterfactual_evaluation = True
        # This stage calibrates a deterministic always-on graph feature. The
        # activation head and the already learned graph topology stay frozen.
        track_model.decoder.GASF.interaction_activation_override = 1.0
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        graph_fusion_modules = (
            track_model.decoder.GASF.interaction_projection,
            track_model.decoder.motion_model.gasf_concat_fusion,
        )
        for module in graph_fusion_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        graph_fusion_parameter_count = sum(
            parameter.numel() for parameter in track_model.parameters()
            if parameter.requires_grad
        )
        if graph_fusion_parameter_count == 0:
            raise RuntimeError("graph_fusion_adapt stage has no trainable parameters")
        print(
            f"Graph-fusion adaptation stage: {graph_fusion_parameter_count} "
            "trainable parameters; activation, graph topology, encoder, ODE "
            "backbone and covariance path are frozen; interaction is always on"
        )
    elif args.training_stage == 'expert' or staged_joint_finetune:
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        for parameter in track_model.decoder.GASF.parameters():
            parameter.requires_grad = True
        if staged_joint_finetune and args.decoder_use_SEAN:
            for parameter in track_model.decoder.SEAN.parameters():
                parameter.requires_grad = True
        fusion_module = None
        if args.gasf_fusion_mode == 'concat':
            fusion_module = track_model.decoder.motion_model.gasf_concat_fusion
        elif args.gasf_fusion_mode == 'film':
            fusion_module = track_model.decoder.motion_model.gasf_film
        if fusion_module is not None:
            for parameter in fusion_module.parameters():
                parameter.requires_grad = True
        for parameter in track_model.decoder.motion_model.aux_velocity_head.parameters():
            parameter.requires_grad = True
    elif args.training_stage == 'gate':
        if not args.decoder_use_GASF or not args.decoder_use_SEAN:
            raise ValueError("gate stage requires embedded spectral SEAN")
        # Validation runs in eval mode, but the gate objective still needs the
        # paired alpha=0/alpha=1 branches from exactly the same causal state.
        track_model.counterfactual_evaluation = True
        track_model.decoder.GASF.interaction_activation_override = None
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        # Direct-concatenation V2 uses the embedded SEAN activation as its
        # interaction selector. The legacy ODE gasf_gate is bypassed in this
        # mode, so unfreezing it produced a no-op calibration stage.
        activation_module = (
            track_model.decoder.GASF.context_activation_head
            if args.interaction_physical_residual
            else track_model.decoder.SEAN.activation_head
        )
        for parameter in activation_module.parameters():
            parameter.requires_grad = True
        gate_parameter_count = sum(
            parameter.numel() for parameter in track_model.parameters()
            if parameter.requires_grad
        )
        if gate_parameter_count == 0:
            raise RuntimeError("gate stage has no trainable parameters")
        print(
            f"On-demand activation stage: {gate_parameter_count} trainable "
            "parameters; temporal GASF, graph expert, encoder and ODE are frozen"
        )
    if staged_joint_finetune:
        fusion_parameters = [
            parameter for parameter in track_model.parameters()
            if parameter.requires_grad
        ]
        fusion_ids = {id(parameter) for parameter in fusion_parameters}
        backbone_parameters = [
            parameter for parameter in track_model.parameters()
            if id(parameter) not in fusion_ids
        ]
        track_model_optimizer = torch.optim.AdamW(
            [
                {'params': fusion_parameters, 'lr': args.lr, 'name': 'fusion'},
                {'params': backbone_parameters, 'lr': args.lr, 'name': 'backbone'},
            ],
            weight_decay=args.weight_decay,
        )
    elif (
        args.training_stage in ('joint', 'finetune')
        and args.decoder_use_GASF
        and args.decoder_use_SEAN
        and float(args.interaction_gate_lr_multiplier) != 1.0
    ):
        selector_module = (
            track_model.decoder.GASF.context_activation_head
            if args.interaction_physical_residual
            else track_model.decoder.SEAN.activation_head
        )
        gate_parameters = [
            parameter
            for parameter in selector_module.parameters()
            if parameter.requires_grad
        ]
        if not gate_parameters:
            raise RuntimeError(
                "interaction selector LR multiplier requested, but the SEAN "
                "activation head has no trainable parameters"
            )
        gate_ids = {id(parameter) for parameter in gate_parameters}
        backbone_parameters = [
            parameter for parameter in track_model.parameters()
            if parameter.requires_grad and id(parameter) not in gate_ids
        ]
        track_model_optimizer = torch.optim.AdamW(
            [
                {'params': backbone_parameters, 'lr': args.lr, 'name': 'backbone'},
                {
                    'params': gate_parameters,
                    'lr': args.lr * float(args.interaction_gate_lr_multiplier),
                    'name': 'interaction_selector',
                },
            ],
            weight_decay=args.weight_decay,
        )
        optimizer_group_names = {
            group.get('name') for group in track_model_optimizer.param_groups
        }
        if optimizer_group_names != {'backbone', 'interaction_selector'}:
            raise RuntimeError(
                f"unexpected optimizer groups: {sorted(optimizer_group_names)}"
            )
        print(
            "Optimizer groups: "
            + ", ".join(
                f"{group['name']}={group['lr']:.3e} "
                f"({sum(parameter.numel() for parameter in group['params'])} parameters)"
                for group in track_model_optimizer.param_groups
            )
        )
    else:
        track_model_optimizer = torch.optim.AdamW(
            (parameter for parameter in track_model.parameters() if parameter.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay
        )
    warmup_epochs = max(0, min(args.warmup_epochs, args.train_epochs - 1))
    if staged_joint_finetune:
        adapt_epochs = min(args.fusion_adapt_epochs, args.train_epochs - 1)
        decay_epochs = max(1, args.train_epochs - adapt_epochs - 1)
        fine_ratio = args.finetune_lr / args.lr
        min_ratio = args.min_lr / args.lr

        def staged_lr(epoch):
            if epoch < adapt_epochs:
                return 1.0
            progress = min(1.0, max(0.0, (epoch - adapt_epochs) / decay_epochs))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_ratio + (fine_ratio - min_ratio) * cosine

        track_model_lr_schedule = torch.optim.lr_scheduler.LambdaLR(
            track_model_optimizer, lr_lambda=staged_lr
        )
    elif warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            track_model_optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            track_model_optimizer,
            T_max=max(1, args.train_epochs - warmup_epochs),
            eta_min=args.min_lr,
        )
        track_model_lr_schedule = torch.optim.lr_scheduler.SequentialLR(
            track_model_optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        track_model_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            track_model_optimizer,
            T_max=max(1, args.train_epochs),
            eta_min=args.min_lr,
        )
    args.start_epoch = 0
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)
        track_model.load_state_dict(resume['state_dict'])
        track_model_optimizer.load_state_dict(resume['optimizer_state_dict'])
        if args.resume_reset_scheduler:
            for parameter_group in track_model_optimizer.param_groups:
                parameter_group['lr'] = args.lr
                parameter_group['initial_lr'] = args.lr
            print(
                f"Reset scheduler on resume; continuation LR starts at {args.lr:g}"
            )
        else:
            track_model_lr_schedule.load_state_dict(resume['lr_schedule'])
        args.start_epoch = int(resume['epoch']) + 1
        print(f"Resuming from epoch {args.start_epoch}: {args.resume_checkpoint}")
    total_params = sum(p.numel() for p in track_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (Method 1): {total_params}")

    # 训练
    training_entry = train_and_evaluation_v3 if is_v3_pack else train_and_evaluation
    training_entry(track_model, train_dataloader, test_dataloader, track_model_optimizer,
                   track_model_lr_schedule, logger, config, args)
