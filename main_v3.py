from data.load_data_v3 import *
import torch
from torch.utils.data import Subset
import numpy as np
from loguru import logger
from model.sggt_net import SGGT_Net
from train.trainer_v3 import *
from torch_geometric.loader import DataLoader
from config import *
from model.motion_models import *
from model.motion_models_base import *
import hashlib
import json
from pathlib import Path



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_evaluation_protocol(dataset, dataset_root, config, output_dir):
    """Persist an auditable, model-independent validation trajectory manifest."""
    root = Path(dataset_root).resolve()
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        selected_indices = [int(index) for index in dataset.indices]
    else:
        base_dataset = dataset
        selected_indices = list(range(len(dataset)))
    files = []
    for index in selected_indices:
        target_path = base_dataset.target_files[index]
        input_path = base_dataset.input_files[index]
        files.append({
            "target": str(target_path.resolve().relative_to(root)),
            "target_bytes": target_path.stat().st_size,
            "input": str(input_path.resolve().relative_to(root)),
            "input_bytes": input_path.stat().st_size,
        })
    protocol_core = {
        # Version 2 is the schema frozen into the seed-123/456 formal
        # Temporal checkpoints.  Keep this byte-for-byte semantic identity so
        # a newly trained seed cannot silently use a different validation or
        # covariance protocol.
        "version": 2,
        "dataset_root": str(root),
        "split": base_dataset.mode,
        "trajectory_count": len(files),
        "trajectories": files,
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
    canonical = json.dumps(protocol_core, sort_keys=True, separators=(",", ":"))
    protocol = dict(protocol_core)
    protocol["protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = Path(output_dir) / "evaluation_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol

if __name__ == '__main__':
    args = Args().get_parser()
    setup_seed(args.seed)
    logger.add(args.log_dir)
    config = dict(CONFIG)
    config.update({'S_VEL': 40.0, 'S_EDGE_V_DIST': 40.0,
                   'output_graph_window_len': 1,
                   'window_sampling_ratio': [0.4, 0.3, 0.3]})
    # Several legacy model/EKF components retain a reference to the shared
    # CONFIG dictionary.  Keep that dictionary synchronized with the v3
    # runtime configuration; otherwise ODE position integration uses
    # S_VEL=300 while the trainer normalizes velocity with S_VEL=40.
    CONFIG.update(config)
    # 鏁版嵁闆嗗姞杞?
    dataset_str = args.data_dir
    dataset_options = dict(
        load_structure=False,
        load_analysis_metadata=False,
    )
    train_dataset = GroupTargetTrackingDatasetV3(dataset_str, 'train', **dataset_options)
    test_dataset = GroupTargetTrackingDatasetV3(dataset_str, 'val', **dataset_options)
    iid_dataset = GroupTargetTrackingDatasetV3(dataset_str, 'iid_test', **dataset_options)
    ood_dataset = GroupTargetTrackingDatasetV3(dataset_str, 'ood_test', **dataset_options)
    if args.debug_train_samples > 0:
        train_start = min(max(0, args.debug_train_start), len(train_dataset))
        train_stop = min(train_start + args.debug_train_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, range(train_start, train_stop))
    if args.debug_val_samples > 0:
        def validation_subset(dataset):
            start = min(max(0, args.debug_val_start), len(dataset))
            stop = min(start + args.debug_val_samples, len(dataset))
            return Subset(dataset, range(start, stop))
        test_dataset = validation_subset(test_dataset)
        iid_dataset = validation_subset(iid_dataset)
        ood_dataset = validation_subset(ood_dataset)

    # Build the manifest after applying development subsets.  Otherwise a
    # 200/50 run is evaluated on 50 trajectories but incorrectly records all
    # 200 validation files in its checkpoint metadata.
    args.evaluation_protocol = build_evaluation_protocol(
        test_dataset, dataset_str, config, args.output_dir
    )
    print(
        "Evaluation protocol: "
        f"{args.evaluation_protocol['protocol_sha256']} "
        f"({args.evaluation_protocol['trajectory_count']} validation trajectories)"
    )

    # 鎸夌収batch size 鍒掑垎
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0)
    loader_options = dict(
        num_workers=args.loader_workers,
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
    )
    if args.loader_workers > 0:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=args.loader_prefetch_factor,
        )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, **loader_options
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_options
    )
    iid_dataloader = DataLoader(
        iid_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_options
    )
    ood_dataloader = DataLoader(
        ood_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_options
    )
    if args.skip_final_tests:
        iid_dataloader = None
        ood_dataloader = None

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
                                       gasf_aux_source=args.gasf_aux_source)
    # 妯″瀷
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
                                  causal_innovation_max_acceleration_mps2=(
                                      args.causal_innovation_max_acceleration_mps2
                                  ),
                                  causal_innovation_max_position_m=args.causal_innovation_max_position_m,
                                  causal_innovation_max_velocity_mps=args.causal_innovation_max_velocity_mps,
                                  causal_innovation_dynamic_activation=args.causal_innovation_dynamic_activation,
                                  causal_innovation_activation_floor=args.causal_innovation_activation_floor,
                                  causal_innovation_activation_nis_low=args.causal_innovation_activation_nis_low,
                                  causal_innovation_activation_nis_high=args.causal_innovation_activation_nis_high,
                                  causal_innovation_minimum_valid_history=(
                                      args.causal_innovation_minimum_valid_history or None
                                  ),
                                  causal_innovation_recent_valid_window=args.causal_innovation_recent_valid_window,
                                  causal_innovation_association_context=args.causal_innovation_association_context,
                                  causal_innovation_use_peer=args.causal_innovation_use_peer,
                                  association_augmentation=args.association_augmentation,
                                  association_pd_min=args.association_pd_min,
                                  association_pd_max=args.association_pd_max,
                                  association_clutter_mu_min=args.association_clutter_mu_min,
                                  association_clutter_mu_max=args.association_clutter_mu_max,
                                  association_surveillance_side_m=args.association_surveillance_side_m,
                                  association_max_clutter=args.association_max_clutter,
                                  association_local_clutter_rate=args.association_local_clutter_rate,
                                  association_max_local_clutter=args.association_max_local_clutter,
                                  association_local_clutter_scale=args.association_local_clutter_scale,
                                  association_gate_threshold=args.association_gate_threshold,
                                  association_minimum_detection_mass=args.association_minimum_detection_mass,
                                  association_include_target_competitors=args.association_include_target_competitors,
                                  association_clutter_requires_detection=args.association_clutter_requires_detection)



    track_model = track_model.to(args.device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        incompatible = track_model.load_state_dict(initial['state_dict'], strict=False)
        allowed_missing_prefixes = (
            (('decoder.motion_model.causal_innovation_corrector.',)
             if args.causal_innovation_correction else ())
            + (('decoder.Q_innovation_encoder.',)
               if args.adaptive_q_mode != 'legacy' else ())
            # The parameter-only recurrence heads are instantiated in both
            # ablation arms so the common initialization and parameter count
            # remain identical. Legacy Temporal checkpoints predate them.
            + ('decoder.GASF.parameter_update_gates.',)
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "init checkpoint incompatibility: "
                f"missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        print(
            f"Initialized from {args.init_checkpoint}; "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )

    staged_joint_finetune = (
        args.training_stage in ('joint', 'finetune')
        and args.decoder_use_GASF
        and args.gasf_fusion_mode == 'concat'
        and args.fusion_adapt_epochs > 0
    )
    args.staged_joint_finetune = staged_joint_finetune
    if args.training_stage == 'filter_recurrence':
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
        # Produce the frozen MPSIF/ODE counterfactual only during corrector
        # identification. Deployment uses a single corrected RK4 solve.
        track_model.counterfactual_training = True
        track_model.counterfactual_evaluation = True
    elif args.training_stage == 'causal_innovation_gate':
        raise ValueError(
            "causal_innovation_gate was removed; acceleration CIPIC is always "
            "active when its causal history is eligible"
        )
    elif args.training_stage == 'expert' or staged_joint_finetune:
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        for parameter in track_model.decoder.GASF.parameters():
            parameter.requires_grad = True
        if staged_joint_finetune and args.decoder_use_SEAN:
            for parameter in track_model.decoder.SEAN.parameters():
                parameter.requires_grad = True
        fusion_module = (
            track_model.decoder.motion_model.gasf_concat_fusion
            if args.gasf_fusion_mode == 'concat'
            else track_model.decoder.motion_model.gasf_film
        )
        for parameter in fusion_module.parameters():
            parameter.requires_grad = True
        for parameter in track_model.decoder.motion_model.aux_velocity_head.parameters():
            parameter.requires_grad = True
    elif args.training_stage == 'gate':
        for parameter in track_model.parameters():
            parameter.requires_grad = False
        for parameter in track_model.decoder.motion_model.gasf_gate.parameters():
            parameter.requires_grad = True
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
        track_model_lr_schedule.load_state_dict(resume['lr_schedule'])
        args.start_epoch = int(resume['epoch']) + 1
        print(f"Resuming from epoch {args.start_epoch}: {args.resume_checkpoint}")
    total_params = sum(p.numel() for p in track_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (Method 1): {total_params}")

    # 璁粌
    train_and_evaluation(track_model, train_dataloader, test_dataloader, track_model_optimizer,
                    track_model_lr_schedule, logger, config, args,
                    iid_dataloader=iid_dataloader, ood_dataloader=ood_dataloader)
