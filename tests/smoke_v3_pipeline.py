"""One-window smoke test for a packed Dataset2D v3 directory."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data.load_data_v3 import GroupTargetTrackingDatasetV3
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from model.sggt_net import SGGT_Net
from train.trainer_v3 import Trainer
from train.training_utils_v3 import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--adaptive-q", action="store_true")
    parser.add_argument(
        "--adaptive-q-mode", choices=("innovation", "causal_nis"),
        help="explicit adaptive-Q implementation; also enables Q gradient checks",
    )
    parser.add_argument("--innovation-decomposition", action="store_true")
    parser.add_argument("--association-augmentation", action="store_true")
    parser.add_argument("--causal-innovation", action="store_true")
    parser.add_argument(
        "--causal-output-mode",
        choices=("acceleration", "state_offset"),
        default="acceleration",
    )
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()

    args = SimpleNamespace(
        device=options.device, train_epochs=1, T=2.0,
        decoder_residual_length=[8, 16], neural_net="sggt_net",
        lambda_struct=0.0, lambda_temporal=0.0,
        lambda_aux=0.1, lambda_nll=1e-4,
        lambda_cv_guard=0.0, cv_guard_margin=0.0,
        cv_guard_position_weight=1.0, cv_guard_velocity_weight=1.0,
        position_loss_weight=1.0, velocity_loss_weight=1.0,
        lambda_q_rank=0.2, q_rank_margin=0.25,
        lambda_q_smooth=0.05, lambda_q_baseline=0.01,
        adaptive_q_initial_multiplier=(
            8.0 if options.adaptive_q_mode == "causal_nis" else 1.0
        ),
        adaptive_q_initial_position_variance=(
            1.0 if options.adaptive_q_mode == "causal_nis" else 0.01
        ),
        adaptive_q_max_position_variance=(
            16.0 if options.adaptive_q_mode == "causal_nis" else 400.0
        ),
        lambda_interaction_base=0.05, lambda_interaction_expert=0.25,
        lambda_interaction_gate=0.0, lambda_interaction_regret=0.1,
        lambda_interaction_demand=0.01, interaction_maneuver_threshold=2.0,
        interaction_gate_temperature=1e-3, interaction_gate_margin=1e-5,
        output_dir="./checkpoint",
        association_augmentation=options.association_augmentation,
        causal_innovation_output_mode=options.causal_output_mode,
    )
    config = dict(CONFIG)
    config.update({"output_graph_window_len": 1, "S_VEL": 40.0,
                   "S_EDGE_V_DIST": 40.0,
                   "window_sampling_ratio": [0.4, 0.3, 0.3]})
    dataset = GroupTargetTrackingDatasetV3(
        options.dataset_root, options.split, load_structure=False
    )
    full = next(iter(DataLoader([dataset[options.index]], batch_size=1, shuffle=False)))
    full = full.to(options.device)
    motion = SecondOrderNeuralODE_groupTrack(
        solver="rk4", dt=args.T, mixtures=1, static_f_dim=0,
        n_hidden=16, n_layers=1, use_SEAN=True, use_GASF=True,
        gasf_fusion_mode="add",
    )
    adaptive_q_mode = (
        options.adaptive_q_mode
        or ("innovation" if options.adaptive_q else "legacy")
    )
    model = SGGT_Net(
        4, 16, 1, 1, motion.mixtures, 0.0, "graphconv", True,
        motion, 9, 16, 1, 1, 0.2, 0.0, args.decoder_residual_length,
        2, "graphconv", True, True, args.T,
        model_variant="sggt_v2", covariance_transition="ode_jacobian",
        counterfactual_training=True,
        adaptive_q_mode=adaptive_q_mode,
        adaptive_q_max_multiplier=(16.0 if adaptive_q_mode == "causal_nis" else 5.0),
        adaptive_q_max_position_variance=(
            16.0 if adaptive_q_mode == "causal_nis" else 400.0
        ),
        adaptive_q_initial_multiplier=(8.0 if adaptive_q_mode == "causal_nis" else 1.0),
        adaptive_q_initial_position_variance=(
            1.0 if adaptive_q_mode == "causal_nis" else 0.01
        ),
        innovation_decomposition=options.innovation_decomposition,
        causal_innovation_correction=options.causal_innovation,
        causal_innovation_output_mode=options.causal_output_mode,
        causal_innovation_max_acceleration_mps2=5.0,
        # One real online innovation is sufficient for this two-step smoke;
        # formal training retains the configured four-frame eligibility rule.
        causal_innovation_minimum_valid_history=1,
        causal_innovation_recent_valid_window=3,
        causal_innovation_association_context=False,
        association_augmentation=options.association_augmentation,
        association_pd_min=0.75,
        association_pd_max=0.98,
        association_clutter_mu_min=10.0,
        association_clutter_mu_max=30.0,
        association_local_clutter_rate=0.25,
    ).to(options.device)
    model.association_augmentation_active = options.association_augmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = Trainer(model, optimizer, schedule, logger, config, args)

    window = GraphDataProcessor.init_input_window(full, config, args)
    normalized, centroids = minmax_scaler(
        window, window.clone(), config, args.neural_net
    )
    raw_r, normalized_r = trainer.compute_measurement_noise(
        full, config["input_graph_window_len"]
    )
    covariance = trainer.init_state_covariance(raw_r)
    output = model(normalized, 0.0, covariance, None, normalized_r)
    weight = trainer._event_node_weight(full, CONFIG["input_graph_window_len"])
    event_class = trainer._event_node_class(full, CONFIG["input_graph_window_len"])
    loss = trainer.compute_loss(output, window, weight, event_class=event_class)
    prior = trainer._denormalize_prior(output, centroids, full.batch)
    _, posterior = trainer.compute_evaluation_metrics(window, output, centroids)
    second_window = GraphDataProcessor.update_input_window(
        window,
        full,
        config,
        posterior,
        prior,
        1,
        args,
        measurement_innovation=output.measurement_innovation,
        innovation_covariance=output.innovation_covariance,
        whitened_innovation=output.whitened_innovation,
        nis=output.nis,
        association_detection_mass=output.association_detection_mass,
        association_entropy=output.association_entropy,
        association_spread=output.association_spread,
        association_valid=output.association_valid,
    )
    expected_innovation = (
        output.measurement_innovation[:, 0]
        if options.association_augmentation
        else full.measurement[:, 8] - prior[:, 0, [0, 2]]
    )
    valid = full.real_mask[:, 8, 0].unsqueeze(-1)
    expected_innovation = torch.where(
        valid, expected_innovation, torch.zeros_like(expected_innovation)
    )
    if options.association_augmentation:
        expected_innovation = torch.where(
            output.association_valid[:, 0],
            expected_innovation,
            torch.zeros_like(expected_innovation),
        )

    assert output.states.shape[1] == config["output_graph_window_len"]
    assert prior.shape == (full.x.size(0), 1, 4)
    assert torch.isfinite(loss)
    assert torch.isfinite(output.association_detection_mass).all()
    assert torch.isfinite(output.association_entropy).all()
    assert torch.isfinite(trainer.last_loss_tensors["cv_guard"])
    assert torch.isfinite(trainer.last_loss_tensors["q_baseline"])
    assert torch.allclose(second_window.residual[:, -1], expected_innovation)
    if options.causal_innovation:
        expected_dimension = (
            4 if options.causal_output_mode == "state_offset" else 2
        )
        assert (
            output.causal_innovation_correction.shape[-1]
            == expected_dimension
        )
        assert torch.isfinite(output.causal_innovation_correction).all()
        assert output.counterfactual_base.numel() > 0
        torch.testing.assert_close(
            output.counterfactual_base, output.counterfactual_expert,
            atol=0.0, rtol=0.0,
        )
    backward_loss = loss
    if options.innovation_decomposition or options.causal_innovation:
        # The initial causal window contains no network-generated innovation,
        # so the new branch is intentionally masked. Advance once to verify
        # gradients only after a genuine online residual has been formed.
        second_normalized, second_centroids = minmax_scaler(
            second_window, second_window.clone(), config, args.neural_net
        )
        _, second_r = trainer.compute_measurement_noise(full, 9)
        second_output = model(
            second_normalized,
            0.0,
            output.updated_covariance.detach(),
            None,
            second_r,
        )
        second_weight = trainer._event_node_weight(full, 9)
        second_event = trainer._event_node_class(full, 9)
        backward_loss = backward_loss + trainer.compute_loss(
            second_output,
            second_window,
            second_weight,
            event_class=second_event,
        )
        assert torch.isfinite(second_output.states).all()
    causal_weight_before = (
        model.causal_innovation_corrector.fusion[-1].weight.detach().clone()
        if options.causal_innovation else None
    )
    backward_loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if options.innovation_decomposition:
        decomposition_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.decoder.GASF.innovation_decomposition_projection.parameters()
            if parameter.grad is not None
        )
        assert decomposition_grad > 0.0
        continuing = (
            full.real_mask[:, 8, 0].bool()
            & full.real_mask[:, 7, 0].bool()
        )
        assert torch.equal(
            second_window.innovation_valid[:, -1, 0], continuing
        )
        assert torch.isfinite(second_window.innovation_whitened).all()
        assert torch.isfinite(second_window.innovation_covariance).all()
    else:
        activation_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.decoder.SEAN.activation_head.parameters()
            if parameter.grad is not None
        )
        assert activation_grad > 0.0
    if options.association_augmentation:
        torch.testing.assert_close(
            second_window.association_detection_mass[:, -1],
            output.association_detection_mass[:, 0],
        )
        assert torch.equal(
            second_window.innovation_valid[:, -1, 0],
            (
                full.real_mask[:, 8, 0].bool()
                & full.real_mask[:, 7, 0].bool()
                & output.association_valid[:, 0, 0]
            ),
        )
    if adaptive_q_mode != "legacy":
        q_grad = sum(
            float(parameter.grad.abs().sum()) for parameter in model.decoder.Q_innovation_encoder.parameters()
            if parameter.grad is not None
        )
        assert q_grad > 0.0
    if options.causal_innovation:
        causal_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.causal_innovation_corrector.parameters()
            if parameter.grad is not None
        )
        assert causal_grad > 0.0
    optimizer.step()
    if options.causal_innovation:
        causal_weight_after = (
            model.causal_innovation_corrector.fusion[-1].weight.detach()
        )
        assert torch.isfinite(causal_weight_after).all()
        assert not torch.equal(causal_weight_before, causal_weight_after)
    print({
        "nodes": full.x.size(0),
        "forecast_shape": tuple(output.states.shape),
        "loss": float(loss.detach()),
        "event_weight": float(weight[0]),
    })


if __name__ == "__main__":
    main()
