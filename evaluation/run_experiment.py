"""Unified SGGT experiment evaluation for baselines and ablations."""

import argparse
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data import GroupTargetTrackingDataset_load_whole_trajectory
try:
    from evaluation.evaluate_v2_structure import build_model
except ModuleNotFoundError:
    from evaluate_v2_structure import build_model
from train.trainer import Trainer
from train.training_utils import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


TRACK_KEYS = (
    'location_rmse_prediction', 'velocity_rmse_prediction',
    'location_rmse_update', 'velocity_rmse_update',
    'location_rmse_measurement',
)


def confusion_metrics(probabilities, targets, threshold):
    predictions = probabilities >= threshold
    targets = targets.bool()
    tp = int((predictions & targets).sum())
    fp = int((predictions & ~targets).sum())
    fn = int((~predictions & targets).sum())
    tn = int((~predictions & ~targets).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return {
        'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'balanced_accuracy': 0.5 * (recall + specificity),
        'f1': f1, 'mcc': (tp * tn - fp * fn) / denominator,
    }


def gaussian_nll(outputs):
    if outputs.covariances.numel() == 0:
        return float('nan')
    difference = (outputs.target.unsqueeze(2) - outputs.states).unsqueeze(-1)
    covariance = outputs.covariances
    eye = torch.eye(covariance.size(-1), device=covariance.device)
    covariance = covariance + 1e-5 * eye
    solution = torch.linalg.solve(covariance, difference)
    mahalanobis = (difference.transpose(-1, -2) @ solution).squeeze(-1).squeeze(-1)
    logdet = torch.linalg.slogdet(covariance).logabsdet
    best = (0.5 * (mahalanobis + logdet)).min(dim=2).values.clamp(-20.0, 100.0)
    mask = outputs.real_mask
    return float((best * mask).sum() / mask.sum().clamp_min(1.0))


def mean_rows(rows):
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


class Accumulator:
    def __init__(self):
        self.probabilities = []
        self.targets = []
        self.rows = []
        self.scenarios = defaultdict(list)
        self.min_eigenvalues = []
        self.symmetry_errors = []
        self.traces = []
        self.logdets = []
        self.non_positive_covariances = 0
        self.alignment_errors = 0
        self.forward_times_ms = []
        self.kalman_rows = []
        self.horizon_rows = defaultdict(list)

    def result(self, threshold, batch_size):
        probabilities = torch.cat(self.probabilities) if self.probabilities else torch.empty(0)
        targets = torch.cat(self.targets) if self.targets else torch.empty(0)
        scan = [confusion_metrics(probabilities, targets, round(x, 2))
                for x in np.arange(0.05, 1.0, 0.05)] if targets.numel() else []
        positive = targets.bool()
        return {
            'global': mean_rows(self.rows),
            'structure': {
                'valid_edges': int(targets.numel()),
                'positive_edges': int(positive.sum()),
                'positive_rate': float(positive.float().mean()) if targets.numel() else 0.0,
                'mean_positive_probability': float(probabilities[positive].mean()) if positive.any() else 0.0,
                'mean_negative_probability': float(probabilities[~positive].mean()) if (~positive).any() else 0.0,
                'at_configured_threshold': confusion_metrics(probabilities, targets, threshold) if targets.numel() else {},
                'best_f1_threshold': max(scan, key=lambda row: row['f1']) if scan else {},
                'best_balanced_accuracy_threshold': max(scan, key=lambda row: row['balanced_accuracy']) if scan else {},
                'all_positive_f1': (2 * int(positive.sum()) / max(int(targets.numel()) + int(positive.sum()), 1)),
                'alignment_errors': self.alignment_errors,
                'threshold_scan': scan,
            },
            'covariance': {
                'minimum_eigenvalue': min(self.min_eigenvalues, default=float('nan')),
                'mean_minimum_eigenvalue': float(np.mean(self.min_eigenvalues)) if self.min_eigenvalues else float('nan'),
                'maximum_symmetry_error': max(self.symmetry_errors, default=float('nan')),
                'mean_trace': float(np.mean(self.traces)) if self.traces else float('nan'),
                'mean_logdet': float(np.mean(self.logdets)) if self.logdets else float('nan'),
                'non_positive_count': self.non_positive_covariances,
            },
            'scenarios': {name: mean_rows(rows) for name, rows in self.scenarios.items()},
            'timing': {
                'mean_forward_batch_ms': float(np.mean(self.forward_times_ms)) if self.forward_times_ms else 0.0,
                'p95_forward_batch_ms': float(np.percentile(self.forward_times_ms, 95)) if self.forward_times_ms else 0.0,
                'estimated_mean_forward_ms_per_trajectory': (
                    float(np.mean(self.forward_times_ms)) / batch_size
                    if self.forward_times_ms else 0.0
                ),
                'measured_steps': len(self.forward_times_ms),
            },
            'kalman_update': mean_rows(self.kalman_rows),
            'horizon_curve': {
                str(step): mean_rows(rows)
                for step, rows in sorted(self.horizon_rows.items())
            },
        }


def kalman_diagnostics(outputs, measurement, measurement_covariance):
    """Return physical-unit EKF diagnostics for the one-step update."""
    predicted_state = outputs.states[:, 0, 0]
    target = outputs.target[:, 0]
    predicted_covariance = outputs.covariances[:, 0, 0]
    updated_state = outputs.updated_state[:, 0]
    dtype, device = predicted_state.dtype, predicted_state.device
    scales = torch.tensor(
        [CONFIG['S_POS'], CONFIG['S_POS'], CONFIG['S_VEL'], CONFIG['S_VEL']],
        dtype=dtype, device=device,
    )
    H = torch.zeros(2, 4, dtype=dtype, device=device)
    H[0, 0] = H[1, 1] = 1.0
    D = torch.diag(scales)
    H_bar = H @ D
    R_physical = measurement_covariance * CONFIG['S_POS'] ** 2
    innovation = CONFIG['S_POS'] * (
        measurement[:, 0] - predicted_state[:, :2]
    )
    S = H_bar.unsqueeze(0) @ predicted_covariance @ H_bar.T.unsqueeze(0) + R_physical
    PHt = predicted_covariance @ H_bar.T.unsqueeze(0)
    K = torch.linalg.solve(S, PHt.transpose(1, 2)).transpose(1, 2)
    nis = (
        innovation.unsqueeze(1)
        @ torch.linalg.solve(S, innovation.unsqueeze(-1))
    ).flatten()

    prediction_error = (predicted_state - target) * scales
    update_error = (updated_state - target) * scales
    prediction_position = torch.linalg.vector_norm(prediction_error[:, :2], dim=-1)
    update_position = torch.linalg.vector_norm(update_error[:, :2], dim=-1)
    prediction_velocity = torch.linalg.vector_norm(prediction_error[:, 2:], dim=-1)
    update_velocity = torch.linalg.vector_norm(update_error[:, 2:], dim=-1)
    cross_covariance = predicted_covariance[:, :2, 2:]
    return {
        'position_gain_frobenius': float(torch.linalg.matrix_norm(K[:, :2], dim=(-2, -1)).mean()),
        'velocity_gain_frobenius': float(torch.linalg.matrix_norm(K[:, 2:], dim=(-2, -1)).mean()),
        'position_velocity_cross_cov_frobenius': float(
            torch.linalg.matrix_norm(cross_covariance, dim=(-2, -1)).mean()
        ),
        'innovation_norm': float(torch.linalg.vector_norm(innovation, dim=-1).mean()),
        'nis_mean': float(nis.mean()),
        'nis_above_95pct_rate': float((nis > 5.991).float().mean()),
        'position_error_before_update': float(prediction_position.mean()),
        'position_error_after_update': float(update_position.mean()),
        'position_update_worsened_rate': float((update_position > prediction_position).float().mean()),
        'velocity_error_before_update': float(prediction_velocity.mean()),
        'velocity_error_after_update': float(update_velocity.mean()),
        'velocity_update_worsened_rate': float((update_velocity > prediction_velocity).float().mean()),
    }


def scenario_names(graph_data, time_step, radius):
    changes = graph_data.param_change_time.detach().reshape(-1)
    maneuver = bool(((changes > 1) & (changes < graph_data.y.shape[1])
                     & ((changes - time_step).abs() <= radius)).any())
    structure_change = False
    if time_step > 0:
        current = graph_data.struct_feat[time_step]
        previous = graph_data.struct_feat[time_step - 1]
        structure_change = current.shape != previous.shape or not torch.equal(current, previous)
    names = ['all']
    names.append('maneuver' if maneuver else 'non_maneuver')
    names.append('structure_change' if structure_change else 'structure_stable')
    return names


@torch.no_grad()
def evaluate_segment(model, trainer, graph_data, start_bias, rollout_steps,
                     accumulator, maneuver_radius, encoder_mode='once',
                     initial_context=None):
    if initial_context is None:
        graph_data = graph_data.clone()
        first_time = trainer.config['input_graph_window_len'] + start_bias
        measure_cov, measure_cov_norm = trainer.compute_measurement_noise(
            graph_data, first_time
        )
        graph_data = graph_data.to(trainer.device)
        measure_cov = measure_cov.to(trainer.device)
        measure_cov_norm = measure_cov_norm.to(trainer.device)
        graph_work = graph_data.clone()
        window = GraphDataProcessor.init_input_window(
            graph_work, trainer.config, trainer.args, start_bias=start_bias
        )
        normalized = window.clone()
        covariance = trainer.init_state_covariance(measure_cov)
        encoder_output_cache = None
        decoder_hidden_cache = None
        mixture_coeffs_cache = None
        gasf_recurrent_state = None
    else:
        graph_work = initial_context['graph_work']
        window = initial_context['window']
        normalized = initial_context['normalized']
        covariance = initial_context['covariance']
        measure_cov = initial_context['measure_cov']
        measure_cov_norm = initial_context['measure_cov_norm']
        first_time = initial_context['next_time']
        encoder_output_cache = initial_context['encoder_output_cache']
        decoder_hidden_cache = initial_context['decoder_hidden_cache']
        mixture_coeffs_cache = initial_context['mixture_coeffs_cache']
        gasf_recurrent_state = initial_context.get('gasf_recurrent_state')

    for rollout_index, time_step in enumerate(
        range(first_time, first_time + rollout_steps)
    ):
        measure_cov, measure_cov_norm = trainer.compute_measurement_noise(
            graph_work, time_step
        )
        measure_cov = measure_cov.to(trainer.device)
        measure_cov_norm = measure_cov_norm.to(trainer.device)
        normalized, centroids = minmax_scaler(
            window, normalized, trainer.config, trainer.args.neural_net
        )
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(
            normalized, 0.0, covariance, None, measure_cov_norm,
            encoder_output_cache=(
                encoder_output_cache if encoder_mode == 'once' else None
            ),
            decoder_hidden_cache=decoder_hidden_cache,
            mixture_coeffs_cache=(
                mixture_coeffs_cache if encoder_mode == 'once' else None
            ),
            gasf_recurrent_state=gasf_recurrent_state,
        )
        encoder_output_cache = outputs.encoder_output_cache
        decoder_hidden_cache = outputs.decoder_hidden_cache
        mixture_coeffs_cache = outputs.mixture_coeffs_cache
        gasf_recurrent_state = outputs.gasf_recurrent_state
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize()
        accumulator.forward_times_ms.append((time.perf_counter() - start) * 1000)

        expected = graph_work.struct_feat[time_step].to(outputs.adjacency)
        current = window.struct_feat[0].to(outputs.adjacency)
        if expected.shape != current.shape or not torch.equal(expected, current):
            accumulator.alignment_errors += 1
        if outputs.adjacency.numel() and expected.numel() == outputs.adjacency.size(0):
            target = expected.reshape(-1, 1).expand_as(outputs.adjacency)
            valid = outputs.adjacency_mask.bool()
            accumulator.probabilities.append(outputs.adjacency[valid].cpu())
            accumulator.targets.append(target[valid].cpu())

        metrics, updated_state = trainer.compute_evaluation_metrics(window, outputs, centroids)
        row = {key: metrics[key] for key in TRACK_KEYS}
        row['gaussian_nll'] = gaussian_nll(outputs)
        gate = outputs.film_gate.detach()
        row['film_gate_mean'] = float(gate.mean())
        row['film_gate_below_001_rate'] = float((gate < 0.01).float().mean())
        row['film_gate_above_09_rate'] = float((gate > 0.9).float().mean())
        policy_gate = outputs.policy_gate.detach()
        confidence = outputs.gasf_confidence.detach()
        row['policy_gate_mean'] = float(policy_gate.mean())
        row['gasf_confidence_mean'] = float(confidence.mean())
        if outputs.counterfactual_base.numel():
            correction = (
                outputs.counterfactual_expert[:, 0, 0]
                - outputs.counterfactual_base[:, 0, 0]
            )
            row['gasf_position_correction_norm'] = float(
                torch.linalg.vector_norm(
                    correction[:, :2] * CONFIG['S_POS'], dim=-1
                ).mean()
            )
            row['gasf_velocity_correction_norm'] = float(
                torch.linalg.vector_norm(
                    correction[:, 2:] * CONFIG['S_VEL'], dim=-1
                ).mean()
            )
        if outputs.auxiliary_delta_v.numel():
            auxiliary = outputs.auxiliary_delta_v[:, 0]
            target_delta = outputs.target[:, 0, 2:] - normalized.x[:, -1, 2:]
            row['aux_delta_v_rmse'] = float(
                torch.mean((auxiliary - target_delta).square()).sqrt()
            )
            cosine = torch.nn.functional.cosine_similarity(
                auxiliary, target_delta, dim=-1, eps=1e-8
            )
            row['aux_delta_v_cosine'] = float(cosine.mean())
        gasf = getattr(model.decoder, 'GASF', None)
        diagnostics = getattr(gasf, 'last_filter_diagnostics', []) if gasf else []
        for key in (
            'lowpass_dc_gain', 'lowpass_nyquist_gain', 'lowpass_mean_gain',
            'low_high_energy_ratio', 'a', 'b', 'c',
            'parameter_update_gate', 'parameter_step_norm',
            'filter_response_step_rms',
        ):
            values = [entry[key] for entry in diagnostics if key in entry]
            if values:
                row[f'gasf_{key}'] = float(torch.stack(values).mean())
        accumulator.rows.append(row)
        accumulator.horizon_rows[rollout_index].append(row)
        accumulator.kalman_rows.append(kalman_diagnostics(
            outputs, normalized.measurement_future, measure_cov_norm
        ))
        for name in scenario_names(graph_work, time_step, maneuver_radius):
            accumulator.scenarios[name].append(row)

        updated_covariance = outputs.updated_covariance.detach()
        eigenvalues = torch.linalg.eigvalsh(updated_covariance)
        accumulator.min_eigenvalues.append(float(eigenvalues.min()))
        accumulator.non_positive_covariances += int((eigenvalues <= 0).any(dim=-1).sum())
        accumulator.symmetry_errors.append(float(
            (updated_covariance - updated_covariance.transpose(-1, -2)).abs().max()
        ))
        accumulator.traces.append(float(updated_covariance.diagonal(dim1=-2, dim2=-1).sum(-1).mean()))
        accumulator.logdets.append(float(torch.linalg.slogdet(updated_covariance).logabsdet.mean()))
        covariance = updated_covariance
        bias = time_step - trainer.config['input_graph_window_len'] + 1
        window = GraphDataProcessor.update_input_window(
            window, graph_work, trainer.config, updated_state, bias, trainer.args
        )
    return {
        'graph_work': graph_work,
        'window': window,
        'normalized': normalized,
        'covariance': covariance,
        'measure_cov': measure_cov,
        'measure_cov_norm': measure_cov_norm,
        'next_time': first_time + rollout_steps,
        'encoder_output_cache': encoder_output_cache,
        'decoder_hidden_cache': decoder_hidden_cache,
        'mixture_coeffs_cache': mixture_coeffs_cache,
        'gasf_recurrent_state': gasf_recurrent_state,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data_dir', default='E:/Dataset/GroupTargetsDataset_dataRate_2_v8_pack/')
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--mode', choices=['fixed', 'full', 'both'], default='full')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--rollout_steps', type=int, default=8)
    parser.add_argument('--encoder_mode', choices=['once', 'every_step'],
                        default='once')
    parser.add_argument('--fixed_starts', choices=['midpoint', 'three', 'start'],
                        default='midpoint')
    parser.add_argument('--burn_in_steps', type=int, default=0,
                        help='Unscored recurrent steps before each fixed window')
    parser.add_argument('--structure_threshold', type=float, default=0.5)
    parser.add_argument('--maneuver_radius', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    cli = parse_args()
    device = torch.device(cli.device)
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    args = Args().initialize(Args().parse()).parse_args([])
    args.device = device
    args.model_variant = checkpoint.get('model_variant', 'sggt_v1')
    args.covariance_transition = checkpoint.get('covariance_transition', 'cv')
    args.decoder_use_SEAN = checkpoint.get('use_SEAN', True)
    args.decoder_use_GASF = checkpoint.get('use_GASF', True)
    args.gasf_filter_mode = checkpoint.get('gasf_filter_mode', 'abc')
    args.gasf_graph_mode = checkpoint.get('gasf_graph_mode', 'graph')
    args.gasf_recurrent_filter = checkpoint.get(
        'gasf_recurrent_filter', False
    )
    args.gasf_recurrent_gate_initial_update = checkpoint.get(
        'gasf_recurrent_gate_initial_update', 0.1
    )
    args.gasf_fusion_mode = checkpoint.get('gasf_fusion_mode', 'film')
    args.gasf_aux_source = checkpoint.get('gasf_aux_source', 'gasf')
    args.counterfactual_training = checkpoint.get('counterfactual_training', False)
    model = build_model(args).to(device)
    incompatible = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"checkpoint compatibility: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = Trainer(model, optimizer, scheduler, None, CONFIG, args)
    dataset = GroupTargetTrackingDataset_load_whole_trajectory(cli.data_dir, cli.split)
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=0)
    accumulators = {name: Accumulator() for name in ('fixed', 'full')}
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()

    for batch_index, graph_data in enumerate(loader):
        if cli.max_batches and batch_index >= cli.max_batches:
            break
        total = graph_data.y.shape[1]
        history = CONFIG['input_graph_window_len']
        if cli.mode in ('fixed', 'both'):
            rollout = min(cli.rollout_steps, total - history - CONFIG['output_graph_window_len'])
            max_start = total - history - CONFIG['output_graph_window_len'] - rollout
            if cli.fixed_starts == 'three':
                starts = sorted(set((0, max_start // 2, max_start)))
            elif cli.fixed_starts == 'start':
                starts = [0]
            else:
                starts = [max_start // 2]
            for start in starts:
                context = None
                if cli.burn_in_steps > 0 and start > 0:
                    burn_in = min(cli.burn_in_steps, start)
                    context = evaluate_segment(
                        model, trainer, graph_data, start - burn_in, burn_in,
                        Accumulator(), cli.maneuver_radius, cli.encoder_mode
                    )
                evaluate_segment(model, trainer, graph_data, start, rollout,
                                 accumulators['fixed'], cli.maneuver_radius,
                                 cli.encoder_mode, context)
        if cli.mode in ('full', 'both'):
            rollout = total - history - CONFIG['output_graph_window_len']
            evaluate_segment(model, trainer, graph_data, 0, rollout,
                             accumulators['full'], cli.maneuver_radius,
                             cli.encoder_mode)
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(f'evaluated batch {batch_index + 1}', flush=True)

    result = {
        'metadata': {
            'experiment_name': cli.experiment_name,
            'checkpoint': str(Path(cli.checkpoint).resolve()),
            'checkpoint_epoch': checkpoint.get('epoch'),
            'split': cli.split,
            'model_variant': args.model_variant,
            'use_SEAN': args.decoder_use_SEAN,
            'use_GASF': args.decoder_use_GASF,
            'covariance_transition': args.covariance_transition,
            'total_parameter_count': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameter_count': sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            'seed': cli.seed,
            'torch_version': torch.__version__,
            'python_version': platform.python_version(),
            'device': str(device),
        },
        'config': vars(cli),
        'results': {
            name: accumulator.result(cli.structure_threshold, cli.batch_size)
            for name, accumulator in accumulators.items() if accumulator.rows
        },
        'resources': {
            'wall_time_seconds': time.perf_counter() - wall_start,
            'peak_cpu_rss_gb': peak_rss / 2 ** 30,
            'peak_cuda_allocated_gb': (
                torch.cuda.max_memory_allocated(device) / 2 ** 30
                if device.type == 'cuda' else 0.0
            ),
            'peak_cuda_reserved_gb': (
                torch.cuda.max_memory_reserved(device) / 2 ** 30
                if device.type == 'cuda' else 0.0
            ),
        },
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()
