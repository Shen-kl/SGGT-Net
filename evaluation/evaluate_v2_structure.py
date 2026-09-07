import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, Args
from data.load_data import GroupTargetTrackingDataset_load_whole_trajectory
from model.motion_models_base import SecondOrderNeuralODE_groupTrack
from model.sggt_net import SGGT_Net
from train.trainer import Trainer
from train.training_utils import GraphDataProcessor
from utils.minmax_scaler import minmax_scaler


def build_model(args):
    motion_model = SecondOrderNeuralODE_groupTrack(
        solver=args.ode_solver,
        dt=args.T,
        mixtures=args.n_mixtures,
        static_f_dim=0,
        n_hidden=args.n_ode_hidden,
        n_layers=args.n_ode_layers,
        use_SEAN=args.decoder_use_SEAN,
        use_GASF=args.decoder_use_GASF,
        gasf_fusion_mode=getattr(args, 'gasf_fusion_mode', 'film'),
        gasf_aux_source=getattr(args, 'gasf_aux_source', 'gasf'),
    )
    return SGGT_Net(
        args.encoder_input_size, args.encoder_hidden_size,
        args.encoder_n_heads, args.encoder_n_layers, motion_model.mixtures,
        args.encoder_dropout, args.encoder_gnn_layer,
        args.encoder_use_edge_features, motion_model,
        args.decoder_max_length, args.decoder_hidden_size,
        args.decoder_n_heads, args.decoder_n_layers, args.decoder_alpha,
        args.decoder_dropout, args.decoder_residual_length,
        args.decoder_z_dimension, args.decoder_gnn_layer,
        args.decoder_use_GASF, args.decoder_use_SEAN, args.T,
        model_variant=args.model_variant,
        covariance_transition=args.covariance_transition,
        gasf_filter_mode=args.gasf_filter_mode,
        gasf_graph_mode=getattr(args, 'gasf_graph_mode', 'graph'),
        counterfactual_training=getattr(args, 'counterfactual_training', False),
    )


def threshold_metrics(probabilities, targets, threshold):
    predictions = probabilities >= threshold
    targets = targets.bool()
    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1}


@torch.no_grad()
def evaluate_segment(model, trainer, graph_data, start_bias, rollout_steps):
    graph_data = graph_data.clone()
    measure_cov, measure_cov_norm = trainer.compute_measurement_noise(graph_data, start_bias)
    graph_data = graph_data.to(trainer.device)
    measure_cov = measure_cov.to(trainer.device)
    measure_cov_norm = measure_cov_norm.to(trainer.device)
    graph_work = graph_data.clone()
    window = GraphDataProcessor.init_input_window(
        graph_work, trainer.config, trainer.args, start_bias=start_bias
    )
    normalized = window.clone()
    covariance = trainer.init_state_covariance(measure_cov)
    probabilities, targets = [], []
    alignment_errors = 0
    trajectory_metrics = []
    first_time = trainer.config['input_graph_window_len'] + start_bias

    for time_step in range(first_time, first_time + rollout_steps):
        normalized, centroids = minmax_scaler(
            window, normalized, trainer.config, trainer.args.neural_net
        )
        outputs = model(normalized, 0.0, covariance, None, measure_cov_norm)
        expected = graph_work.struct_feat[time_step].to(outputs.adjacency)
        current = window.struct_feat[0].to(outputs.adjacency)
        if expected.shape != current.shape or not torch.equal(expected, current):
            alignment_errors += 1

        adjacency = outputs.adjacency
        mask = outputs.adjacency_mask.bool()
        target = expected.reshape(-1)
        if adjacency.numel() > 0 and target.numel() == adjacency.size(0):
            target = target.view(-1, 1).expand_as(adjacency)
            probabilities.append(adjacency[mask].detach().cpu())
            targets.append(target[mask].detach().cpu())

        metrics, updated_state = trainer.compute_evaluation_metrics(
            window, outputs, centroids
        )
        trajectory_metrics.append(metrics)
        covariance = outputs.updated_covariance.detach()
        bias = time_step - trainer.config['input_graph_window_len'] + 1
        window = GraphDataProcessor.update_input_window(
            window, graph_work, trainer.config, updated_state, bias, trainer.args
        )

    return probabilities, targets, alignment_errors, trajectory_metrics


def summarize(probability_parts, target_parts, alignment_errors, metric_rows):
    probabilities = torch.cat(probability_parts) if probability_parts else torch.empty(0)
    targets = torch.cat(target_parts) if target_parts else torch.empty(0)
    thresholds = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    scan = [threshold_metrics(probabilities, targets, threshold) for threshold in thresholds]
    best = max(scan, key=lambda row: row['f1']) if scan else {}
    at_half = threshold_metrics(probabilities, targets, 0.5) if scan else {}
    positive = targets > 0.5
    return {
        'edges': int(targets.numel()),
        'positive_edges': int(positive.sum().item()),
        'positive_rate': float(positive.float().mean().item()) if targets.numel() else 0.0,
        'mean_positive_probability': float(probabilities[positive].mean().item()) if positive.any() else 0.0,
        'mean_negative_probability': float(probabilities[~positive].mean().item()) if (~positive).any() else 0.0,
        'f1_at_0.5': at_half,
        'best_threshold': best,
        'alignment_errors': alignment_errors,
        'trajectory_metrics': {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        } if metric_rows else {},
        'threshold_scan': scan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='E:/Dataset/GroupTargetsDataset_dataRate_2_v8_pack/')
    parser.add_argument('--max_trajectories', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mode', choices=['fixed', 'full', 'both'], default='both')
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()

    args = Args().initialize(Args().parse()).parse_args([])
    args.data_dir = cli.data_dir
    args.train_batch_size = 1
    args.eval_batch_size = 1
    args.model_variant = 'sggt_v2'
    args.covariance_transition = 'ode_jacobian'
    args.decoder_use_SEAN = True
    args.decoder_use_GASF = True
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args).to(args.device)
    checkpoint = torch.load(cli.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = Trainer(model, optimizer, scheduler, None, CONFIG, args)
    dataset = GroupTargetTrackingDataset_load_whole_trajectory(cli.data_dir, 'test')
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=0)

    accumulators = {
        'fixed': {'p': [], 't': [], 'a': 0, 'm': []},
        'full': {'p': [], 't': [], 'a': 0, 'm': []},
    }
    for trajectory_index, graph_data in enumerate(loader):
        if cli.max_trajectories and trajectory_index >= cli.max_trajectories:
            break
        total = graph_data.y.shape[1]
        history = CONFIG['input_graph_window_len']
        if cli.mode in ('fixed', 'both'):
            rollout = 8
            max_start = total - history - CONFIG['output_graph_window_len'] - rollout
            for start in (0, max_start // 2, max_start):
                p, t, a, m = evaluate_segment(model, trainer, graph_data, start, rollout)
                accumulators['fixed']['p'].extend(p)
                accumulators['fixed']['t'].extend(t)
                accumulators['fixed']['a'] += a
                accumulators['fixed']['m'].extend(m)
        if cli.mode in ('full', 'both'):
            rollout = total - history - CONFIG['output_graph_window_len']
            p, t, a, m = evaluate_segment(model, trainer, graph_data, 0, rollout)
            accumulators['full']['p'].extend(p)
            accumulators['full']['t'].extend(t)
            accumulators['full']['a'] += a
            accumulators['full']['m'].extend(m)
        print(f'evaluated batch {trajectory_index + 1}', flush=True)

    result = {
        key: summarize(value['p'], value['t'], value['a'], value['m'])
        for key, value in accumulators.items()
        if value['m']
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
