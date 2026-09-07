"""Create the immutable manifest for formal innovation ablations."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data_v3 import GroupTargetTrackingDatasetV3


ENTRY_FILES = (
    "config.py",
    "main.py",
    "scripts/run_formal_innovation_multiseed.ps1",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_manifest(dataset_root, split):
    dataset = GroupTargetTrackingDatasetV3(
        dataset_root,
        split,
        load_structure=False,
        load_analysis_metadata=False,
    )
    root = Path(dataset_root).resolve()
    trajectories = []
    for target_path, input_path in zip(dataset.target_files, dataset.input_files):
        trajectories.append({
            "target": str(target_path.resolve().relative_to(root)).replace("\\", "/"),
            "target_bytes": target_path.stat().st_size,
            "input": str(input_path.resolve().relative_to(root)).replace("\\", "/"),
            "input_bytes": input_path.stat().st_size,
        })
    return {
        "count": len(trajectories),
        "manifest_sha256": canonical_hash(trajectories),
        "trajectories": trajectories,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    cli = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code_files = set(ENTRY_FILES)
    for directory in ("data", "model", "train", "utils", "evaluation"):
        code_files.update(
            str(path.relative_to(root)).replace("\\", "/")
            for path in (root / directory).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    code = {}
    for relative in sorted(code_files):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = git_branch = "unavailable"

    protocol = {
        "name": "SGGT-V2 whitened-innovation formal ablation",
        "version": 1,
        "frozen_at": "2026-08-10 Asia/Shanghai",
        "git": {"branch": git_branch, "head": git_head},
        "code_files": code,
        "code_manifest_sha256": canonical_hash(code),
        "dataset_root": str(Path(cli.data_dir).resolve()),
        "dataset": {
            split: split_manifest(cli.data_dir, split)
            for split in ("train", "val", "iid_test", "ood_test")
        },
        "seeds": [123, 456, 789],
        "shared_training": {
            "train_trajectories": "all 1000 in frozen order",
            "validation_trajectories": "all 200 in frozen order",
            "supervision": "all legal causal target steps",
            "real_mask": "current_real & previous_real",
            "encoder": "one initial window then streaming updates",
            "train_batch_size": 4,
            "validation_batch_size": 4,
            "optimizer": "AdamW",
            "weight_decay": 0.0001,
            "covariance_transition": "cv",
            "gasf_filter": "abc monotone low-pass",
            "gasf_fusion": "concat",
            "checkpoint_primary_metric": "position prediction RMSE",
            "temporal_position_min_delta_m": 0.05,
            "full_expert_min_delta_task_loss": 0.0001,
            "iid_ood_policy": "evaluate once only after every seed/model is frozen",
        },
        "models": {
            "A_temporal": {
                "graph_mode": "temporal_only",
                "SEAN": False,
                "innovation_decomposition": False,
                "initialization": "scratch per seed",
                "max_epochs": 60,
                "max_lr": 0.0006,
                "checkpoint_metric": "prediction",
            },
            "B_no_SEAN_fixed_graph": {
                "graph_mode": "graph",
                "SEAN": False,
                "innovation_decomposition": False,
                "initialization": "same-seed A_temporal best_prediction",
                "trainable": "GASF, concat fusion, auxiliary head",
                "max_epochs": 30,
                "max_lr": 0.0001,
                "checkpoint_metric": "prediction",
            },
            "C_full_innovation": {
                "graph_mode": "graph",
                "SEAN": True,
                "innovation_decomposition": True,
                "initialization": "same-seed A_temporal best_prediction",
                "trainable": "SEAN graph and innovation expert only",
                "online_training_rollout": "alpha=0 frozen Temporal comparator",
                "deployment_and_evaluation": "alpha=1",
                "max_epochs": 30,
                "max_lr": 0.0001,
                "checkpoint_metric": "counterfactual expert task loss",
            },
        },
        "reported_metrics": [
            "position prediction RMSE",
            "position update RMSE",
            "velocity prediction RMSE",
            "velocity update RMSE",
            "Gaussian NLL",
            "current NIS calibration",
            "group/individual innovation statistics",
            "steady/switch/strong event metrics",
            "parameter count",
            "online inference time",
        ],
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    output = Path(cli.output)
    if cli.verify:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(
                "formal protocol verification failed; code or dataset changed"
            )
        print(protocol["protocol_sha256"])
        return
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen protocol: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(protocol["protocol_sha256"])


if __name__ == "__main__":
    main()
