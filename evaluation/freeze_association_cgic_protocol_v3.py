"""Create or verify the immutable association-aware CGIC formal protocol."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.load_data_v3 import GroupTargetTrackingDatasetV3


ENTRY_FILES = (
    "config.py",
    "main_v3.py",
    "scripts/run_formal_association_cgic_multiseed.ps1",
    "scripts/run_missing_temporal_baselines.ps1",
    "scripts/run_paper_experiment_pipeline.ps1",
    "scripts/run_paper_experiment_pipeline.cmd",
    "scripts/run_physical_baselines_v3.ps1",
    "scripts/run_bp_tracker_robustness_matrix.ps1",
    "evaluation/verify_temporal_checkpoints_v3.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_manifest(dataset_root, split):
    dataset = GroupTargetTrackingDatasetV3(
        dataset_root,
        split,
        load_structure=False,
        load_analysis_metadata=False,
    )
    root = Path(dataset_root).resolve()
    trajectories = [
        {
            "target": str(target.resolve().relative_to(root)).replace("\\", "/"),
            "target_bytes": target.stat().st_size,
            "target_sha256": sha256_file(target),
            "input": str(model_input.resolve().relative_to(root)).replace(
                "\\", "/"
            ),
            "input_bytes": model_input.stat().st_size,
            "input_sha256": sha256_file(model_input),
        }
        for target, model_input in zip(dataset.target_files, dataset.input_files)
    ]
    return {
        "count": len(trajectories),
        "manifest_sha256": canonical_hash(trajectories),
        "trajectories": trajectories,
    }


def build_protocol(root, data_dir, frozen_at=None):
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
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    protocol = {
        "name": "Association-aware causal group-innovation correction",
        "version": 1,
        "frozen_at": frozen_at or datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "git": {
            "branch": git_branch,
            "head": git_head,
            "working_tree_changes_are_included_in_code_manifest": True,
        },
        "code_files": code,
        "code_manifest_sha256": canonical_hash(code),
        "dataset_root": str(Path(data_dir).resolve()),
        "dataset": {
            split: split_manifest(data_dir, split)
            for split in ("train", "val", "iid_test", "ood_test")
        },
        "seeds": [123, 456, 789],
        "shared_training": {
            "train_trajectories": 1000,
            "validation_trajectories": 200,
            "supervision": "all legal causal target steps",
            "real_mask": "target exists at previous and current frame",
            "encoder": "one initial window then streaming updates",
            "train_batch_size": 4,
            "validation_batch_size": 4,
            "optimizer": "AdamW",
            "maximum_lr": 0.0001,
            "minimum_lr": 0.000003,
            "warmup_epochs": 1,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 5.0,
            "covariance_transition": "cv",
            "checkpoint_metric": "fixed association position prediction+update RMSE",
            "early_stopping_min_delta_m": 0.05,
            "early_stopping_patience": 6,
            "maximum_epochs": 30,
        },
        "causal_model": {
            "temporal_initialization": "same-seed frozen Temporal best_prediction",
            "history_frames": 8,
            "minimum_valid_history": 4,
            "recent_valid_window": 3,
            "native_alpha": 1.0,
            "group_labels": False,
            "association_truth_input": False,
            "variants": {
                "P3_own_only": {"peer": False, "association_context": True},
                "P4_full": {"peer": True, "association_context": True},
                "P5_no_association_context": {
                    "peer": True,
                    "association_context": False,
                },
            },
        },
        "training_augmentation": {
            "pd_uniform": [0.75, 0.98],
            "global_clutter_mean": [0.0, 0.0],
            "local_clutter_rate": 0.25,
            "maximum_local_clutter_candidates": 2,
            "local_clutter_scale": 2.5,
            "target_competitor_candidates": False,
            "clutter_requires_target_detection": True,
        },
        "evaluation": {
            "splits": ["val", "iid_test", "ood_test"],
            "fixed_association_seed": 20260813,
            "events": ["steady", "switch", "strong"],
            "bootstrap_unit": "trajectory",
            "bootstrap_samples": 10000,
            "full_bp_actual_pd": [0.75, 0.85, 0.95],
            "full_bp_actual_clutter_mean": [10, 20, 30],
            "full_bp_nominal_pd": 0.85,
            "full_bp_nominal_clutter_mean": 20,
            "full_bp_adaptive_detection_probability": False,
            "physical_baseline_selection_split": "val",
            "physical_baseline_selection_metric": (
                "prediction position RMSE + update position RMSE"
            ),
            "physical_baseline_test_tuning": False,
        },
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if args.verify:
        existing = json.loads(output.read_text("utf-8"))
        current = build_protocol(
            root, args.data_dir, frozen_at=existing["frozen_at"]
        )
        if existing != current:
            raise RuntimeError("formal association-CGIC protocol verification failed")
        print(current["protocol_sha256"])
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")
    protocol = build_protocol(root, args.data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(protocol["protocol_sha256"])


if __name__ == "__main__":
    main()
