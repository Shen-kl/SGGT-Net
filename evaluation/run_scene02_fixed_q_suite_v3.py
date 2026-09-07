"""Run the complete frozen scene02 fixed-Q experiment suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "paper_scene02_fixed_q_protocol_v2.json"
DEFAULT_OUTPUT = ROOT / "results" / "paper_scene02_fixed_q_20260824"
MAIN_RUNNER = ROOT / "evaluation" / "run_paper_scene02_protocol_v3.py"
ROBUSTNESS_RUNNER = ROOT / "evaluation" / "run_scene02_fixed_q_robustness_v3.py"
SUMMARIZER = ROOT / "evaluation" / "summarize_scene02_fixed_q_suite_v3.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    common = [
        "--manifest", str(args.manifest.resolve()),
        "--output", str(args.output.resolve()),
        "--mode", args.mode,
        "--device", args.device,
    ]
    subprocess.run([sys.executable, str(MAIN_RUNNER), *common], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, str(ROBUSTNESS_RUNNER), *common], cwd=ROOT, check=True
    )
    if args.mode == "formal":
        subprocess.run(
            [
                sys.executable,
                str(SUMMARIZER),
                "--manifest", str(args.manifest.resolve()),
                "--results", str(args.output.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
