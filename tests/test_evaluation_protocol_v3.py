import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import main_v3
from torch.utils.data import Subset


class _ManifestDataset:
    def __init__(self, root, count=5):
        self.mode = "val"
        self.target_files = []
        self.input_files = []
        for index in range(count):
            target = root / "val" / "target" / f"batch_data_{index:05d}.pt"
            model_input = root / "val" / "input" / f"batch_data_{index:05d}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            model_input.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes([index + 1]))
            model_input.write_bytes(bytes([index + 2, index + 3]))
            self.target_files.append(target)
            self.input_files.append(model_input)

    def __len__(self):
        return len(self.target_files)


class EvaluationProtocolTest(unittest.TestCase):
    def test_manifest_records_the_selected_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            dataset_root = tmp_path / "dataset"
            dataset = _ManifestDataset(dataset_root)
            subset = Subset(dataset, [1, 3])
            runtime_args = SimpleNamespace(
                encoder_execution_mode="stream",
                encoder_stream_reset_interval=0,
                full_time_supervision=True,
                covariance_transition="cv",
            )

            with patch.object(main_v3, "args", runtime_args, create=True):
                protocol = main_v3.build_evaluation_protocol(
                    subset,
                    dataset_root,
                    {"input_graph_window_len": 8, "output_graph_window_len": 1},
                    tmp_path / "run",
                )

            self.assertEqual(protocol["trajectory_count"], 2)
            self.assertEqual(protocol["version"], 2)
            self.assertEqual(
                protocol["training_supervision"], "all_legal_causal_steps"
            )
            self.assertEqual(protocol["covariance_transition"], "cv")
            self.assertEqual(
                [entry["target"] for entry in protocol["trajectories"]],
                [
                    "val\\target\\batch_data_00001.pt",
                    "val\\target\\batch_data_00003.pt",
                ],
            )
            persisted = json.loads(
                (tmp_path / "run" / "evaluation_protocol.json").read_text("utf-8")
            )
            self.assertEqual(
                persisted["protocol_sha256"], protocol["protocol_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
