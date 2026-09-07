from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def _torch_load(path):
    """Load trusted, locally generated dataset files across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _window_mask(length, starts):
    mask = torch.zeros(length, dtype=torch.bool)
    if starts is None:
        return mask
    starts = torch.as_tensor(starts, dtype=torch.long).reshape(-1)
    starts = starts[(starts >= 0) & (starts < length)]
    if starts.numel():
        mask[starts] = True
    return mask


class GroupTargetTrackingDataset_load_whole_trajectory(Dataset):
    """Load packed whole trajectories from legacy or Dataset2D v3 PT files."""

    def __init__(self, dataset_path: str, mode: str, load_structure: bool = True,
                 load_analysis_metadata: bool = False):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.load_structure = load_structure
        self.load_analysis_metadata = load_analysis_metadata
        self.target_files = sorted(
            (self.dataset_path / mode / "target").glob("batch_data_*.pt")
        )
        if not self.target_files:
            raise FileNotFoundError(
                f"No batch_data_*.pt files in {self.dataset_path / mode / 'target'}"
            )
        self.input_files = [
            self.dataset_path / mode / "input" / path.name
            for path in self.target_files
        ]
        missing = [str(path) for path in self.input_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing paired input PT file: {missing[0]}")

    def len(self):
        return len(self.target_files)

    def get(self, idx):
        idx = int(idx)
        input_data = _torch_load(self.input_files[idx])
        target_data = _torch_load(self.target_files[idx])

        measurement = input_data["graph_measurement"].float()
        if "graph_initial_state" in input_data:
            state_input = input_data["graph_initial_state"].float()
            is_v3 = True
        else:
            state_input = input_data["graph_estimation"].float()
            is_v3 = False

        nan_mask = input_data["nan_mask"].bool()
        edge_index = [value.long() for value in input_data["graph_input_edge_index"]]
        edge_features = [value.float() for value in input_data["graph_input_edge_feature"]]

        target = target_data["graph_target"].float()
        real_mask = target_data["real_mask"].bool()

        frame_count = target.size(1)
        steady_window_mask = _window_mask(
            frame_count, input_data.get("steady_windows")
        )
        switch_window_mask = _window_mask(
            frame_count, input_data.get("switch_windows")
        )
        strong_window_mask = _window_mask(
            frame_count, input_data.get("strong_windows")
        )

        # Old datasets have no event metadata. Treat every legal start uniformly.
        if not is_v3:
            steady_window_mask[:] = True

        data = Data(
            x=state_input,
            y=target,
            measurement=measurement,
            nan_mask=nan_mask,
            real_mask=real_mask,
            edge_index=edge_index,
            edge_features=edge_features,
            steady_window_mask=steady_window_mask.unsqueeze(0),
            switch_window_mask=switch_window_mask.unsqueeze(0),
            strong_window_mask=strong_window_mask.unsqueeze(0),
            range_noise_std=torch.as_tensor(
                input_data.get("range_noise_std", 5.0), dtype=torch.float32
            ).reshape(1),
            azimuth_noise_std_deg=torch.as_tensor(
                input_data.get("azimuth_noise_std_deg", 0.2), dtype=torch.float32
            ).reshape(1),
            is_v3=torch.tensor([is_v3], dtype=torch.bool),
            # PyG treats attribute names containing ``index`` as index tensors
            # and increments them by the node offset during batching.  Keep the
            # legacy field for batch=1 compatibility, but use ``trajectory_id``
            # whenever trajectories must be paired across batched evaluations.
            trajectory_id=torch.tensor([idx], dtype=torch.long),
            trajectory_index=torch.tensor([idx], dtype=torch.long),
        )
        if self.load_structure:
            data.struct_feat = [
                value.float().reshape(-1)
                for value in target_data.get("graph_struct_total", [])
            ]
        if self.load_analysis_metadata:
            data.switch_mask = input_data.get(
                "switch_mask", torch.zeros(frame_count, dtype=torch.bool)
            ).bool().reshape(1, -1)
            data.strong_mask = input_data.get(
                "strong_mask", torch.zeros(frame_count, dtype=torch.bool)
            ).bool().reshape(1, -1)
            data.membership_event_mask = input_data.get(
                "membership_event_mask", torch.zeros(frame_count, dtype=torch.bool)
            ).bool().reshape(1, -1)
            data.mode_id = input_data.get(
                "mode_id", torch.zeros(frame_count, dtype=torch.long)
            ).long().reshape(1, -1)
            data.scene_id = torch.as_tensor(
                input_data.get("scene_id", idx), dtype=torch.long
            ).reshape(1)
        if "param_change_time" in input_data:
            data.param_change_time = input_data["param_change_time"].float()
        return data


GroupTargetTrackingDatasetV3 = GroupTargetTrackingDataset_load_whole_trajectory


if __name__ == "__main__":
    dataset = GroupTargetTrackingDataset_load_whole_trajectory(
        "D:/Dataset/GroupTargetsDataset_v3_pack/", "train"
    )
    sample = dataset.get(0)
    print(sample)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(next(iter(dataloader)))
