import unittest

import torch
from torch_geometric.data import Data

from evaluation.benchmark_model_scaling_v3 import replicate_window


class ModelScalingBenchmarkTest(unittest.TestCase):
    def test_replication_offsets_edges_without_cross_block_links(self):
        window = Data(
            x=torch.zeros(5, 8, 4),
            y=torch.zeros(5, 1, 4),
            edge_index=[torch.tensor([[0, 1], [1, 0]])],
            edge_features=[torch.zeros(2, 2)],
        )
        result = replicate_window(window, 2)
        self.assertEqual(tuple(result.x.shape), (10, 8, 4))
        self.assertTrue(torch.equal(
            result.edge_index[0],
            torch.tensor([[0, 1, 5, 6], [1, 0, 6, 5]]),
        ))
        self.assertEqual(int(result.batch.sum()), 0)


if __name__ == "__main__":
    unittest.main()
