import unittest

import torch

from evaluation.run_experiment import confusion_metrics


class UnifiedEvaluationTest(unittest.TestCase):
    def test_confusion_metrics_include_imbalance_aware_scores(self):
        probabilities = torch.tensor([0.9, 0.8, 0.6, 0.4, 0.1])
        targets = torch.tensor([1, 1, 0, 1, 0])

        metrics = confusion_metrics(probabilities, targets, 0.5)

        self.assertEqual(metrics['tp'], 2)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(metrics['tn'], 1)
        self.assertAlmostEqual(metrics['precision'], 2 / 3)
        self.assertAlmostEqual(metrics['recall'], 2 / 3)
        self.assertAlmostEqual(metrics['specificity'], 1 / 2)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 7 / 12)
        self.assertTrue(-1.0 <= metrics['mcc'] <= 1.0)


if __name__ == '__main__':
    unittest.main()
