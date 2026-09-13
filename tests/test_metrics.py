import unittest
import torch
from src.training.metrics import Metrics
from evaluate import compute_metrics
import tempfile
from pathlib import Path


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.metrics = Metrics()

    def test_metrics_update_and_compute(self):
        outputs = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]])
        labels = torch.tensor([0, 1, 1])

        self.metrics.update(outputs, labels)
        results = self.metrics.compute()

        self.assertIn("accuracy", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1", results)
        self.assertAlmostEqual(results["accuracy"], 1.0)

    def test_compute_metrics_evaluation_helper(self):
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)

        results = {
            "labels": [0, 1, 0, 1],
            "predictions": [0, 1, 0, 1],
            "probabilities": [0.1, 0.9, 0.2, 0.8],
        }

        metrics, cm = compute_metrics(results, output_dir)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertTrue((output_dir / "metrics.json").exists())

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()