"""Pure CPU tests for validation comparisons; no user data required."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.compare_cached_models import load_checkpoint_model, summarize
from scripts.train_cached_bilstm import CachedBiLSTM


class TestValidationComparison(unittest.TestCase):
    def test_direct_script_entrypoint(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "compare_cached_models.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cache-root", result.stdout)

    def test_confusion_matrix_and_scores(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.7, 0.9, 0.8])
        stats = summarize(labels, scores)
        self.assertEqual(stats["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(stats["accuracy"], 0.75)
        self.assertAlmostEqual(stats["recall_fake"], 1.0)
        self.assertAlmostEqual(stats["specificity_real"], 0.5)

    def test_checkpoint_loader_accepts_correct_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pt"
            model = CachedBiLSTM(hidden_dim=8)
            torch.save({
                "architecture": "cached_efficientnet_b2_bilstm",
                "feature_dim": 1408,
                "hidden_dim": 8, "dropout": 0.4,
                "model_state_dict": model.state_dict(),
            }, checkpoint)
            restored = load_checkpoint_model(checkpoint)
            self.assertEqual(
                tuple(restored(torch.zeros(1, 3, 1408)).shape), (1, 2)
            )

    def test_reject_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "wrong.pt"
            torch.save({"architecture": "vit_temporal_pooling"}, checkpoint)
            with self.assertRaisesRegex(ValueError, "not a cached"):
                load_checkpoint_model(checkpoint)


if __name__ == "__main__":
    unittest.main()
