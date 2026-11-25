"""Fast CPU-only tests for inference inputs and sequence alignment."""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.predict_cached_bilstm import prepare_sequence, select_uniform_indices


class TestCachedInference(unittest.TestCase):
    def test_same_normalization_as_training_loader(self):
        # Constant uint8 RGB image; verify ImageNet normalization per channel.
        seq = np.full((30, 260, 260, 3), 128, dtype=np.uint8)
        result = prepare_sequence(seq)
        self.assertEqual(tuple(result.shape), (30, 3, 224, 224))
        expected = (128.0 / 255.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), expected, places=5)
        self.assertEqual(result.dtype, torch.float32)

    def test_uniform_face_sampling(self):
        np.testing.assert_array_equal(
            select_uniform_indices(60), np.linspace(0, 59, 30).astype(int)
        )
        with self.assertRaisesRegex(ValueError, "at least 30"):
            select_uniform_indices(29)

    def test_reject_invalid_sequence(self):
        with self.assertRaisesRegex(ValueError, "All-zero"):
            prepare_sequence(np.zeros((30, 260, 260, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "30 RGB"):
            prepare_sequence(np.ones((29, 260, 260, 3), dtype=np.uint8))

    def test_direct_cli(self):
        script = Path(__file__).resolve().parents[1] / "scripts/predict_cached_bilstm.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)


if __name__ == "__main__":
    unittest.main()
