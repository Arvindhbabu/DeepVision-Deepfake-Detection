"""Unit tests for feature-level parity metrics without backbone downloads."""
import unittest

import numpy as np

from scripts.compare_efficientnet_feature_parity import feature_distances


class TestFeatureParity(unittest.TestCase):
    def test_identical_features(self):
        rng = np.random.default_rng(42)
        features = rng.random((30, 1408), dtype=np.float32)
        result = feature_distances(features, features.copy())
        self.assertAlmostEqual(result["mean_cosine_similarity"], 1.0, places=12)
        self.assertAlmostEqual(result["mean_relative_l2"], 0.0, places=12)
        self.assertAlmostEqual(result["mean_absolute_feature_difference"], 0.0, places=12)

    def test_orthogonal_vectors(self):
        x = np.zeros((30, 1408), dtype=np.float32)
        y = np.zeros_like(x)
        x[:, 0] = 1
        y[:, 1] = 1
        result = feature_distances(x, y)
        self.assertAlmostEqual(result["mean_cosine_similarity"], 0.0, places=12)
        self.assertAlmostEqual(result["mean_relative_l2"], np.sqrt(2), places=12)

    def test_reject_zero_vectors(self):
        x = np.ones((30, 1408), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Zero-norm"):
            feature_distances(x, np.zeros_like(x))

    def test_reject_wrong_dimensions(self):
        x = np.ones((30, 1408), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            feature_distances(x, x[:29])

    def test_reject_nan(self):
        x = np.ones((30, 1408), dtype=np.float32)
        x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            feature_distances(x, np.ones_like(x))


if __name__ == "__main__":
    unittest.main()
