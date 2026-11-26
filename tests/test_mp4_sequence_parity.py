"""Fast synthetic tests for MP4/sequence pixel comparison (no MP4 needed)."""
import unittest

import numpy as np

from scripts.compare_mp4_sequence_parity import frame_differences


class TestMP4SequenceParity(unittest.TestCase):
    def test_identical_sequences(self):
        frames = np.full((30, 260, 260, 3), 50, dtype=np.uint8)
        result = frame_differences(frames, frames.copy())
        self.assertTrue(result["exact_pixel_match"])
        self.assertTrue(result["native_shapes_match"])
        self.assertEqual(result["mean_absolute_pixel_error"], 0.0)

    def test_pixel_difference_detected(self):
        ref = np.full((30, 16, 16, 3), 50, dtype=np.uint8)
        candidate = ref.copy()
        candidate[0] = 60
        result = frame_differences(ref, candidate)
        self.assertFalse(result["exact_pixel_match"])
        self.assertGreater(result["mean_absolute_pixel_error"], 0.0)
        self.assertEqual(len(result["per_frame_mae"]), 30)

    def test_resize_explicitly_marked(self):
        reference = np.full((30, 20, 20, 3), 50, dtype=np.uint8)
        candidate = np.full((30, 12, 12, 3), 50, dtype=np.uint8)
        result = frame_differences(reference, candidate)
        self.assertFalse(result["native_shapes_match"])
        self.assertFalse(result["exact_pixel_match"])
        self.assertIn("resized", result["comparison_mode"])

    def test_invalid_sequence_rejected(self):
        with self.assertRaisesRegex(ValueError, "30 frames"):
            frame_differences(
                np.ones((30, 8, 8, 3), dtype=np.uint8),
                np.ones((29, 8, 8, 3), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
