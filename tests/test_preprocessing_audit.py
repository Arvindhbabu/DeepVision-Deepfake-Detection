"""Stage 6 preprocessing parity tests, using local synthetic JPEGs only."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_preprocessing_parity import audit_sequence, compare_aligned_directory
from scripts.predict_cached_bilstm import select_uniform_indices


def build_example(directory: Path):
    aligned = directory / "aligned"
    aligned.mkdir()
    # JPEG round-trip is intentional: the production sequences read JPEG crops.
    images = []
    for i in range(36):
        arr = np.full((12, 12, 3), i + 15, dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(aligned / f"frame_{i:05d}.jpg")
        with Image.open(aligned / f"frame_{i:05d}.jpg") as image:
            images.append(np.array(image.convert("RGB")))
    sequence = np.stack([images[i] for i in select_uniform_indices(len(images))])
    sequence_path = directory / "sample.npy"
    np.save(sequence_path, sequence)
    return sequence_path, aligned


class TestPreprocessingAudit(unittest.TestCase):
    def test_reconstructed_sequence_exact_match(self):
        with tempfile.TemporaryDirectory() as folder:
            source, aligned = build_example(Path(folder))
            report = audit_sequence(source, aligned, configuration_face_size=12)
            self.assertEqual(report["aligned_directory_comparison"]["status"], "exact_match")
            self.assertEqual(report["aligned_directory_comparison"]["mean_absolute_pixel_error"], 0)
            self.assertTrue(report["configured_face_size_matches_stored"])
            self.assertFalse(report["verified_vs_raw_mp4"])

    def test_mismatch_detected_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source, aligned = build_example(Path(folder))
            seq = np.load(source, allow_pickle=False)
            seq[0, 0, 0, 0] = np.uint8((int(seq[0, 0, 0, 0]) + 1) % 256)
            np.save(source, seq)
            result = compare_aligned_directory(seq, aligned)
            self.assertEqual(result["status"], "pixel_mismatch")
            self.assertGreater(result["mean_absolute_pixel_error"], 0)

    def test_missing_directory_and_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            source, _ = build_example(Path(folder))
            result = audit_sequence(source, Path(folder) / "missing")
            self.assertFalse(result["configured_face_size_matches_stored"])
            self.assertEqual(result["aligned_directory_comparison"]["status"], "unavailable")

    def test_reject_placeholder(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "zero.npy"
            np.save(source, np.zeros((30, 12, 12, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "Zero-filled"):
                audit_sequence(source)

    def test_direct_script_help(self):
        path = Path(__file__).resolve().parents[1] / "scripts/audit_preprocessing_parity.py"
        result = subprocess.run([sys.executable, str(path), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--aligned-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
