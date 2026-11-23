"""Focused CPU-only tests for Stage 2 input safeguards."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.sequence_dataset import SequenceDataset


class TestStabilization(unittest.TestCase):
    def test_relative_manifest_path_resolves_from_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "sequences" / "ffpp"
            seq.mkdir(parents=True)
            np.save(seq / "001.npy", np.full((2, 8, 8, 3), 128, dtype=np.uint8))
            split_dir = root / "splits"
            split_dir.mkdir()
            csv = split_dir / "train.csv"
            pd.DataFrame([{
                "dataset": "FFPP", "video_name": "001",
                "label": 0, "sequence_path": "sequences/ffpp/001.npy"
            }]).to_csv(csv, index=False)
            ds = SequenceDataset(csv, sequence_root=root / "sequences", image_size=8)
            item = ds[0]
            self.assertEqual(tuple(item["sequence"].shape), (2, 3, 8, 8))
            self.assertAlmostEqual(float(item["sequence"][0, 0, 0, 0]), 128 / 255)

    def test_imagenet_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "001.npy", np.full((2, 8, 8, 3), 255, dtype=np.uint8))
            csv = root / "train.csv"
            pd.DataFrame([{
                "dataset": "FFPP", "video_name": "001", "label": 1,
                "sequence_path": str(root / "001.npy")
            }]).to_csv(csv, index=False)
            ds = SequenceDataset(csv, image_size=8, imagenet_normalize=True)
            got = ds[0]["sequence"]
            self.assertAlmostEqual(float(got[0, 0, 0, 0]), (1 - 0.485) / 0.229, places=4)

    def test_invalid_sequence_shape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "bad.npy", np.ones((8, 8, 3), dtype=np.uint8))
            csv = root / "test.csv"
            pd.DataFrame([{
                "dataset": "FFPP", "video_name": "bad", "label": 0,
                "sequence_path": str(root / "bad.npy")
            }]).to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid sequence shape"):
                SequenceDataset(csv, image_size=8)[0]


if __name__ == "__main__":
    unittest.main()
