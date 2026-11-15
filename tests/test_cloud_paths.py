"""
DeepVision AI — Cloud Paths & Dataset Resolution Tests.
Validates Kaggle input discovery, pre-flight safeguards, split CSV resolution,
and zero-placeholder guards without requiring an external cloud environment.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.cloud_paths import (
    is_kaggle_environment,
    discover_kaggle_dataset,
    resolve_split_paths,
    verify_real_sequences_exist,
)


class TestCloudPaths(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Empty Kaggle input detection
    def test_discover_kaggle_dataset_empty_dir(self):
        empty_input = self.base_dir / "kaggle_input"
        empty_input.mkdir(parents=True, exist_ok=True)

        ds_root, seq_root = discover_kaggle_dataset(kaggle_input_dir=empty_input)
        self.assertIsNone(ds_root)
        self.assertIsNone(seq_root)

    # 2. Discover dataset with direct sequences folder
    def test_discover_kaggle_dataset_direct_sequences(self):
        mock_input = self.base_dir / "kaggle_input"
        mock_ds = mock_input / "deepvision-dataset"
        mock_seq = mock_ds / "sequences"
        mock_seq.mkdir(parents=True, exist_ok=True)

        ds_root, seq_root = discover_kaggle_dataset(kaggle_input_dir=mock_input)
        self.assertEqual(ds_root, mock_ds)
        self.assertEqual(seq_root, mock_seq)

    # 3. Discover dataset with nested intermediate sequences folder
    def test_discover_kaggle_dataset_nested_sequences(self):
        mock_input = self.base_dir / "kaggle_input"
        mock_ds = mock_input / "deepvision-dataset"
        mock_seq = mock_ds / "intermediate" / "sequences"
        mock_seq.mkdir(parents=True, exist_ok=True)

        ds_root, seq_root = discover_kaggle_dataset(kaggle_input_dir=mock_input)
        self.assertEqual(ds_root, mock_ds)
        self.assertEqual(seq_root, mock_seq)

    # 4. Resolve split CSV paths
    def test_resolve_split_paths_success(self):
        splits_dir = self.base_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        for name in ["train.csv", "val.csv", "test.csv"]:
            (splits_dir / name).write_text("dataset,video_name,label\n")

        paths = resolve_split_paths(splits_dir=splits_dir)
        self.assertEqual(paths["splits_dir"], splits_dir)
        self.assertTrue(paths["train_csv"].exists())
        self.assertTrue(paths["val_csv"].exists())
        self.assertTrue(paths["test_csv"].exists())

    # 5. Missing split CSV raises FileNotFoundError
    def test_resolve_split_paths_missing_raises_error(self):
        empty_splits = self.base_dir / "empty_splits"
        empty_splits.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(FileNotFoundError):
            resolve_split_paths(splits_dir=empty_splits)

    # 6. Verify real sequences detects and rejects zero-filled placeholders
    def test_verify_real_sequences_rejects_zero_placeholder(self):
        seq_dir = self.base_dir / "sequences" / "ffpp"
        seq_dir.mkdir(parents=True, exist_ok=True)

        zero_arr = np.zeros((30, 32, 32, 3), dtype=np.uint8)
        zero_file = seq_dir / "001.npy"
        np.save(zero_file, zero_arr)

        csv_file = self.base_dir / "mock_train.csv"
        df = pd.DataFrame([{
            "video_name": "001",
            "label": 0,
            "sequence_path": str(zero_file),
        }])
        df.to_csv(csv_file, index=False)

        res = verify_real_sequences_exist(csv_file, sequence_root=self.base_dir / "sequences")
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["zero_placeholder_count"], 1)
        self.assertEqual(res["verified_real_count"], 0)

    # 7. Verify real sequences accepts valid non-zero arrays
    def test_verify_real_sequences_accepts_valid_arrays(self):
        seq_dir = self.base_dir / "sequences" / "ffpp"
        seq_dir.mkdir(parents=True, exist_ok=True)

        valid_arr = np.random.randint(1, 255, (30, 32, 32, 3), dtype=np.uint8)
        valid_file = seq_dir / "002.npy"
        np.save(valid_file, valid_arr)

        csv_file = self.base_dir / "mock_train.csv"
        df = pd.DataFrame([{
            "video_name": "002",
            "label": 0,
            "sequence_path": str(valid_file),
        }])
        df.to_csv(csv_file, index=False)

        res = verify_real_sequences_exist(csv_file, sequence_root=self.base_dir / "sequences")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["zero_placeholder_count"], 0)
        self.assertEqual(res["verified_real_count"], 1)


if __name__ == "__main__":
    unittest.main()

