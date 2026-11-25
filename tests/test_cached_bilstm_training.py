"""CPU-only integration checks for real-format frozen feature training."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_cached_bilstm import (
    CachedBiLSTM, CachedFeatureDataset, train,
)


def make_cache(root: Path, labels, prefix: str):
    root.mkdir(parents=True)
    rows = []
    for i, label in enumerate(labels):
        filename = f"{i:05d}.npy"
        matrix = np.full((3, 1408), 0.1 + i / 20, dtype=np.float32)
        np.save(root / filename, matrix)
        rows.append({
            "source_path": f"{prefix}/{i}.npy",
            "video_name": f"{prefix}_{i}",
            "dataset": "ffpp", "label": label,
            "feature_path": filename, "frames": 3,
        })
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    (root / "metadata.json").write_text(
        json.dumps({"pretrained": True, "sequences": len(labels)}), encoding="utf-8"
    )


class TestCachedTraining(unittest.TestCase):
    def test_dataset_and_model_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            make_cache(root, [0, 1], "train")
            ds = CachedFeatureDataset(root, expected_count=2)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (3, 1408))
            self.assertEqual(y.dtype, torch.long)
            self.assertEqual(tuple(CachedBiLSTM(hidden_dim=8)(x[None]).shape), (1, 2))

    def test_reject_untrained_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train"
            make_cache(root, [0], "train")
            (root / "metadata.json").write_text(
                json.dumps({"pretrained": False, "sequences": 1})
            )
            with self.assertRaisesRegex(ValueError, "Refusing"):
                CachedFeatureDataset(root)

    def test_train_saves_checkpoint_without_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root / "train", [0, 1, 0, 1], "train")
            make_cache(root / "val", [0, 1], "val")
            report = train(
                CachedFeatureDataset(root / "train"),
                CachedFeatureDataset(root / "val"),
                root / "experiment", epochs=2, patience=2,
                batch_size=2, hidden_dim=8, seed=7,
            )
            self.assertFalse(report["test_used"])
            self.assertTrue((root / "experiment" / "best_bilstm.pt").is_file())
            self.assertEqual(len(pd.read_csv(root / "experiment" / "history.csv")), 2)

    def test_overlap_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root / "train", [0, 1], "same")
            make_cache(root / "val", [0, 1], "same")
            output = root / "experiment"
            with self.assertRaisesRegex(ValueError, "leakage"):
                train(CachedFeatureDataset(root / "train"),
                      CachedFeatureDataset(root / "val"),
                      output, epochs=1, hidden_dim=8)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
