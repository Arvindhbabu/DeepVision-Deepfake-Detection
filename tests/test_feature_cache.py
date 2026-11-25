"""Small synthetic test of frame-feature batching; no pretrained weights needed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import numpy as np
import torch

from scripts.cache_efficientnet_features import cache_split, encode_sequence


class ConstantBackbone(torch.nn.Module):
    def forward(self, inputs):
        # Shape matches EfficientNet-B2 final feature channels.
        return inputs.mean(dim=1, keepdim=True).expand(-1, 1408, -1, -1)


class TestFeatureCache(unittest.TestCase):
    def test_bounded_batches_preserve_frame_order(self):
        seq = torch.stack([torch.full((3, 8, 8), value) for value in (1., 2., 3.)])
        result = encode_sequence(seq, ConstantBackbone(),
                                 torch.nn.AdaptiveAvgPool2d(1), frame_batch_size=1)
        self.assertEqual(result.shape, (3, 1408))
        np.testing.assert_allclose(result[:, 0], [1., 2., 3.], rtol=1e-6)

    def test_invalid_batch_size_rejected(self):
        with self.assertRaises(ValueError):
            encode_sequence(torch.zeros(2, 3, 8, 8), ConstantBackbone(),
                            torch.nn.AdaptiveAvgPool2d(1), frame_batch_size=0)


    def test_resume_reuses_cached_video_and_processes_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "sequences"
            source.mkdir()
            rows = []
            for idx in range(2):
                path = source / f"video_{idx}.npy"
                np.save(path, np.full((2, 8, 8, 3), 100 + idx, dtype=np.uint8))
                rows.append({
                    "sequence_path": str(path),
                    "dataset": "ffpp",
                    "video_name": f"video_{idx}",
                    "label": idx,
                })
            split = base / "train.csv"
            pd.DataFrame(rows).to_csv(split, index=False)
            dest = base / "cache"

            def fake_encoder(sequence, backbone, pool, frame_batch_size):
                return np.ones((sequence.shape[0], 1408), dtype=np.float32)

            with patch("scripts.cache_efficientnet_features.encode_sequence",
                       side_effect=fake_encoder) as encoder:
                first = cache_split(split, source, dest, torch.nn.Identity(), max_samples=1)
                self.assertEqual(first["sequences"], 1)
                second = cache_split(split, source, dest, torch.nn.Identity(), resume=True)
                self.assertEqual(second["sequences"], 2)
                self.assertEqual(encoder.call_count, 2)
                self.assertEqual(len(pd.read_csv(dest / "index.csv")), 2)

    def test_resume_rejects_modified_split_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            np.save(source / "one.npy", np.ones((2, 8, 8, 3), dtype=np.uint8))
            split = base / "train.csv"
            pd.DataFrame([{
                "sequence_path": str(source / "one.npy"),
                "dataset": "ffpp", "video_name": "one", "label": 0
            }]).to_csv(split, index=False)
            dest = base / "cache"
            with patch("scripts.cache_efficientnet_features.encode_sequence",
                       return_value=np.ones((2, 1408), dtype=np.float32)):
                cache_split(split, source, dest, torch.nn.Identity(), max_samples=1)
            changed = pd.read_csv(split)
            changed.loc[0, "label"] = 1
            changed.to_csv(split, index=False)
            with self.assertRaisesRegex(ValueError, "does not match"):
                cache_split(split, source, dest, torch.nn.Identity(), resume=True)


if __name__ == "__main__":
    unittest.main()
