import unittest
import tempfile
import pandas as pd
from pathlib import Path
from src.datasets.sequence_dataset import SequenceDataset


class TestSequenceDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_split.csv"

        df = pd.DataFrame([
            {"dataset": "FFPP", "video_name": "video_001", "label": 0},
            {"dataset": "CelebDF", "video_name": "video_002", "label": 1},
        ])
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_synthetic_fallback(self):
        ds = SequenceDataset(
            csv_file=self.csv_path,
            sequence_root=self.temp_dir.name,
            image_size=224,
            synthetic_if_missing=True,
            sequence_length=5,
        )

        self.assertEqual(len(ds), 2)
        sample = ds[0]

        self.assertIn("sequence", sample)
        self.assertIn("label", sample)
        self.assertEqual(sample["sequence"].shape, (5, 3, 224, 224))
        self.assertEqual(sample["label"].item(), 0)

    def test_missing_file_raises_error(self):
        ds = SequenceDataset(
            csv_file=self.csv_path,
            sequence_root=self.temp_dir.name,
            synthetic_if_missing=False,
        )
        with self.assertRaises(FileNotFoundError):
            _ = ds[0]


if __name__ == "__main__":
    unittest.main()