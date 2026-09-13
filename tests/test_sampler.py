import unittest
import pandas as pd
import tempfile
from pathlib import Path
from src.datasets.sequence_dataset import SequenceDataset
from src.datasets.balanced_sampler import create_balanced_sampler


class TestBalancedSampler(unittest.TestCase):

    def test_sampler_creation(self):
        temp_dir = tempfile.TemporaryDirectory()
        csv_path = Path(temp_dir.name) / "test.csv"

        df = pd.DataFrame([
            {"dataset": "FFPP", "video_name": "v1", "label": 0},
            {"dataset": "FFPP", "video_name": "v2", "label": 0},
            {"dataset": "CelebDF", "video_name": "v3", "label": 1},
        ])
        df.to_csv(csv_path, index=False)

        dataset = SequenceDataset(csv_file=csv_path, synthetic_if_missing=True)
        sampler = create_balanced_sampler(dataset)

        self.assertIsNotNone(sampler)
        self.assertEqual(len(sampler), 3)

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()