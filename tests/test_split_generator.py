import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.datasets.split_generator import DatasetSplitGenerator


class TestSplitGenerator(unittest.TestCase):
    def test_unsafe_split_rejected_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "index.csv"
            target = Path(directory) / "splits"
            pd.DataFrame([
                {"video_name": f"vid_{i}", "dataset": "FFPP", "label": i % 2}
                for i in range(20)
            ]).to_csv(source, index=False)
            generator = DatasetSplitGenerator(source, target, random_state=42)
            with self.assertRaisesRegex(RuntimeError, "actor-disjoint"):
                generator.generate()
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
