import unittest
import tempfile
import pandas as pd
from pathlib import Path
from src.datasets.split_generator import DatasetSplitGenerator


class TestSplitGenerator(unittest.TestCase):

    def test_split_generation(self):
        temp_dir = tempfile.TemporaryDirectory()
        input_csv = Path(temp_dir.name) / "input.csv"
        output_dir = Path(temp_dir.name) / "splits"

        df = pd.DataFrame([
            {"video_name": f"vid_{i}", "dataset": "FFPP", "label": i % 2}
            for i in range(20)
        ])
        df.to_csv(input_csv, index=False)

        generator = DatasetSplitGenerator(
            input_csv=str(input_csv),
            output_dir=str(output_dir),
            random_state=42,
        )
        generator.generate()

        self.assertTrue((output_dir / "train.csv").exists())
        self.assertTrue((output_dir / "val.csv").exists())
        self.assertTrue((output_dir / "test.csv").exists())

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()