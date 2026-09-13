import unittest
import tempfile
from pathlib import Path
from src.datasets.tf_dataset_builder import DatasetIndexer


class TestDatasetBuilder(unittest.TestCase):

    def test_dataset_indexer_initialization(self):
        temp_dir = tempfile.TemporaryDirectory()
        indexer = DatasetIndexer(root_dir=temp_dir.name)
        samples = indexer.scan()

        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 0)

        output_csv = Path(temp_dir.name) / "index.csv"
        indexer.save_index(output_csv)
        self.assertTrue(output_csv.exists())

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()