import unittest
from pathlib import Path
from src.utils.config import load_config, get_path


class TestConfig(unittest.TestCase):

    def test_load_default_config(self):
        config = load_config("configs/default.yaml")
        self.assertIn("project", config)
        self.assertIn("dataset", config)
        self.assertIn("model", config)
        self.assertIn("training", config)
        self.assertEqual(config["project"]["name"], "DeepVision AI")

    def test_load_smoke_test_config(self):
        config = load_config("configs/smoke_test.yaml")
        self.assertIn("training", config)
        self.assertEqual(config["training"]["epochs"], 1)

    def test_get_path(self):
        config = load_config("configs/default.yaml")
        raw_data_path = get_path(config, "raw_data")
        self.assertIsInstance(raw_data_path, Path)
        self.assertEqual(str(raw_data_path).replace("\\", "/"), "data/raw")


if __name__ == "__main__":
    unittest.main()