import unittest
import torch
from src.utils.config import load_config
from src.datasets.dataloader import create_dataloaders
from src.models.model_factory import ModelFactory


class TestFullPipeline(unittest.TestCase):

    def test_pipeline_data_to_model(self):
        config = load_config("configs/smoke_test.yaml")
        train_loader, _, _ = create_dataloaders(config, synthetic_fallback=True)

        self.assertGreater(len(train_loader), 0)

        batch = next(iter(train_loader))
        self.assertIn("sequence", batch)
        self.assertIn("label", batch)

        model = ModelFactory.create(config)
        model.eval()

        with torch.no_grad():
            outputs = model(batch["sequence"])

        self.assertEqual(outputs.shape[0], batch["sequence"].shape[0])
        self.assertEqual(outputs.shape[1], 2)


if __name__ == "__main__":
    unittest.main()