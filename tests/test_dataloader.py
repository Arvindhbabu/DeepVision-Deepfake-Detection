import unittest
from src.utils.config import load_config
from src.datasets.dataloader import create_dataloaders


class TestDataLoader(unittest.TestCase):

    def test_create_dataloaders(self):
        config = load_config("configs/smoke_test.yaml")
        train_loader, val_loader, test_loader = create_dataloaders(
            config,
            synthetic_fallback=True,
        )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)


if __name__ == "__main__":
    unittest.main()