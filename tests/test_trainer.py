import unittest
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import CheckpointManager
from src.utils.logger import create_logger


class DummyDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, idx):
        return {
            "sequence": torch.randn(2, 3, 224, 224),
            "label": torch.tensor(idx % 2, dtype=torch.long)
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.mean(dim=1).view(B, -1)
        return self.fc(x)


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.loader = DataLoader(DummyDataset(), batch_size=4)
        self.model = DummyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_manager = CheckpointManager(root_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_trainer_train_loop(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_loader=self.loader,
            val_loader=self.loader,
            device="cpu",
            logger=create_logger(),
            checkpoint_manager=self.checkpoint_manager,
            early_stopping=EarlyStopping(patience=2),
        )

        history = trainer.train(epochs=2)
        self.assertEqual(len(history), 2)
        self.assertIn("train_loss", history[0])
        self.assertIn("val_loss", history[0])


if __name__ == "__main__":
    unittest.main()