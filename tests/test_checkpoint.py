import unittest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from src.training.checkpoint import CheckpointManager


class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = CheckpointManager(root_dir=self.temp_dir.name)
        self.model = nn.Linear(10, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_checkpoint(self):
        self.manager.save_best_model(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            best_loss=0.25,
        )

        checkpoint_path = self.manager.path / "best_model.pth"
        self.assertTrue(checkpoint_path.exists())

        new_model = nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)

        epoch, best_loss = self.manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
        )

        self.assertEqual(epoch, 5)
        self.assertAlmostEqual(best_loss, 0.25)


if __name__ == "__main__":
    unittest.main()