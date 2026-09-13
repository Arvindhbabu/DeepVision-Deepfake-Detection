import unittest
import torch.nn as nn
from src.utils.config import load_config
from src.training.optimizer_factory import OptimizerFactory
from src.training.scheduler_factory import SchedulerFactory


class TestOptimizerScheduler(unittest.TestCase):

    def test_optimizer_and_scheduler_creation(self):
        config = load_config("configs/default.yaml")
        model = nn.Linear(10, 2)

        optimizer = OptimizerFactory.create(model, config)
        self.assertEqual(optimizer.__class__.__name__, "AdamW")

        scheduler = SchedulerFactory.create(optimizer, config)
        self.assertIsNotNone(scheduler)
        self.assertEqual(scheduler.__class__.__name__, "CosineAnnealingLR")


if __name__ == "__main__":
    unittest.main()