import unittest
from src.training.base_trainer import BaseTrainer


class DummyTrainer(BaseTrainer):
    def train_one_epoch(self):
        return 0.5, 80.0

    def validate(self):
        return 0.4, 85.0

    def train(self, epochs):
        return []


class TestBaseTrainer(unittest.TestCase):

    def test_base_trainer_subclass(self):
        self.assertTrue(issubclass(DummyTrainer, BaseTrainer))


if __name__ == "__main__":
    unittest.main()