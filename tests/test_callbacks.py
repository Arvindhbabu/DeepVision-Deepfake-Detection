import unittest
from src.training.callbacks import Callback, CallbackManager


class DummyCallback(Callback):
    def __init__(self):
        self.events = []

    def on_train_begin(self, trainer):
        self.events.append("train_begin")

    def on_epoch_begin(self, trainer):
        self.events.append("epoch_begin")

    def on_epoch_end(self, trainer):
        self.events.append("epoch_end")

    def on_train_end(self, trainer):
        self.events.append("train_end")


class TestCallbacks(unittest.TestCase):

    def test_callback_manager_events(self):
        manager = CallbackManager()
        cb = DummyCallback()
        manager.add(cb)

        fake_trainer = None
        manager.on_train_begin(fake_trainer)
        manager.on_epoch_begin(fake_trainer)
        manager.on_epoch_end(fake_trainer)
        manager.on_train_end(fake_trainer)

        self.assertEqual(
            cb.events,
            ["train_begin", "epoch_begin", "epoch_end", "train_end"]
        )


if __name__ == "__main__":
    unittest.main()