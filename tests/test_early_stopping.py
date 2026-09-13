import unittest
from src.training.early_stopping import EarlyStopping


class TestEarlyStopping(unittest.TestCase):

    def test_early_stopping_trigger(self):
        early_stop = EarlyStopping(patience=3, min_delta=0.001)

        losses = [0.70, 0.55, 0.40, 0.35, 0.35, 0.35, 0.35]
        stop_triggered = False
        triggered_epoch = None

        for idx, loss in enumerate(losses, start=1):
            if early_stop(loss):
                stop_triggered = True
                triggered_epoch = idx
                break

        self.assertTrue(stop_triggered)
        self.assertEqual(triggered_epoch, 7)


if __name__ == "__main__":
    unittest.main()