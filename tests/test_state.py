import unittest
from src.training.state import TrainingState


class TestTrainingState(unittest.TestCase):

    def test_training_state_update(self):
        state = TrainingState()
        state.update(epoch=5, global_step=100, best_val_loss=0.25, resumed=True)

        state_dict = state.to_dict()
        self.assertEqual(state_dict["epoch"], 5)
        self.assertEqual(state_dict["global_step"], 100)
        self.assertAlmostEqual(state_dict["best_val_loss"], 0.25)
        self.assertTrue(state_dict["resumed"])


if __name__ == "__main__":
    unittest.main()