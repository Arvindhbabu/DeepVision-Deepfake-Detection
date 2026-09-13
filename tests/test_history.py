import unittest
import tempfile
from pathlib import Path
from src.training.history import History


class TestHistory(unittest.TestCase):

    def test_history_logging_and_export(self):
        history = History()
        history.add(epoch=1, train_loss=0.40, val_loss=0.35, train_acc=92.5, val_acc=91.8)
        history.add(epoch=2, train_loss=0.32, val_loss=0.28, train_acc=95.2, val_acc=94.7)

        self.assertEqual(len(history), 2)
        best_entry = history.best()
        self.assertEqual(best_entry["epoch"], 2)
        self.assertAlmostEqual(best_entry["val_loss"], 0.28)

        temp_dir = tempfile.TemporaryDirectory()
        json_path = Path(temp_dir.name) / "history.json"
        csv_path = Path(temp_dir.name) / "history.csv"

        history.save_json(json_path)
        history.save_csv(csv_path)

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()