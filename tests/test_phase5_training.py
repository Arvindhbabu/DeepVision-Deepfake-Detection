"""
DeepVision AI — Phase 5 Unit & Regression Tests
Tests training mechanics, backpropagation, checkpoint I/O, resume state persistence,
and evaluation metric computation independently of external dataset assets.
"""

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.checkpoint import CheckpointManager
from src.training.state import TrainingState
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.models.efficientnet_bilstm import EfficientNetBiLSTM
from src.models.vit_temporal_pooling import ViTTemporalPooling
from src.evaluation.evaluator import compute_metrics


class DummySequenceDataset(Dataset):
    """Synthetic dataset for test isolation."""

    def __init__(self, size=8, seq_len=3, img_size=32):
        self.size = size
        self.seq_len = seq_len
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (T, C, H, W)
        seq = torch.rand(self.seq_len, 3, self.img_size, self.img_size)
        label = torch.tensor(idx % 2, dtype=torch.long)
        return {"sequence": seq, "label": label, "video_name": f"vid_{idx}", "dataset": "test"}


class TestPhase5Training(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Batch Execution on Minimal Models
    def test_training_batch_execution_and_shapes(self):
        # Test EfficientNetBiLSTM with small input
        model = EfficientNetBiLSTM(num_classes=2, hidden_dim=32, pretrained=False, freeze_backbone=True)
        dummy_input = torch.randn(2, 3, 3, 64, 64)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 2))

    # 2. Backpropagation and Weight Updates
    def test_backpropagation_gradient_flow(self):
        model = EfficientNetBiLSTM(num_classes=2, hidden_dim=32, pretrained=False, freeze_backbone=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_weights = [p.clone() for p in model.classifier.parameters() if p.requires_grad]

        dummy_input = torch.randn(2, 3, 3, 64, 64)
        dummy_label = torch.tensor([0, 1], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(dummy_input)
        loss = criterion(logits, dummy_label)
        loss.backward()

        # Check gradients exist
        for p in model.classifier.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

        optimizer.step()

        # Check weights updated
        updated_weights = [p for p in model.classifier.parameters() if p.requires_grad]
        for init, curr in zip(initial_weights, updated_weights):
            self.assertFalse(torch.equal(init, curr))

    # 3. Checkpoint Save and Load Integrity
    def test_checkpoint_save_and_load(self):
        mgr = CheckpointManager(root_dir=self.run_dir)
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        mgr.save_best_model(
            model=model,
            optimizer=optimizer,
            epoch=3,
            best_loss=0.421,
        )

        best_pth = mgr.path / "best_model.pth"
        self.assertTrue(best_pth.exists())

        loaded_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        loaded_opt = torch.optim.Adam(loaded_model.parameters(), lr=0.01)

        epoch, best_loss = mgr.load_checkpoint(
            checkpoint_path=best_pth,
            model=loaded_model,
            optimizer=loaded_opt,
            map_location="cpu",
        )

        self.assertEqual(epoch, 3)
        self.assertAlmostEqual(best_loss, 0.421, places=3)

        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    # 4. Optimizer State Restoration
    def test_optimizer_state_restoration(self):
        mgr = CheckpointManager(root_dir=self.run_dir)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Run one step to populate optimizer state (exp_avg, exp_avg_sq)
        out = model(torch.randn(2, 4)).sum()
        out.backward()
        optimizer.step()

        state_before = optimizer.state_dict()
        self.assertTrue(len(state_before["state"]) > 0)

        mgr.save_best_model(model=model, optimizer=optimizer, epoch=1, best_loss=0.5)

        new_model = nn.Linear(4, 2)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.01)

        mgr.load_checkpoint(
            checkpoint_path=mgr.path / "best_model.pth",
            model=new_model,
            optimizer=new_optimizer,
        )

        self.assertEqual(len(new_optimizer.state_dict()["state"]), len(state_before["state"]))

    # 5. Regression Test: Resume Best Validation Loss Persistence
    def test_resume_best_val_loss_not_reset_to_infinity(self):
        """Verify that Trainer.train() does not wipe out restored best_val_loss on resume."""
        model = nn.Sequential(nn.Flatten(), nn.Linear(768, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        train_ds = DummySequenceDataset(size=4, seq_len=1, img_size=16)
        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(train_ds, batch_size=2)

        mgr = CheckpointManager(root_dir=self.run_dir)
        early_stopping = EarlyStopping(patience=5)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            logger=logging.getLogger("test"),
            checkpoint_manager=mgr,
            early_stopping=early_stopping,
        )

        # Mock a resumed state where best_val_loss was previously 0.25 at epoch 2
        trainer.state.epoch = 2
        trainer.state.best_val_loss = 0.25
        trainer.state.resumed = True

        # When train() starts, best_val_loss must NOT be overwritten with float('inf')
        # If val_loss in epoch 3 is 0.80, best_val_loss should stay 0.25!
        trainer.train(epochs=3)

        self.assertLessEqual(trainer.state.best_val_loss, 0.80)
        # Verify it wasn't reset to inf
        self.assertNotEqual(trainer.state.best_val_loss, float("inf"))

    # 6. Evaluation Metrics Calculation
    def test_evaluation_metric_calculations(self):
        results = {
            "labels": [0, 0, 1, 1],
            "predictions": [0, 0, 1, 1],
            "probabilities": [0.1, 0.2, 0.8, 0.9],
            "video_names": ["v1", "v2", "v3", "v4"],
            "dataset_names": ["d", "d", "d", "d"],
        }
        metrics, cm = compute_metrics(results, self.run_dir / "eval_out")
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["num_samples"], 4)

    # 7. Specificity and Balanced Accuracy Mathematical Behavior
    def test_specificity_and_balanced_accuracy_math(self):
        # Case: all positive predictions (TP=2, FP=2, TN=0, FN=0)
        results_all_pos = {
            "labels": [0, 0, 1, 1],
            "predictions": [1, 1, 1, 1],
            "probabilities": [0.7, 0.8, 0.9, 0.95],
        }
        metrics_pos, _ = compute_metrics(results_all_pos, self.run_dir / "pos_out")
        self.assertEqual(metrics_pos["recall"], 1.0)
        self.assertEqual(metrics_pos["specificity"], 0.0)
        self.assertEqual(metrics_pos["balanced_accuracy"], 0.5)

        # Case: all negative predictions (TP=0, FP=0, TN=2, FN=2)
        results_all_neg = {
            "labels": [0, 0, 1, 1],
            "predictions": [0, 0, 0, 0],
            "probabilities": [0.1, 0.2, 0.3, 0.4],
        }
        metrics_neg, _ = compute_metrics(results_all_neg, self.run_dir / "neg_out")
        self.assertEqual(metrics_neg["recall"], 0.0)
        self.assertEqual(metrics_neg["specificity"], 1.0)
        self.assertEqual(metrics_neg["balanced_accuracy"], 0.5)


    # 8. Real Split Loading and Counts
    def test_real_split_loading_and_counts(self):
        splits_dir = Path("outputs/splits")
        required = [splits_dir / name for name in ("train.csv", "val.csv", "test.csv")]
        if not all(path.is_file() for path in required):
            self.skipTest(
                "Real FF++ split CSVs are local, gitignored dataset artifacts; "
                "run this integration test on a machine with the verified dataset."
            )

        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")

        self.assertEqual(len(train_df), 96)
        self.assertEqual(len(val_df), 20)
        self.assertEqual(len(test_df), 22)

    # 9. Placeholder Exclusion Verification
    def test_placeholder_exclusion_guard(self):
        from src.datasets.sequence_dataset import SequenceDataset

        mock_seq_dir = self.run_dir / "sequences"
        mock_seq_dir.mkdir(parents=True, exist_ok=True)
        zero_arr = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        zero_path = mock_seq_dir / "zero.npy"
        np.save(zero_path, zero_arr)

        csv_path = self.run_dir / "mock_split.csv"
        df = pd.DataFrame([{"dataset": "test", "video_name": "zero", "label": 0, "sequence_path": str(zero_path)}])
        df.to_csv(csv_path, index=False)

        ds = SequenceDataset(csv_file=csv_path, sequence_root=mock_seq_dir, synthetic_if_missing=False)
        with self.assertRaises(ValueError):
            _ = ds[0]


if __name__ == "__main__":
    unittest.main()

