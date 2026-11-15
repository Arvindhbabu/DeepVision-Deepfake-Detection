"""
DeepVision AI — Phase 2 Regression Tests.

Verifies:
1. Configuration loading.
2. Environment variable overrides.
3. Configuration precedence.
4. Path resolution.
5. Dataset root overrides.
6. Output directory overrides.
7. Evaluation package imports.
8. CPU model construction.
9. CPU forward pass.
10. CPU checkpoint loading.
"""

import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.config import load_config, get_path, resolve_path
from src.models.model_factory import ModelFactory
from src.training.checkpoint import CheckpointManager


class TestPhase2Regression(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Clear potential environment pollution
        self.old_env = {
            "DEEPVISION_CONFIG": os.environ.get("DEEPVISION_CONFIG"),
            "DEEPVISION_DATA_ROOT": os.environ.get("DEEPVISION_DATA_ROOT"),
            "DEEPVISION_OUTPUT_DIR": os.environ.get("DEEPVISION_OUTPUT_DIR"),
        }
        for k in self.old_env:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.old_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        self.temp_dir.cleanup()

    # 1. Configuration loading
    def test_01_configuration_loading(self):
        config = load_config("configs/default.yaml")
        self.assertIsInstance(config, dict)
        self.assertIn("project", config)
        self.assertIn("paths", config)
        self.assertIn("dataset", config)
        self.assertIn("dataloader", config)
        self.assertIn("training", config)
        self.assertIn("model", config)
        self.assertEqual(config["project"]["name"], "DeepVision AI")

    # 2. Environment variable overrides
    def test_02_environment_variable_overrides(self):
        custom_data_root = self.temp_path / "custom_data"
        os.environ["DEEPVISION_DATA_ROOT"] = str(custom_data_root)

        config = load_config("configs/default.yaml")
        self.assertEqual(config["paths"]["data_root"], str(custom_data_root))
        self.assertEqual(config["paths"]["raw_data"], str(custom_data_root / "raw"))
        self.assertEqual(config["paths"]["sequences"], str(custom_data_root / "intermediate" / "sequences"))

    # 3. Configuration precedence
    def test_03_configuration_precedence(self):
        # Env var sets data root A
        env_root = self.temp_path / "env_root"
        os.environ["DEEPVISION_DATA_ROOT"] = str(env_root)

        # Explicit argument sets data root B (should override Env Var)
        arg_root = self.temp_path / "arg_root"
        config = load_config("configs/default.yaml", data_root=arg_root)

        self.assertEqual(config["paths"]["data_root"], str(arg_root))
        self.assertEqual(config["paths"]["raw_data"], str(arg_root / "raw"))

    # 4. Path resolution
    def test_04_path_resolution(self):
        rel = resolve_path("data/raw", base_dir=self.temp_path)
        self.assertTrue(rel.is_absolute())
        self.assertEqual(rel, self.temp_path / "data" / "raw")

        direct = resolve_path(self.temp_path / "test")
        self.assertEqual(direct, self.temp_path / "test")

        config = load_config("configs/default.yaml")
        p = get_path(config, "raw_data")
        self.assertIsInstance(p, Path)

    # 5. Dataset root overrides
    def test_05_dataset_root_overrides(self):
        custom_dataset_root = self.temp_path / "kaggle_dataset"
        config = load_config("configs/smoke_test.yaml", data_root=custom_dataset_root)

        self.assertEqual(config["paths"]["data_root"], str(custom_dataset_root))
        self.assertEqual(config["paths"]["processed_data"], str(custom_dataset_root / "processed"))
        self.assertEqual(
            config["paths"]["sequences"],
            str(custom_dataset_root / "intermediate" / "sequences"),
        )

    # 6. Output directory overrides
    def test_06_output_directory_overrides(self):
        custom_out = self.temp_path / "custom_outputs"
        os.environ["DEEPVISION_OUTPUT_DIR"] = str(custom_out)

        config = load_config("configs/default.yaml")
        self.assertEqual(config["paths"]["outputs"], str(custom_out))
        self.assertEqual(config["paths"]["checkpoints"], str(custom_out / "checkpoints"))
        self.assertEqual(config["paths"]["logs"], str(custom_out / "logs"))
        self.assertEqual(config["paths"]["metrics"], str(custom_out / "metrics"))
        self.assertEqual(config["paths"]["runs"], str(custom_out / "runs"))

    # 7. Evaluation package imports
    def test_07_evaluation_package_imports(self):
        import src.evaluation as ev
        self.assertTrue(hasattr(ev, "evaluate"))
        self.assertTrue(hasattr(ev, "load_model"))
        self.assertTrue(hasattr(ev, "compute_metrics"))
        self.assertTrue(hasattr(ev, "run_evaluation_inference"))
        self.assertTrue(hasattr(ev, "plot_confusion_matrix"))
        self.assertTrue(hasattr(ev, "plot_roc"))
        self.assertTrue(hasattr(ev, "save_predictions"))

        from src.evaluation.evaluator import compute_metrics
        dummy_results = {
            "labels": [0, 1],
            "predictions": [0, 1],
            "probabilities": [0.1, 0.9],
        }
        metrics, cm = compute_metrics(dummy_results, self.temp_path)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(cm.shape, (2, 2))

    # 8. CPU model construction
    def test_08_cpu_model_construction(self):
        config_vit = {
            "model": {
                "name": "vit_temporal_pooling",
                "pretrained": False,
                "freeze_backbone": True,
                "num_classes": 2,
                "pooling": "mean",
                "hidden_dim": 64,
                "dropout": 0.1,
            },
            "dataset": {"image_size": 224},
        }
        vit_model = ModelFactory.create(config_vit)
        self.assertIsNotNone(vit_model)
        vit_model = vit_model.to("cpu")
        self.assertEqual(next(vit_model.parameters()).device.type, "cpu")

        config_eff = {
            "model": {
                "name": "efficientnet_bilstm",
                "pretrained": False,
                "freeze_backbone": True,
                "num_classes": 2,
                "hidden_dim": 64,
                "dropout": 0.1,
            },
            "dataset": {"image_size": 224},
        }
        eff_model = ModelFactory.create(config_eff)
        self.assertIsNotNone(eff_model)
        eff_model = eff_model.to("cpu")
        self.assertEqual(next(eff_model.parameters()).device.type, "cpu")

    # 9. CPU forward pass
    def test_09_cpu_forward_pass(self):
        config_vit = {
            "model": {
                "name": "vit_temporal_pooling",
                "pretrained": False,
                "freeze_backbone": True,
                "num_classes": 2,
                "pooling": "mean",
                "hidden_dim": 64,
                "dropout": 0.0,
            },
            "dataset": {"image_size": 224},
        }
        model = ModelFactory.create(config_vit).to("cpu")
        model.eval()

        # Input: (Batch=2, Time=3, Channels=3, Height=224, Width=224)
        dummy_input = torch.randn(2, 3, 3, 224, 224, device="cpu")
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(output.device.type, "cpu")

    # 10. CPU checkpoint loading
    def test_10_cpu_checkpoint_loading(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        model1 = SimpleNet()
        ckpt_path = self.temp_path / "test_ckpt.pth"

        ckpt_data = {
            "epoch": 3,
            "best_loss": 0.42,
            "model_state_dict": model1.state_dict(),
        }
        torch.save(ckpt_data, ckpt_path)

        # Load back onto CPU
        model2 = SimpleNet()
        mgr = CheckpointManager(root_dir=self.temp_path)
        epoch, best_loss = mgr.load_checkpoint(
            checkpoint_path=ckpt_path,
            model=model2,
            map_location="cpu",
        )

        self.assertEqual(epoch, 3)
        self.assertAlmostEqual(best_loss, 0.42)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()

