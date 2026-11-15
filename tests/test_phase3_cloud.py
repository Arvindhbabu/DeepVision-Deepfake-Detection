"""
DeepVision AI — Phase 3 Cloud Compatibility & Diagnostic Regression Tests.

Verifies:
1. CPU-only diagnostics execution.
2. Missing optional packages handling.
3. Missing CUDA handling (reported as WARNING, not crashing).
4. Kaggle environment detection with mocked environment.
5. Colab environment detection with mocked environment.
6. JSON serialization validity.
7. Data/output environment variable overrides.
8. Read-only input and writable output validation.
9. Smoke test model construction on CPU.
10. Smoke test checkpoint save and load cycle.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_environment import (
    collect_diagnostics,
    detect_cloud_environment,
    check_packages,
    check_cuda_and_gpu,
)
from scripts.smoke_test_cloud import run_smoke_test
from src.utils.config import load_config, validate_paths
from src.models.model_factory import ModelFactory


class TestPhase3Cloud(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Preserve and clear target environment variables
        self.env_keys = [
            "DEEPVISION_CONFIG",
            "DEEPVISION_DATA_ROOT",
            "DEEPVISION_OUTPUT_DIR",
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_URL_BASE",
            "COLAB_GPU",
            "COLAB_RELEASE_TAG",
        ]
        self.original_env = {k: os.environ.get(k) for k in self.env_keys}
        for k in self.env_keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.original_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        self.temp_dir.cleanup()

    # 1. CPU-only diagnostics execution
    def test_01_cpu_diagnostics_execution(self):
        diag = collect_diagnostics("configs/smoke_test.yaml")
        self.assertIsInstance(diag, dict)
        self.assertIn("status", diag)
        self.assertIn("python", diag)
        self.assertIn("platform", diag)
        self.assertIn("hardware", diag)
        self.assertIn("cloud", diag)
        self.assertIn("packages", diag)
        self.assertIn("cuda", diag)
        self.assertIn("configuration", diag)
        self.assertFalse(diag["critical_failure"])

    # 2. Missing optional packages handling
    def test_02_missing_optional_packages_graceful(self):
        packages = check_packages()
        self.assertIsInstance(packages, dict)
        self.assertIn("torch", packages)
        self.assertIn("numpy", packages)
        self.assertIn("timm", packages)
        # timm is optional, should not cause critical crash
        if not packages["timm"]["installed"]:
            self.assertEqual(packages["timm"]["status"], "WARNING")

    # 3. Missing CUDA handling
    def test_03_missing_cuda_warning(self):
        # Mock torch.cuda.is_available to return False
        with patch("torch.cuda.is_available", return_value=False):
            cuda_info = check_cuda_and_gpu()
            self.assertFalse(cuda_info["cuda_available"])
            self.assertEqual(cuda_info["status"], "WARNING")
            self.assertIn("CPU", cuda_info["message"])

    # 4. Kaggle environment detection
    def test_04_kaggle_detection_mocked(self):
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
            env_info = detect_cloud_environment()
            self.assertTrue(env_info["is_kaggle"])
            self.assertEqual(env_info["platform_name"], "Kaggle Notebook")

    # 5. Colab environment detection
    def test_05_colab_detection_mocked(self):
        with patch.dict(os.environ, {"COLAB_GPU": "0"}):
            env_info = detect_cloud_environment()
            self.assertTrue(env_info["is_colab"])
            self.assertEqual(env_info["platform_name"], "Google Colab")

    # 6. JSON serialization validity
    def test_06_json_serialization_validity(self):
        diag = collect_diagnostics("configs/smoke_test.yaml")
        json_str = json.dumps(diag)
        reloaded = json.loads(json_str)
        self.assertEqual(reloaded["status"], diag["status"])
        self.assertEqual(reloaded["python"]["version"], diag["python"]["version"])

    # 7. Data and output environment overrides
    def test_07_cloud_data_output_overrides(self):
        mock_data_root = self.temp_path / "cloud_input"
        mock_output_dir = self.temp_path / "cloud_output"
        mock_data_root.mkdir()
        mock_output_dir.mkdir()

        os.environ["DEEPVISION_DATA_ROOT"] = str(mock_data_root)
        os.environ["DEEPVISION_OUTPUT_DIR"] = str(mock_output_dir)

        config = load_config("configs/smoke_test.yaml")
        self.assertEqual(config["paths"]["data_root"], str(mock_data_root))
        self.assertEqual(config["paths"]["outputs"], str(mock_output_dir))
        self.assertEqual(config["paths"]["checkpoints"], str(mock_output_dir / "checkpoints"))

    # 8. Read-only input and writable output validation
    def test_08_path_validation_read_write(self):
        mock_in = self.temp_path / "readonly_input"
        mock_in.mkdir()
        mock_out = self.temp_path / "writable_output"

        cfg = {
            "paths": {
                "raw_data": str(mock_in),
                "outputs": str(mock_out),
            }
        }
        report = validate_paths(cfg)
        self.assertTrue(report["valid"])
        self.assertTrue(report["input_paths"]["raw_data"]["exists"])
        self.assertTrue(report["input_paths"]["raw_data"]["readable"])
        self.assertTrue(report["output_paths"]["outputs"]["writable"])

    # 9. Smoke test model construction on CPU
    def test_09_smoke_model_construction(self):
        vit_cfg = {
            "model": {
                "name": "vit_temporal_pooling",
                "pretrained": False,
                "freeze_backbone": True,
                "num_classes": 2,
                "hidden_dim": 64,
                "pooling": "mean",
            },
            "dataset": {"image_size": 224},
        }
        vit = ModelFactory.create(vit_cfg).to("cpu")
        self.assertIsNotNone(vit)

        eff_cfg = {
            "model": {
                "name": "efficientnet_bilstm",
                "pretrained": False,
                "freeze_backbone": True,
                "num_classes": 2,
                "hidden_dim": 64,
            },
            "dataset": {"image_size": 224},
        }
        eff = ModelFactory.create(eff_cfg).to("cpu")
        self.assertIsNotNone(eff)

    # 10. Smoke test checkpoint save and load cycle
    def test_10_smoke_checkpoint_cycle(self):
        out_dir = self.temp_path / "smoke_out"
        results = run_smoke_test(
            config_path="configs/smoke_test.yaml",
            model_choice="all",
            custom_output_dir=str(out_dir),
        )
        self.assertTrue(results["all_passed"])
        self.assertEqual(results["stages"]["config_loading"]["status"], "PASS")
        self.assertEqual(results["stages"]["model_instantiation"]["status"], "PASS")
        self.assertEqual(results["stages"]["forward_pass"]["status"], "PASS")
        self.assertEqual(results["stages"]["checkpoint_io"]["status"], "PASS")
        self.assertTrue((out_dir / "smoke_test_results.json").exists())


if __name__ == "__main__":
    unittest.main()

