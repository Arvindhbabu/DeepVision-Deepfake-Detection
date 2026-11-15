#!/usr/bin/env python3
"""
DeepVision AI — Cloud & Local Lightweight Smoke Test.

Verifies end-to-end environment readiness WITHOUT loading full 40GB datasets:
1. Loads active configuration.
2. Validates dataset and output paths.
3. Tests ModelFactory construction for ViT and EfficientNet-B2 + BiLSTM.
4. Executes forward passes on CPU using synthetic sequence tensors.
5. Verifies atomic checkpoint saving and loading.
6. Records smoke test results to the configured output directory.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.utils.config import load_config, validate_paths
from src.models.model_factory import ModelFactory
from src.training.checkpoint import CheckpointManager


def run_smoke_test(
    config_path: str = "configs/smoke_test.yaml",
    model_choice: str = "all",
    custom_output_dir: str = None,
) -> Dict[str, Any]:
    """
    Execute lightweight cloud smoke tests.

    Args:
        config_path: Path to configuration YAML file.
        model_choice: 'vit', 'efficientnet', or 'all'.
        custom_output_dir: Optional override for output directory.

    Returns:
        dict: Detailed results of all test stages.
    """
    start_time = time.time()
    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": config_path,
        "stages": {},
        "all_passed": True,
        "failures": [],
    }

    print("=" * 65)
    print(" DeepVision AI — Cloud & Environment Smoke Test")
    print("=" * 65)

    # Stage 1: Load Configuration
    print("\n[Stage 1/5] Loading configuration...")
    try:
        config = load_config(config_path, output_dir=custom_output_dir)
        output_dir = Path(config["paths"].get("outputs", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        results["stages"]["config_loading"] = {
            "status": "PASS",
            "project_name": config.get("project", {}).get("name", "Unknown"),
            "data_root": config.get("paths", {}).get("data_root", "data"),
            "output_dir": str(output_dir),
        }
        print(f"  • Config loaded successfully: {config_path} [PASS]")
        print(f"  • Output directory: {output_dir}")
    except Exception as e:
        results["stages"]["config_loading"] = {"status": "FAIL", "error": str(e)}
        results["failures"].append(f"Stage 1 Config Loading: {e}")
        results["all_passed"] = False
        print(f"  • Config loading FAILED: {e}")
        return results

    # Stage 2: Path Validation
    print("\n[Stage 2/5] Validating dataset and output paths...")
    path_report = validate_paths(config)
    results["stages"]["path_validation"] = {
        "status": "PASS" if path_report["valid"] else "WARNING",
        "details": path_report,
    }
    print(f"  • Path writability check: {'PASS' if path_report['valid'] else 'WARNING'}")
    if path_report["issues"]:
        for issue in path_report["issues"]:
            print(f"    - Note: {issue}")

    # Stage 3: Model Instantiation on CPU
    print("\n[Stage 3/5] Instantiating models on CPU...")
    models_to_test = []
    if model_choice in ("vit", "all"):
        models_to_test.append(("vit_temporal_pooling", 224, 64))
    if model_choice in ("efficientnet", "all"):
        models_to_test.append(("efficientnet_bilstm", 224, 64))

    stage3_passed = True
    stage3_details = {}

    for name, img_size, hidden in models_to_test:
        try:
            cfg = {
                "model": {
                    "name": name,
                    "pretrained": False,
                    "freeze_backbone": True,
                    "num_classes": 2,
                    "hidden_dim": hidden,
                    "dropout": 0.1,
                    "pooling": "mean",
                },
                "dataset": {"image_size": img_size},
            }
            model = ModelFactory.create(cfg).to("cpu")
            param_count = sum(p.numel() for p in model.parameters())
            stage3_details[name] = {
                "status": "PASS",
                "parameters": param_count,
            }
            print(f"  • Model '{name}': Instantiated on CPU ({param_count:,} parameters) [PASS]")
        except Exception as e:
            stage3_details[name] = {"status": "FAIL", "error": str(e)}
            results["failures"].append(f"Model construction for '{name}': {e}")
            stage3_passed = False
            print(f"  • Model '{name}': FAILED: {e}")

    results["stages"]["model_instantiation"] = {
        "status": "PASS" if stage3_passed else "FAIL",
        "models": stage3_details,
    }
    if not stage3_passed:
        results["all_passed"] = False

    # Stage 4: Synthetic Forward Pass
    print("\n[Stage 4/5] Running synthetic forward passes on CPU...")
    stage4_passed = True
    stage4_details = {}

    # Dummy batch: Batch=1, Sequence Length=4, Channels=3, Height=224, Width=224
    dummy_input = torch.randn(1, 4, 3, 224, 224, device="cpu")

    for name, img_size, hidden in models_to_test:
        try:
            cfg = {
                "model": {
                    "name": name,
                    "pretrained": False,
                    "freeze_backbone": True,
                    "num_classes": 2,
                    "hidden_dim": hidden,
                    "dropout": 0.0,
                    "pooling": "mean",
                },
                "dataset": {"image_size": img_size},
            }
            model = ModelFactory.create(cfg).to("cpu")
            model.eval()

            with torch.no_grad():
                out = model(dummy_input)

            expected_shape = (1, 2)
            if tuple(out.shape) == expected_shape:
                stage4_details[name] = {
                    "status": "PASS",
                    "output_shape": list(out.shape),
                }
                print(f"  • Forward pass '{name}': Output shape {list(out.shape)} [PASS]")
            else:
                raise ValueError(f"Expected output shape {expected_shape}, got {list(out.shape)}")
        except Exception as e:
            stage4_details[name] = {"status": "FAIL", "error": str(e)}
            results["failures"].append(f"Forward pass for '{name}': {e}")
            stage4_passed = False
            print(f"  • Forward pass '{name}': FAILED: {e}")

    results["stages"]["forward_pass"] = {
        "status": "PASS" if stage4_passed else "FAIL",
        "models": stage4_details,
    }
    if not stage4_passed:
        results["all_passed"] = False

    # Stage 5: Checkpoint Save and Load on CPU
    print("\n[Stage 5/5] Testing checkpoint save & load on CPU...")
    try:
        ckpt_mgr = CheckpointManager(root_dir=output_dir / "smoke_test_runs")
        test_model = torch.nn.Linear(10, 2)
        test_opt = torch.optim.SGD(test_model.parameters(), lr=0.01)

        # Save test checkpoint
        ckpt_mgr.save_last_model(
            model=test_model,
            optimizer=test_opt,
            epoch=1,
            best_loss=0.5,
        )

        # Load back
        load_model = torch.nn.Linear(10, 2)
        epoch, best_loss = ckpt_mgr.load_checkpoint(
            checkpoint_path=ckpt_mgr.path / "last_model.pth",
            model=load_model,
            map_location="cpu",
        )

        match = all(
            torch.equal(p1, p2)
            for p1, p2 in zip(test_model.parameters(), load_model.parameters())
        )

        if match and epoch == 1:
            results["stages"]["checkpoint_io"] = {
                "status": "PASS",
                "checkpoint_dir": str(ckpt_mgr.path),
            }
            print(f"  • Checkpoint save & load verified successfully [PASS]")
        else:
            raise RuntimeError("Restored checkpoint weights did not match original weights.")

        # Clean up temporary smoke run
        try:
            import shutil
            shutil.rmtree(str(ckpt_mgr.path), ignore_errors=True)
        except Exception:
            pass

    except Exception as e:
        results["stages"]["checkpoint_io"] = {"status": "FAIL", "error": str(e)}
        results["failures"].append(f"Checkpoint I/O: {e}")
        results["all_passed"] = False
        print(f"  • Checkpoint I/O FAILED: {e}")

    # Summary
    results["duration_seconds"] = round(time.time() - start_time, 2)
    results_file = output_dir / "smoke_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    if results["all_passed"]:
        print(f" Smoke Test COMPLETED SUCCESSFULLY in {results['duration_seconds']}s [PASS]")
    else:
        print(f" Smoke Test COMPLETED WITH FAILURES ({len(results['failures'])} errors) [FAIL]")
    print(f" Summary saved to: {results_file}")
    print("=" * 65)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DeepVision AI Cloud & Local Smoke Test"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke_test.yaml",
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["vit", "efficientnet", "all"],
        help="Model architecture to smoke test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory override",
    )

    args = parser.parse_args()

    results = run_smoke_test(
        config_path=args.config,
        model_choice=args.model,
        custom_output_dir=args.output_dir,
    )

    if not results["all_passed"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

