"""
DeepVision AI — CPU Training Smoke Test Utility

Validates end-to-end CPU training mechanics (forward pass, loss, backward pass,
optimizer step, validation, checkpoint saving, and metrics) using the verified
real dataset splits without full epoch training.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.utils.config import load_config
from src.utils.logger import create_logger
from src.datasets.sequence_dataset import SequenceDataset
from src.models.model_factory import ModelFactory
from src.training.optimizer_factory import OptimizerFactory
from src.training.scheduler_factory import SchedulerFactory
from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_memory_usage_mb() -> float:
    """Return process resident memory in MB if psutil is available."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="DeepVision AI — CPU Training Smoke Test")
    parser.add_argument(
        "--model",
        type=str,
        default="vit_temporal_pooling",
        choices=["vit_temporal_pooling", "efficientnet_bilstm"],
        help="Model architecture to test",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke_test.yaml",
        help="Base configuration YAML",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=2,
        help="Maximum training batches per epoch",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=2,
        help="Maximum validation batches per epoch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save smoke test artifacts",
    )
    return parser.parse_args()


def run_cpu_smoke_train(
    model_name: str = "vit_temporal_pooling",
    config_path: str = "configs/smoke_test.yaml",
    max_train_batches: int = 2,
    max_val_batches: int = 2,
    epochs: int = 1,
    seed: int = 42,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Execute lightweight CPU smoke training."""
    start_time = time.perf_counter()
    start_mem = get_memory_usage_mb()

    set_seed(seed)

    # 1. Load config
    config = load_config(config_path)
    config["model"]["name"] = model_name
    # Ensure backbone is frozen for CPU smoke test speed
    config["model"]["freeze_backbone"] = True
    config["model"]["pretrained"] = False

    device = torch.device("cpu")

    # Set up dedicated smoke output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        base_out = Path(output_dir)
    else:
        base_out = Path("outputs/smoke_train_cpu") / f"{model_name}_{timestamp}"
    base_out.mkdir(parents=True, exist_ok=True)

    log_dir = base_out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(log_dir=log_dir, log_name="smoke_train.log")

    logger.info("=" * 60)
    logger.info(f"DeepVision AI — CPU Training Smoke Test: {model_name}")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {base_out}")

    # 2. Prepare real split datasets
    splits_dir = Path("outputs/splits")
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing required split CSVs in {splits_dir}")

    image_size = config["dataset"].get("image_size", 224)
    seq_len = config["dataset"].get("sequence_length", 5)
    batch_size = max(1, config["dataloader"].get("batch_size", 2))

    train_ds_full = SequenceDataset(
        csv_file=train_csv,
        sequence_root="data/intermediate/sequences",
        image_size=image_size,
        synthetic_if_missing=False,
        sequence_length=seq_len,
    )

    val_ds_full = SequenceDataset(
        csv_file=val_csv,
        sequence_root="data/intermediate/sequences",
        image_size=image_size,
        synthetic_if_missing=False,
        sequence_length=seq_len,
    )

    train_subset_size = min(len(train_ds_full), max_train_batches * batch_size)
    val_subset_size = min(len(val_ds_full), max_val_batches * batch_size)

    # Alternate real and fake samples in subset if possible
    train_subset = Subset(train_ds_full, list(range(train_subset_size)))
    val_subset = Subset(val_ds_full, list(range(val_subset_size)))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    logger.info(f"Train subset size: {train_subset_size} ({len(train_loader)} batches)")
    logger.info(f"Val subset size  : {val_subset_size} ({len(val_loader)} batches)")

    # 3. Model instantiation
    model = ModelFactory.create(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total params     : {total_params:,}")
    logger.info(f"Trainable params : {trainable_params:,}")

    # 4. Optimizer & criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = OptimizerFactory.create(model, config)
    scheduler = SchedulerFactory.create(optimizer, config)
    early_stopping = EarlyStopping(patience=5)

    # 5. Checkpoint Manager
    checkpoint_mgr = CheckpointManager(root_dir=base_out / "runs")

    # 6. Trainer instantiation
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        checkpoint_manager=checkpoint_mgr,
        early_stopping=early_stopping,
        scheduler=scheduler,
    )

    # 7. Execute training loop
    logger.info("Executing smoke training loop...")
    history = trainer.train(epochs=epochs)

    # 8. Checkpoint verification
    best_ckpt = checkpoint_mgr.path / "best_model.pth"
    last_ckpt = checkpoint_mgr.path / "last_model.pth"

    checkpoint_verified = False
    if best_ckpt.exists() and last_ckpt.exists():
        # Verify checkpoint loads correctly
        loaded_model = ModelFactory.create(config).to(device)
        loaded_opt = OptimizerFactory.create(loaded_model, config)
        res_epoch, res_best_loss = checkpoint_mgr.load_checkpoint(
            checkpoint_path=best_ckpt,
            model=loaded_model,
            optimizer=loaded_opt,
            map_location=device,
        )
        checkpoint_verified = (res_epoch > 0 and res_best_loss < float("inf"))

    total_time = time.perf_counter() - start_time
    end_mem = get_memory_usage_mb()

    if isinstance(history, list):
        train_losses = [r.get("train_loss") for r in history if isinstance(r, dict)]
        val_losses = [r.get("val_loss") for r in history if isinstance(r, dict)]
        train_accs = [r.get("train_acc") for r in history if isinstance(r, dict)]
        val_accs = [r.get("val_acc") for r in history if isinstance(r, dict)]
    elif isinstance(history, dict):
        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])
        train_accs = history.get("train_acc", [])
        val_accs = history.get("val_acc", [])
    else:
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

    results = {
        "model": model_name,
        "epochs_completed": len(train_losses),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "history_records": history,
        "checkpoint_created": best_ckpt.exists() and last_ckpt.exists(),
        "checkpoint_verified": checkpoint_verified,
        "elapsed_seconds": round(total_time, 2),
        "peak_ram_mb": round(max(start_mem, end_mem), 2),
        "status": "PASS" if checkpoint_verified else "FAIL",
    }

    # Save results json
    results_path = base_out / "smoke_train_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info("=" * 60)
    logger.info(f"CPU Smoke Test Completed: {results['status']} in {total_time:.2f}s")
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)

    return results


def main():
    args = parse_args()
    results = run_cpu_smoke_train(
        model_name=args.model,
        config_path=args.config,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(results, indent=2))
    if results["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

