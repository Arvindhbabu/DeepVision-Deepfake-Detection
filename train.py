"""
=============================================================
DeepVision AI

Main Training Script

Author : Arvindh Babu
=============================================================
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import load_config
from src.utils.logger import create_logger
from src.datasets.dataloader import create_dataloaders
from src.models.model_factory import ModelFactory
from src.training.optimizer_factory import OptimizerFactory
from src.training.scheduler_factory import SchedulerFactory
from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer


# ============================================================
# Seed & Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Argument Parser
# ============================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepVision AI Training Script"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Allow synthetic dummy sequence generation if sequence files are missing (useful for smoke tests)",
    )

    return parser.parse_args()


# ============================================================
# Device Selection
# ============================================================

def get_device():
    """Detect and return available compute device (CUDA / CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=" * 60)
        print("CUDA AVAILABLE")
        print("=" * 60)
        print("GPU          :", torch.cuda.get_device_name(0))
        print("CUDA Version :", torch.version.cuda)
        print("=" * 60)
    else:
        device = torch.device("cpu")
        print("=" * 60)
        print("Running on CPU")
        print("=" * 60)

    return device


# ============================================================
# Main Entry Point
# ============================================================

def main():
    args = parse_arguments()

    set_seed(args.seed)

    config = load_config(args.config)

    logger = create_logger()

    logger.info("=" * 60)
    logger.info("DeepVision AI — Training System")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device : {device}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    logger.info("Creating DataLoaders...")

    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        synthetic_fallback=args.synthetic_fallback,
    )

    logger.info("DataLoaders Ready")
    logger.info(f"Training batches   : {len(train_loader)}")
    logger.info(f"Validation batches : {len(val_loader)}")
    logger.info(f"Test batches       : {len(test_loader)}")

    if len(train_loader) == 0:
        logger.warning(
            "Training loader is EMPTY. Please run data indexer/split generator or use --synthetic-fallback."
        )

    # --------------------------------------------------------
    # Model Creation
    # --------------------------------------------------------
    logger.info("Creating Model via ModelFactory...")

    model = ModelFactory.create(config)
    model = model.to(device)

    logger.info(f"Model Architecture : {model.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total Parameters     : {total_params:,}")
    logger.info(f"Trainable Parameters : {trainable_params:,}")

    # --------------------------------------------------------
    # Criterion, Optimizer, Scheduler
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    logger.info("Loss Criterion : CrossEntropyLoss")

    optimizer = OptimizerFactory.create(model, config)
    logger.info(f"Optimizer      : {optimizer.__class__.__name__}")

    scheduler = SchedulerFactory.create(optimizer, config)
    if scheduler is not None:
        logger.info(f"Scheduler      : {scheduler.__class__.__name__}")
    else:
        logger.info("Scheduler      : None")

    # --------------------------------------------------------
    # Checkpoint Manager
    # --------------------------------------------------------
    config["runtime"] = {
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    checkpoint_manager = CheckpointManager()
    checkpoint_manager.save_config(config)

    logger.info(f"Experiment Folder  : {checkpoint_manager.path}")

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------
    patience = config["training"].get("early_stopping_patience", 5)
    early_stopping = EarlyStopping(patience=patience)

    # --------------------------------------------------------
    # Resume Checkpoint
    # --------------------------------------------------------
    resume_epoch = 0
    resume_best_loss = float("inf")

    if args.resume is not None:
        logger.info(f"Loading checkpoint from: {args.resume}")
        resume_epoch, resume_best_loss = checkpoint_manager.load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        logger.info(f"Resuming from Epoch {resume_epoch} (Best Val Loss: {resume_best_loss:.4f})")

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        scheduler=scheduler,
    )

    if args.resume is not None:
        trainer.state.epoch = resume_epoch
        trainer.state.best_val_loss = resume_best_loss

    # --------------------------------------------------------
    # Run Training Loop
    # --------------------------------------------------------
    epochs = config["training"].get("epochs", 50)

    logger.info("=" * 60)
    logger.info(f"Model         : {model.__class__.__name__}")
    logger.info(f"Device        : {device}")
    logger.info(f"Epochs        : {epochs}")
    logger.info(f"Batch Size    : {config['dataloader'].get('batch_size', 2)}")
    logger.info(f"Learning Rate : {config['training'].get('learning_rate', 1e-4)}")
    logger.info(f"Dataset       : {config['dataset'].get('name', 'FFPP_CelebDF')}")
    logger.info("=" * 60)

    history = []

    try:
        if len(train_loader) > 0:
            history = trainer.train(epochs=epochs)
        else:
            logger.warning("Skipping training execution due to empty DataLoader.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.exception("Training encountered an error.")
        raise e
    finally:
        try:
            checkpoint_manager.copy_log("outputs/logs/train.log")
        except Exception:
            pass

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Process Finished")
    logger.info("=" * 60)
    logger.info(f"Experiment Output : {checkpoint_manager.path}")
    logger.info(f"Epochs Completed  : {len(history)}")

    if len(history) > 0:
        best_epoch = min(history, key=lambda x: x.get("val_loss", float("inf")))
        logger.info(f"Best Val Loss     : {best_epoch.get('val_loss', 0.0):.4f}")
        logger.info(f"Best Val Accuracy : {best_epoch.get('val_acc', 0.0):.2f}%")

    logger.info("Artifacts Location:")
    logger.info(f"  • Best Model   : {checkpoint_manager.path / 'best_model.pth'}")
    logger.info(f"  • Last Model   : {checkpoint_manager.path / 'last_model.pth'}")
    logger.info(f"  • Metrics      : {checkpoint_manager.path / 'metrics.json'}")
    logger.info(f"  • Config       : {checkpoint_manager.path / 'config.yaml'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()