"""
DeepVision AI

DataLoader Factory
"""

from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets.sequence_dataset import SequenceDataset
from src.datasets.balanced_sampler import create_balanced_sampler


def create_dataloaders(config: dict, synthetic_fallback: bool = False):
    """
    Create train, validation, and test dataloaders using dataset splits.

    Args:
        config (dict): Configuration dictionary containing dataset & dataloader options.
        synthetic_fallback (bool): If True, allows synthetic sequence fallback when files are missing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    image_size = config["dataset"].get("image_size", 224)
    seq_len = config["dataset"].get("sequence_length", 26)
    batch_size = config["dataloader"].get("batch_size", 2)
    num_workers = config["dataloader"].get("num_workers", 0)
    pin_memory = config["dataloader"].get("pin_memory", True)
    drop_last = config["dataloader"].get("drop_last", False)

    seq_root = config["paths"].get("sequences", "data/intermediate/sequences")
    splits_dir = Path(config["paths"].get("outputs", "outputs")) / "splits"

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    train_dataset = SequenceDataset(
        csv_file=train_csv,
        sequence_root=seq_root,
        image_size=image_size,
        synthetic_if_missing=synthetic_fallback,
        sequence_length=seq_len,
    )

    val_dataset = SequenceDataset(
        csv_file=val_csv,
        sequence_root=seq_root,
        image_size=image_size,
        synthetic_if_missing=synthetic_fallback,
        sequence_length=seq_len,
    )

    test_dataset = SequenceDataset(
        csv_file=test_csv,
        sequence_root=seq_root,
        image_size=image_size,
        synthetic_if_missing=synthetic_fallback,
        sequence_length=seq_len,
    )

    train_sampler = create_balanced_sampler(train_dataset) if len(train_dataset) > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and len(train_dataset) > 0),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if len(train_dataset) >= batch_size else False,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader