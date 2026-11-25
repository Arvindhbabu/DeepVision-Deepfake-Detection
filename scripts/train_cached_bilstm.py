"""Train a lightweight BiLSTM on frozen, pretrained EfficientNet-B2 video features.

Train and validation only. The final test split is deliberately not opened.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_DIM = 1408


class CachedFeatureDataset(Dataset):
    """Read feature caches produced by scripts/cache_efficientnet_features.py."""

    def __init__(self, root: Path, expected_count: int | None = None):
        self.root = Path(root)
        metadata_file = self.root / "metadata.json"
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Missing pretrained cache metadata: {metadata_file}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("pretrained") is not True:
            raise ValueError(f"Refusing random-weight or unverified cache: {self.root}")
        self.rows = pd.read_csv(self.root / "index.csv")
        required = {"feature_path", "label", "source_path", "frames", "video_name"}
        if not required.issubset(self.rows.columns):
            raise ValueError(f"Cache index missing fields: {required - set(self.rows.columns)}")
        if self.rows.empty:
            raise ValueError(f"Empty feature cache: {self.root}")
        if expected_count is not None and len(self.rows) != expected_count:
            raise ValueError(
                f"{self.root}: expected {expected_count} cached videos, got {len(self.rows)}"
            )
        if int(metadata.get("sequences", -1)) != len(self.rows):
            raise ValueError(f"Cache count does not match metadata: {self.root}")
        if not self.rows["label"].isin([0, 1]).all():
            raise ValueError("Labels must be 0 (Real) or 1 (Fake)")
        if self.rows["source_path"].isna().any():
            raise ValueError("Cannot verify cache source identities")
        self.source_paths = set(self.rows["source_path"].astype(str))
        if len(self.source_paths) != len(self.rows):
            raise ValueError("Duplicate source paths within cache")
        for _, row in self.rows.iterrows():
            filename = str(row["feature_path"])
            if Path(filename).name != filename or not filename.endswith(".npy"):
                raise ValueError(f"Invalid feature filename: {filename}")
            if not (self.root / filename).is_file():
                raise FileNotFoundError(self.root / filename)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        arr = np.load(self.root / str(row["feature_path"]), allow_pickle=False)
        if (arr.ndim != 2 or arr.shape != (int(row["frames"]), FEATURE_DIM)
                or arr.dtype != np.float32 or not np.isfinite(arr).all()):
            raise ValueError(f"Invalid cached features for {row['video_name']}: {arr.shape}")
        return torch.from_numpy(arr.copy()), torch.tensor(int(row["label"]), dtype=torch.long)


class CachedBiLSTM(nn.Module):
    """Temporal classification head; no CNN reprocessing during epochs."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 2))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    seen = 0
    for features, labels in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = len(labels)
        loss_total += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += batch_size
    if not seen:
        raise ValueError("No batches evaluated")
    return {"loss": loss_total / seen, "accuracy": correct / seen}


def train(train_ds, val_ds, output: Path, epochs=20, patience=5, batch_size=8,
          hidden_dim=64, dropout=0.4, lr=1e-3, seed=42):
    """Train using train/val only, choosing checkpoint by validation loss."""
    if epochs < 1 or patience < 1 or batch_size < 1 or hidden_dim < 1:
        raise ValueError("epochs, patience, batch size and hidden dim must be positive")
    if not 0 <= dropout < 1 or lr <= 0:
        raise ValueError("Invalid dropout or learning rate")
    overlap = train_ds.source_paths & val_ds.source_paths
    if overlap:
        raise ValueError(f"Train/validation source leakage: {len(overlap)} shared paths")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Experiment directory already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    model = CachedBiLSTM(hidden_dim, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    history = []
    best_loss = float("inf")
    stale = 0
    checkpoint = output / "best_bilstm.pt"
    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, optimizer)
        val_stats = run_epoch(model, val_loader, criterion)
        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d} | train loss {record['train_loss']:.4f}, "
            f"acc {record['train_accuracy']:.3f} | val loss {record['val_loss']:.4f}, "
            f"acc {record['val_accuracy']:.3f}"
        )
        if val_stats["loss"] < best_loss - 1e-6:
            best_loss = val_stats["loss"]
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": "cached_efficientnet_b2_bilstm",
                "feature_dim": FEATURE_DIM,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "image_normalization": "imagenet",
                "class_names": ["Real", "Fake"],
                "seed": seed,
                "best_epoch": epoch,
                "validation_loss": best_loss,
            }, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered.")
                break
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    config = {
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "epochs_requested": epochs, "epochs_ran": len(history),
        "batch_size": batch_size, "hidden_dim": hidden_dim, "dropout": dropout,
        "learning_rate": lr, "seed": seed, "best_val_loss": best_loss,
        "test_used": False,
    }
    (output / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM using cached pretrained features")
    parser.add_argument("--cache-root", type=Path,
                        default=Path("outputs/feature_cache/efficientnet_b2_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/cached_bilstm_v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_ds = CachedFeatureDataset(args.cache_root / "train", expected_count=96)
    val_ds = CachedFeatureDataset(args.cache_root / "val", expected_count=20)
    results = train(
        train_ds, val_ds, args.output, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
        lr=args.learning_rate, seed=args.seed
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
