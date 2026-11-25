"""Compare the frozen-feature BiLSTM with a simple mean-pooled logistic baseline.

Validation-only analysis. This script NEVER opens the held-out test cache.
"""
import argparse
import json
import sys
from pathlib import Path

# Allow direct execution: python scripts/compare_cached_models.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from scripts.train_cached_bilstm import CachedBiLSTM, CachedFeatureDataset


def mean_pool_features(dataset: CachedFeatureDataset) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for index in range(len(dataset)):
        video, label = dataset[index]
        features.append(video.mean(dim=0).numpy())
        labels.append(int(label))
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def summarize(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Use the same fixed 0.5 threshold for both models; do not tune on validation."""
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Scores and labels must have matching nonzero length")
    predictions = (scores >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    stats = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_fake": float(precision_score(labels, predictions, zero_division=0)),
        "recall_fake": float(recall_score(labels, predictions, zero_division=0)),
        "f1_fake": float(f1_score(labels, predictions, zero_division=0)),
        "specificity_real": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    stats["roc_auc"] = (
        float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    )
    return stats


def load_checkpoint_model(path: Path) -> CachedBiLSTM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "cached_efficientnet_b2_bilstm":
        raise ValueError("This is not a cached EfficientNet-B2 BiLSTM checkpoint")
    if checkpoint.get("feature_dim") != 1408:
        raise ValueError("Checkpoint feature dimensions do not match the cached features")
    model = CachedBiLSTM(
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def bilstm_scores(model: CachedBiLSTM, dataset: CachedFeatureDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    collected = []
    with torch.inference_mode():
        for features, _ in loader:
            logits = model(features)
            collected.extend(logits.softmax(dim=-1)[:, 1].tolist())
    return np.asarray(collected, dtype=np.float64)


def compare(cache_root: Path, checkpoint: Path) -> dict:
    train = CachedFeatureDataset(cache_root / "train", expected_count=96)
    validation = CachedFeatureDataset(cache_root / "val", expected_count=20)
    if train.source_paths & validation.source_paths:
        raise ValueError("Train/validation source paths overlap")
    train_vectors, train_y = mean_pool_features(train)
    val_vectors, val_y = mean_pool_features(validation)
    if len(np.unique(train_y)) != 2:
        raise ValueError("Training split must contain both classes")

    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    linear.fit(train_vectors, train_y)
    linear_scores = linear.predict_proba(val_vectors)[:, 1]

    model = load_checkpoint_model(checkpoint)
    temporal_scores = bilstm_scores(model, validation)
    return {
        "protocol": "validation_only_no_threshold_tuning",
        "train_count": len(train),
        "validation_count": len(validation),
        "test_used": False,
        "class_1": "Fake",
        "classification_threshold": 0.5,
        "linear_mean_pool": summarize(val_y, linear_scores),
        "cached_bilstm": summarize(val_y, temporal_scores),
        "checkpoint": str(checkpoint),
    }


def main():
    parser = argparse.ArgumentParser(description="Validation-only comparison against mean-pooled logistic regression")
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/feature_cache/efficientnet_b2_v2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/validation_comparison_v1.json"))
    args = parser.parse_args()
    if args.output.is_file():
        raise FileExistsError(f"Report already exists; choose a new --output: {args.output}")
    result = compare(args.cache_root, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
