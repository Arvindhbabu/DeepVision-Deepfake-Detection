"""One-time, locked-protocol evaluation of the preselected cached BiLSTM.

Do not change model choice, checkpoint, or 0.5 threshold based on these results.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.compare_cached_models import (
    bilstm_scores,
    load_checkpoint_model,
    mean_pool_features,
    summarize,
)
from scripts.train_cached_bilstm import CachedFeatureDataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(cache_root: Path, checkpoint_path: Path) -> dict:
    train = CachedFeatureDataset(cache_root / "train", expected_count=96)
    validation = CachedFeatureDataset(cache_root / "val", expected_count=20)
    test = CachedFeatureDataset(cache_root / "test", expected_count=22)

    for name_a, a, name_b, b in [
        ("train", train, "validation", validation),
        ("train", train, "test", test),
        ("validation", validation, "test", test),
    ]:
        overlap = a.source_paths & b.source_paths
        if overlap:
            raise ValueError(f"Source leakage: {name_a} / {name_b} share {len(overlap)} paths")

    model = load_checkpoint_model(checkpoint_path)
    # This was selected on validation loss before the test split was evaluated.
    # The script MUST NOT switch checkpoints based on test scores.
    checkpoint_hash = sha256_file(checkpoint_path)

    x_train, y_train = mean_pool_features(train)
    x_test, y_test = mean_pool_features(test)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("Both Real and Fake must exist in the train and test sets")

    # Exact same simple baseline as the validation comparison. Fit only on train.
    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    baseline.fit(x_train, y_train)
    linear_scores = baseline.predict_proba(x_test)[:, 1]
    temporal_scores = bilstm_scores(model, test)

    return {
        "protocol": "locked_held_out_test_evaluation_v1",
        "status": "single_test_run_no_test_based_tuning",
        "threshold": 0.5,
        "label_0": "Real",
        "label_1": "Fake",
        "training_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_sha256": checkpoint_hash,
        "split_index_sha256": {
            name: sha256_file(cache_root / name / "index.csv")
            for name in ("train", "val", "test")
        },
        "metrics": {
            "linear_mean_pool": summarize(y_test, linear_scores),
            "cached_bilstm": summarize(y_test, temporal_scores),
        },
        "caveats": [
            "Only 22 test videos; estimates have wide uncertainty.",
            "Historical actor-disjoint split checks rely on FF++ filename metadata.",
            "FF++ is not an independent cross-dataset generalization benchmark.",
            "No threshold or checkpoint should be changed based on the test outcomes.",
            "Model scores are not calibrated forensic probabilities.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate locked checkpoint once on 22 held-out videos")
    parser.add_argument("--cache-root", type=Path,
                        default=Path("outputs/feature_cache/efficientnet_b2_v2"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/locked_test_evaluation_v1.json"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(
            f"Existing test report cannot be overwritten: {args.output}. "
            "Do not repeat evaluation to select a better score."
        )
    result = evaluate(args.cache_root, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # exclusive creation prevents accidental overwriting of a previous evaluation
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
