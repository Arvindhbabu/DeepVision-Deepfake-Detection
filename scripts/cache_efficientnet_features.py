"""Cache frozen EfficientNet-B2 frame embeddings for CPU-friendly experiments.

Run from the repository root. Never mixes train/validation/test splits.
The cached features are not predictions or proof of model accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torchvision import models

from src.datasets.sequence_dataset import SequenceDataset


def encode_sequence(sequence: torch.Tensor, backbone: torch.nn.Module,
                    pool: torch.nn.Module, frame_batch_size: int = 1) -> np.ndarray:
    """Convert (T,3,H,W) normalized frames to (T,1408) float32 features."""
    if frame_batch_size < 1 or sequence.ndim != 4:
        raise ValueError("Expected (T,C,H,W) sequence and positive frame batch size")
    backbone.eval()
    blocks = []
    with torch.inference_mode():
        for part in sequence.split(frame_batch_size, dim=0):
            output = pool(backbone(part))
            blocks.append(output.flatten(1).cpu().numpy().astype(np.float32))
    return np.concatenate(blocks, axis=0)


def cache_split(split_csv: Path, sequence_root: Path, destination: Path,
                backbone: torch.nn.Module, frame_batch_size: int = 1,
                max_samples: int | None = None, resume: bool = False) -> dict:
    """Write one .npy embedding per original sequence and a matching index."""
    import pandas as pd

    ds = SequenceDataset(split_csv, sequence_root=sequence_root,
                         image_size=224, imagenet_normalize=True)
    pool = torch.nn.AdaptiveAvgPool2d(1)
    columns = ["source_path", "video_name", "dataset", "label", "feature_path", "frames"]
    index_path = destination / "index.csv"
    destination.mkdir(parents=True, exist_ok=True)

    if not resume and any(destination.iterdir()):
        raise FileExistsError(
            f"Output already exists at {destination}; use --resume for the same "
            "pretrained cache or choose a new --output-root."
        )
    if resume and not index_path.is_file() and any(destination.iterdir()):
        raise FileNotFoundError(
            f"Cannot resume {destination}: index.csv is absent; existing files "
            "cannot be linked safely to verified split rows."
        )

    prior = {}
    if resume and index_path.is_file():
        existing = pd.read_csv(index_path)
        if not set(columns).issubset(existing.columns):
            raise ValueError(f"Feature cache index has missing columns: {index_path}")
        for i, row in existing.iterrows():
            if i >= len(ds):
                raise ValueError("Cached index has more rows than the current split")
            expected = ds.data.iloc[i]
            if (str(row["source_path"]) != str(expected.get("sequence_path", ""))
                    or str(row["video_name"]) != str(expected["video_name"])
                    or str(row["dataset"]) != str(expected["dataset"])
                    or int(row["label"]) != int(expected["label"])
                    or str(row["feature_path"]) != f"{i:05d}.npy"):
                raise ValueError(
                    f"Cache metadata does not match split row {i}; "
                    "do not resume against changed splits."
                )
            prior[i] = row.to_dict()

    index_rows = []
    limit = len(ds) if max_samples is None else min(len(ds), max_samples)
    for idx in range(limit):
        row = ds.data.iloc[idx]
        target = destination / f"{idx:05d}.npy"
        cached = prior.get(idx)
        if cached and target.is_file():
            features = np.load(target, mmap_mode="r", allow_pickle=False)
            if (features.ndim != 2 or features.shape != (int(cached["frames"]), 1408)
                    or features.dtype != np.float32 or not np.isfinite(features).all()):
                raise ValueError(f"Cached feature file is invalid: {target}")
            index_rows.append(cached)
            continue

        sample = ds[idx]
        features = encode_sequence(sample["sequence"], backbone, pool, frame_batch_size)
        if (features.ndim != 2 or features.shape[1] != 1408
                or not np.isfinite(features).all()):
            raise RuntimeError(f"Unexpected EfficientNet-B2 features for {target}: {features.shape}")

        # A partially written output cannot be mistaken for a complete cache entry.
        temporary = destination / f"{idx:05d}.tmp.npy"
        np.save(temporary, features, allow_pickle=False)
        temporary.replace(target)

        index_rows.append({
            "source_path": str(row.get("sequence_path", "")),
            "video_name": sample["video_name"],
            "dataset": sample["dataset"],
            "label": int(sample["label"]),
            "feature_path": target.name,
            "frames": int(features.shape[0]),
        })
        # Save the index after each sequence for interruption-safe continuation.
        temporary_index = destination / "index.csv.tmp"
        pd.DataFrame(index_rows, columns=columns).to_csv(temporary_index, index=False)
        temporary_index.replace(index_path)
        print(f"[CACHE] {idx + 1}/{limit}: {sample['video_name']}")

    return {"sequences": len(index_rows), "destination": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientNet-B2 frozen feature cache")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--sequence-root", type=Path, default=Path("data/intermediate/sequences"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/feature_cache/efficientnet_b2"))
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--untrained-smoke-test", action="store_true",
                        help="Do not download weights; only check code execution. NOT for training/evaluation.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing pretrained cache without recomputing validated entries.")
    args = parser.parse_args()
    if args.resume and args.untrained_smoke_test:
        parser.error("Cannot resume random-weight smoke-test caches across model initializations")
    if args.frame_batch_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("frame-batch-size and max-samples must be positive")
    if not args.untrained_smoke_test:
        print("Loading pretrained EfficientNet-B2 weights (may download them once).")
    else:
        print("WARNING: Random backbone embeddings are ONLY a smoke test.")
    weights = None if args.untrained_smoke_test else models.EfficientNet_B2_Weights.DEFAULT
    backbone = models.efficientnet_b2(weights=weights).features.cpu().eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    csv = args.splits_dir / (args.split + ".csv")
    if not csv.is_file():
        raise FileNotFoundError(csv)
    destination = args.output_root / ("smoke_" + args.split if args.untrained_smoke_test else args.split)
    if args.resume and (destination / "index.csv").exists():
        metadata_path = destination / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume without metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("pretrained") is not True:
            raise ValueError("Refusing to reuse non-pretrained or unknown cache")
    result = cache_split(csv, args.sequence_root, destination, backbone,
                         frame_batch_size=args.frame_batch_size,
                         max_samples=args.max_samples, resume=args.resume)
    result["pretrained"] = not args.untrained_smoke_test
    (destination / "metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
