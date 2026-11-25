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
                max_samples: int | None = None) -> dict:
    """Write one .npy embedding per original sequence and a matching index."""
    import pandas as pd

    ds = SequenceDataset(split_csv, sequence_root=sequence_root,
                         image_size=224, imagenet_normalize=True)
    pool = torch.nn.AdaptiveAvgPool2d(1)
    destination.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for idx in range(len(ds) if max_samples is None else min(len(ds), max_samples)):
        sample = ds[idx]
        # Use the stable row number to avoid duplicate filenames across sources.
        target = destination / f"{idx:05d}.npy"
        features = encode_sequence(sample["sequence"], backbone, pool, frame_batch_size)
        if features.ndim != 2 or features.shape[1] != 1408:
            raise RuntimeError(f"Unexpected EfficientNet-B2 feature dimensions: {features.shape}")
        np.save(target, features, allow_pickle=False)
        index_rows.append({
            "source_path": str(ds.data.iloc[idx].get("sequence_path", "")),
            "video_name": sample["video_name"],
            "dataset": sample["dataset"],
            "label": int(sample["label"]),
            "feature_path": target.name,
            "frames": int(features.shape[0]),
        })
    pd.DataFrame(index_rows, columns=[
        "source_path", "video_name", "dataset", "label", "feature_path", "frames"
    ]).to_csv(destination / "index.csv", index=False)
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
    args = parser.parse_args()
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
    if (destination / "index.csv").exists():
        raise FileExistsError(f"Cache already exists at {destination}; choose a new --output-root")
    result = cache_split(csv, args.sequence_root, destination, backbone,
                         frame_batch_size=args.frame_batch_size,
                         max_samples=args.max_samples)
    result["pretrained"] = not args.untrained_smoke_test
    (destination / "metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
