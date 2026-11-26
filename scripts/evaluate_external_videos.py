"""Evaluate a frozen DeepVision model on a NEW, independently labeled video manifest.

DO NOT use this to tune model weights/thresholds or select a final test set.
Original FF++ locked test is deliberately not read. The current model is
expected to generalize poorly; this workflow measures that gap transparently.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchvision import models

from scripts.cache_efficientnet_features import encode_sequence
from scripts.compare_cached_models import load_checkpoint_model
from scripts.predict_cached_bilstm import mp4_to_sequence, prepare_sequence

ALLOWED_LABELS = {"Real": 0, "Fake": 1}
EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def load_manifest(path: Path) -> list[dict]:
    """Requires explicit user labels, independent IDs, and existing video paths."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"video_path", "label", "group_id"}.issubset(reader.fieldnames):
            raise ValueError("Manifest needs video_path,label,group_id columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    result = []
    groups = {}
    seen_paths = set()
    for number, row in enumerate(rows, start=2):
        label = (row["label"] or "").strip()
        group = (row["group_id"] or "").strip()
        raw_path = (row["video_path"] or "").strip()
        video = Path(raw_path)
        if not video.is_absolute():
            video = (path.parent / video).resolve()
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Row {number}: expected label Real or Fake")
        if not group:
            raise ValueError(f"Row {number}: group_id required for source/identity leakage review")
        if video.suffix.lower() not in EXTENSIONS or not video.is_file():
            raise ValueError(f"Row {number}: video missing or unsupported: {video}")
        if str(video).casefold() in seen_paths:
            raise ValueError(f"Row {number}: duplicate video path")
        seen_paths.add(str(video).casefold())
        # Same source/actor group may have multiple manipulations, but mark it clearly.
        groups.setdefault(group, set()).add(label)
        result.append({"path": video, "label": ALLOWED_LABELS[label], "group_id": group})
    return result


def summarize_predictions(records: list[dict]) -> dict:
    successful = [r for r in records if r["status"] == "ok"]
    if not successful:
        return {"evaluated": 0, "error_count": len(records), "warning": "No usable predictions"}
    tn = sum(r["label"] == 0 and r["prediction"] == 0 for r in successful)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in successful)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in successful)
    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in successful)
    real_total, fake_total = tn + fp, tp + fn
    return {
        "evaluated": len(successful),
        "error_count": len(records) - len(successful),
        "real_videos": real_total,
        "fake_videos": fake_total,
        "accuracy": (tn + tp) / len(successful),
        "real_false_positive_rate": fp / real_total if real_total else None,
        "fake_recall": tp / fake_total if fake_total else None,
        "balanced_accuracy": ((tn / real_total) + (tp / fake_total)) / 2
        if real_total and fake_total else None,
        "confusion_matrix_real_fake": [[tn, fp], [fn, tp]],
        "warning": "Descriptive external benchmark; no calibration or new held-out claim.",
    }


def evaluate(manifest: Path, checkpoint: Path, face_size: int, frame_batch_size: int) -> dict:
    rows = load_manifest(manifest)
    if face_size not in {224, 260} or frame_batch_size < 1:
        raise ValueError("Choose face size 224 or 260 and frame batch >=1")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    backbone = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.DEFAULT
    ).features.cpu().eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    pool = torch.nn.AdaptiveAvgPool2d(1)
    model = load_checkpoint_model(checkpoint)
    records = []
    for idx, row in enumerate(rows, start=1):
        record = {
            "item": idx,
            "group_id": row["group_id"],
            "label": row["label"],
        }
        try:
            frames, info = mp4_to_sequence(
                row["path"], face_size=face_size, max_sampled_frames=300
            )
            with torch.inference_mode():
                features = encode_sequence(
                    prepare_sequence(frames), backbone, pool,
                    frame_batch_size=frame_batch_size
                )
                logits = model(torch.from_numpy(features.copy()).unsqueeze(0))
                score = float(logits.softmax(dim=-1)[0, 1].item())
            record.update({
                "status": "ok",
                "prediction": int(score >= 0.5),
                "fake_score_uncalibrated": score,
                "aligned_faces": info["aligned_faces"],
                "truncated": info["truncated"],
            })
        except (ValueError, OSError, RuntimeError) as exc:
            record.update({"status": "error", "error": str(exc)})
        records.append(record)
        print(f"[EXTERNAL] {idx}/{len(rows)}: {record['status']}", flush=True)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "protocol": "frozen_model_external_descriptive_evaluation",
        "manifest": str(manifest),
        "checkpoint_sha256": sha,
        "preprocessing": {
            "face_size": face_size,
            "frame_batch_size": frame_batch_size,
            "max_sampled_frames": 300,
            "parity": "not_fully_verified",
        },
        "metrics": summarize_predictions(records),
        "records": records,
        "test_split_reopened": False,
        "model_tuned": False,
        "limitations": [
            "Requires independently sourced, consented, correctly labeled samples.",
            "Do not reuse previously inspected wedding/Celeb-DF videos to choose hyperparameters.",
            "For manipulated videos, keep all derivatives of a source and identities in one group.",
            "Do not use known-test results to select thresholds or publish new accuracy claims.",
            "Small, selectively chosen external samples do not establish generalization.",
            "Output includes group IDs but intentionally omits full local file paths.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-checkpoint external video evaluation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--face-size", type=int, choices=[224, 260], default=260)
    parser.add_argument("--frame-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a prior evaluation: {args.output}")
    report = evaluate(args.manifest, args.checkpoint, args.face_size, args.frame_batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
