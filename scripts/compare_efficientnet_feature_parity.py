"""Compare pretrained EfficientNet-B2 features from original and reconstructed frames.

This is a diagnostic only: no class labels, no BiLSTM, and no test-set evaluation.
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

from scripts.cache_efficientnet_features import encode_sequence
from scripts.predict_cached_bilstm import mp4_to_sequence, prepare_sequence


def feature_distances(reference: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Measure per-frame similarities without interpreting them as probabilities."""
    if (reference.shape != reconstructed.shape or reference.ndim != 2
            or reference.shape != (30, 1408)):
        raise ValueError("Both feature arrays must have shape (30, 1408)")
    if not (np.isfinite(reference).all() and np.isfinite(reconstructed).all()):
        raise ValueError("Feature arrays must be finite")
    a = reference.astype(np.float64)
    b = reconstructed.astype(np.float64)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    if np.any(a_norm == 0) or np.any(b_norm == 0):
        raise ValueError("Zero-norm features cannot be compared by cosine similarity")
    cosine = np.sum(a * b, axis=1) / (a_norm * b_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    relative_l2 = np.linalg.norm(a - b, axis=1) / a_norm
    absolute = np.abs(a - b)
    return {
        "reference_shape": list(reference.shape),
        "reconstructed_shape": list(reconstructed.shape),
        "mean_cosine_similarity": float(cosine.mean()),
        "min_cosine_similarity": float(cosine.min()),
        "mean_relative_l2": float(relative_l2.mean()),
        "max_relative_l2": float(relative_l2.max()),
        "mean_absolute_feature_difference": float(absolute.mean()),
        "mean_cosine_by_frame": [float(value) for value in cosine],
        "relative_l2_by_frame": [float(value) for value in relative_l2],
        "note": (
            "Cosine similarity and relative L2 measure representation differences, "
            "not detector accuracy, calibrated confidence, or real/fake scores."
        ),
    }


def audit_features(video: Path, sequence_path: Path, face_size: int = 260,
                   max_sampled_frames: int = 300, frame_batch_size: int = 1) -> dict:
    if face_size < 32 or max_sampled_frames < 30 or frame_batch_size < 1:
        raise ValueError("Invalid preprocessing or feature batch parameters")
    if not sequence_path.is_file():
        raise FileNotFoundError(sequence_path)
    original = np.load(sequence_path, allow_pickle=False)
    reconstructed, preprocessing = mp4_to_sequence(
        video, face_size=face_size, max_sampled_frames=max_sampled_frames,
    )
    original_input = prepare_sequence(original)
    reconstructed_input = prepare_sequence(reconstructed)
    # Exactly the same ImageNet-pretrained backbone for both sequences.
    backbone = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.DEFAULT
    ).features.cpu().eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    pool = torch.nn.AdaptiveAvgPool2d(1)
    with torch.inference_mode():
        original_features = encode_sequence(
            original_input, backbone, pool, frame_batch_size=frame_batch_size,
        )
        reconstructed_features = encode_sequence(
            reconstructed_input, backbone, pool, frame_batch_size=frame_batch_size,
        )
    return {
        "audit_type": "training_input_efficientnet_b2_feature_parity",
        "video": str(video),
        "stored_sequence": str(sequence_path),
        "candidate_face_size": face_size,
        "preprocessing": preprocessing,
        "feature_metrics": feature_distances(original_features, reconstructed_features),
        "pretrained_backbone": "torchvision EfficientNet_B2_Weights.DEFAULT",
        "frame_preprocessing": "224x224 resize; RGB /255; ImageNet normalization",
        "classifier_invoked": False,
        "test_labels_used": False,
        "checkpoint_modified": False,
        "historical_preprocessing_parity_proven": False,
        "limitations": (
            "A single video may show similar features despite differences in crop "
            "size, JPEG encoding, sampling, or facenet-pytorch version. "
            "Do not tune based on locked test videos or use this metric as accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only FF++ pretrained EfficientNet-B2 feature parity audit"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--face-size", type=int, default=260)
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--max-sampled-frames", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Report exists; choose a new path: {args.output}")
    result = audit_features(
        args.video, args.sequence, args.face_size,
        args.max_sampled_frames, args.frame_batch_size,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
