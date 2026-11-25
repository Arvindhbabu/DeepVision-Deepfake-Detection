"""CPU inference using the ACTUAL cached-feature BiLSTM checkpoint.

Supports verified-format face-sequence .npy files and experimental raw MP4 input.
Raw MP4 preprocessing is approximate until parity with the original dataset is audited.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torchvision import models

from scripts.cache_efficientnet_features import encode_sequence
from scripts.compare_cached_models import load_checkpoint_model

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prepare_sequence(sequence: np.ndarray) -> torch.Tensor:
    """Same 224px/255/ImageNet normalization as SequenceDataset in caching."""
    if sequence.ndim != 4 or sequence.shape[0] != 30 or sequence.shape[-1] != 3:
        raise ValueError("Expected 30 RGB frames, shape (30,H,W,3)")
    if sequence.dtype != np.uint8 or not np.isfinite(sequence).all():
        raise ValueError("Expected finite uint8 RGB images; no float/zero placeholders")
    if sequence.min() == 0 and sequence.max() == 0:
        raise ValueError("All-zero placeholder sequence")
    frames = [
        cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        if frame.shape[:2] != (224, 224) else frame
        for frame in sequence
    ]
    tensor = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    mean = tensor.new_tensor(MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def select_uniform_indices(n: int, length: int = 30) -> np.ndarray:
    if n < length:
        raise ValueError(f"Need at least {length} successfully aligned faces; got {n}")
    return np.linspace(0, n - 1, length).astype(int)


def mp4_to_sequence(path: Path, *, sample_fps: float = 5.0,
                    max_sampled_frames: int = 300,
                    face_size: int = 224) -> tuple[np.ndarray, dict]:
    """Experimental raw-video preprocessing based on the historical dataset pipeline.

    Decode at ~5 fps, JPEG encode as in dataset frame extraction, MTCNN align,
    encode aligned faces as JPEG, then uniformly select 30 valid crops.
    Stop after max_sampled_frames to bound CPU cost; long videos are truncated.
    """
    from facenet_pytorch import MTCNN
    from PIL import Image
    if sample_fps <= 0 or max_sampled_frames < 30 or face_size < 32:
        raise ValueError("Invalid frame extraction parameters")
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    mtcnn = MTCNN(image_size=face_size, margin=0, keep_all=False,
                  device="cpu", post_process=False)
    aligned = []
    sampled = 0
    total = 0
    truncated = False
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video must report valid FPS")
        step = max(1, int(round(fps / sample_fps)))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if total % step == 0:
                if sampled >= max_sampled_frames:
                    truncated = True
                    break
                sampled += 1
                # Historical pipeline stored sampled frames as JPEG first.
                ok_jpg, jpeg = cv2.imencode(".jpg", frame)
                if not ok_jpg:
                    total += 1
                    continue
                bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                face = mtcnn(Image.fromarray(rgb, mode="RGB"))
                if face is not None:
                    arr = face.permute(1, 2, 0).cpu().numpy()
                    crop = arr.clip(0, 255).round().astype(np.uint8)
                    # Original aligned frames were saved as JPEG and read back.
                    encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))[1]
                    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    aligned.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
            total += 1
    finally:
        cap.release()
    indices = select_uniform_indices(len(aligned))
    return np.stack([aligned[i] for i in indices]), {
        "decoded_frames": total,
        "fps": fps,
        "sampled_frames": sampled,
        "aligned_faces": len(aligned),
        "max_sampled_frames": max_sampled_frames,
        "truncated": truncated,
        "preprocessing_parity": "approximate_not_audited",
    }


def classify_sequence(sequence: np.ndarray, checkpoint: Path,
                      frame_batch_size: int = 1) -> dict:
    if frame_batch_size < 1:
        raise ValueError("frame_batch_size must be positive")
    features_input = prepare_sequence(sequence)
    backbone = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.DEFAULT
    ).features.cpu().eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        features = encode_sequence(
            features_input, backbone, torch.nn.AdaptiveAvgPool2d(1),
            frame_batch_size=frame_batch_size,
        )
        model = load_checkpoint_model(checkpoint)
        logits = model(torch.from_numpy(features.copy()).unsqueeze(0))
        fake_score = float(logits.softmax(dim=-1)[0, 1])
    return {
        "label": "Fake" if fake_score >= 0.5 else "Real",
        "fake_score": fake_score,
        "real_score": 1.0 - fake_score,
        "threshold": 0.5,
        "model": "pretrained_efficientnet_b2_cached_feature_bilstm",
        "warning": "Scores are uncalibrated and not forensic proof.",
    }


def main():
    parser = argparse.ArgumentParser(description="DeepVision cached BiLSTM video inference")
    parser.add_argument("--input", type=Path, required=True,
                        help="Face-aligned 30-frame .npy or experimental .mp4")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt"))
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--max-sampled-frames", type=int, default=300)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    path = args.input
    if path.suffix.lower() == ".npy":
        sequence = np.load(path, allow_pickle=False)
        provenance = {"input_type": "prealigned_npy", "preprocessing_parity": "training_format"}
    elif path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        sequence, provenance = mp4_to_sequence(
            path, max_sampled_frames=args.max_sampled_frames
        )
        provenance["input_type"] = "raw_video_experimental"
    else:
        parser.error("Input must be .npy or .mp4/.mov/.avi/.mkv")
    prediction = classify_sequence(
        sequence, args.checkpoint, frame_batch_size=args.frame_batch_size
    )
    print(json.dumps({**prediction, "input": str(path), "preprocessing": provenance}, indent=2))


if __name__ == "__main__":
    main()
