"""Read-only audit of DeepVision dataset and raw-MP4 preprocessing parity.

This does not run inference, modify sequences, or inspect test labels.
It can verify stored aligned JPEG frames -> sequence arrays exactly when files exist.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, UnidentifiedImageError

from scripts.predict_cached_bilstm import select_uniform_indices


def compare_aligned_directory(sequence: np.ndarray, frames_dir: Path) -> dict:
    """Reconstruct the historical 30-frame sequence from sorted readable JPEGs."""
    if not frames_dir.is_dir():
        return {
            "status": "unavailable",
            "reason": f"Aligned JPEG directory not found: {frames_dir}",
        }
    paths = sorted(frames_dir.glob("*.jpg"))
    frames = []
    for path in paths:
        try:
            with Image.open(path) as image:
                frames.append(np.array(image.convert("RGB")))
        except (UnidentifiedImageError, OSError):
            continue
    if len(frames) < 30:
        return {
            "status": "insufficient_aligned_frames",
            "jpeg_files": len(paths),
            "decoded_jpegs": len(frames),
        }
    indices = select_uniform_indices(len(frames))
    sampled = [frames[int(i)] for i in indices]
    if any(frame.shape != sequence.shape[1:] for frame in sampled):
        return {
            "status": "shape_mismatch",
            "stored_frame_shape": list(sequence.shape[1:]),
            "aligned_frame_shapes": sorted({str(x.shape) for x in sampled}),
            "decoded_jpegs": len(frames),
        }
    reconstructed = np.stack(sampled).astype(np.uint8)
    diff = np.abs(reconstructed.astype(np.int16) - sequence.astype(np.int16))
    return {
        "status": "exact_match" if np.array_equal(reconstructed, sequence) else "pixel_mismatch",
        "decoded_jpegs": len(frames),
        "sample_indices": indices.tolist(),
        "exact_equal": bool(np.array_equal(reconstructed, sequence)),
        "mean_absolute_pixel_error": float(diff.mean()),
        "max_absolute_pixel_error": int(diff.max()),
    }


def audit_sequence(sequence_path: Path, aligned_dir: Path | None = None,
                   configuration_face_size: int = 224) -> dict:
    """Audit only local source files; labels and model scores are not used."""
    if not sequence_path.is_file():
        raise FileNotFoundError(sequence_path)
    seq = np.load(sequence_path, allow_pickle=False)
    if seq.ndim != 4 or seq.shape[0] != 30 or seq.shape[-1] != 3:
        raise ValueError(f"Expected (30,H,W,3), received {seq.shape}")
    if seq.dtype != np.uint8 or not np.isfinite(seq).all():
        raise ValueError("Expected uint8 RGB source sequence")
    if np.count_nonzero(seq) == 0:
        raise ValueError("Zero-filled placeholder sequence cannot verify parity")

    height, width = seq.shape[1:3]
    report = {
        "source_sequence": str(sequence_path),
        "sequence_shape": list(seq.shape),
        "sequence_dtype": str(seq.dtype),
        "configured_face_size": configuration_face_size,
        "configured_face_size_matches_stored": (
            height == width == configuration_face_size
        ),
        "pipeline_from_repo": {
            "sampling_fps": 5,
            "frame_sampling": "step=max(1,round(video_fps/5)); indexed from zero",
            "first_jpeg_stage": "cv2.imwrite(*.jpg) default settings",
            "face_detection": "MTCNN(keep_all=False,margin=0,post_process=False)",
            "aligned_jpeg_stage": "PIL.Image.save(*.jpg) default settings",
            "sequence_sampling": "np.linspace(0,n_valid-1,30).astype(int)",
            "feature_preprocessing": "OpenCV resize 224 + /255 + ImageNet normalization",
        },
        "verified_vs_raw_mp4": False,
        "limitations": [],
    }
    if not report["configured_face_size_matches_stored"]:
        report["limitations"].append(
            "Stored sequence face dimensions differ from the current default "
            "MTCNN face size; cannot assert historical preprocessing parity."
        )
    if aligned_dir is not None:
        report["aligned_directory_comparison"] = compare_aligned_directory(seq, aligned_dir)
        status = report["aligned_directory_comparison"]["status"]
        if status != "exact_match":
            report["limitations"].append(
                f"Aligned JPEGs -> sequence reconstruction not verified ({status})."
            )
    else:
        report["aligned_directory_comparison"] = {
            "status": "not_checked", "reason": "Pass --aligned-dir for a reconstruction test."
        }
    report["limitations"].append(
        "Even an exact JPEG-to-sequence match does not establish raw MP4 decoder, "
        "face detector weight/version, or face-crop parity."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Read-only source preprocessing audit")
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path)
    parser.add_argument("--configured-face-size", type=int, default=224)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.configured_face_size <= 0:
        parser.error("configured-face-size must be positive")
    result = audit_sequence(
        args.sequence, args.aligned_dir,
        configuration_face_size=args.configured_face_size,
    )
    print(json.dumps(result, indent=2))
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Audit report exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
