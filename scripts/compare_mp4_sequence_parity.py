"""Read-only source-MP4 to stored-sequence comparison for provenance auditing.

No model inference, no labels, no output changes to any source assets.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from scripts.predict_cached_bilstm import mp4_to_sequence


def frame_differences(reference: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute descriptive pixel differences; alignment equality is not presumed."""
    if reference.ndim != 4 or reconstructed.ndim != 4:
        raise ValueError("Expected two 4-D RGB sequences")
    if reference.shape[0] != 30 or reconstructed.shape[0] != 30:
        raise ValueError("Expected exactly 30 frames in both sequences")
    if reference.shape[-1] != 3 or reconstructed.shape[-1] != 3:
        raise ValueError("Expected three RGB channels")
    if reference.dtype != np.uint8 or reconstructed.dtype != np.uint8:
        raise ValueError("Both sequences must be uint8")
    same_dimensions = reference.shape == reconstructed.shape
    if same_dimensions:
        aligned = reconstructed
        mode = "native_dimensions"
    else:
        aligned = np.stack([
            cv2.resize(frame, (reference.shape[2], reference.shape[1]),
                       interpolation=cv2.INTER_LINEAR)
            for frame in reconstructed
        ])
        mode = "reconstructed_resized_to_reference_for_diagnostic_only"
    diff = np.abs(reference.astype(np.int16) - aligned.astype(np.int16))
    mae_by_frame = diff.mean(axis=(1, 2, 3))
    return {
        "reference_shape": list(reference.shape),
        "reconstructed_shape": list(reconstructed.shape),
        "native_shapes_match": same_dimensions,
        "comparison_mode": mode,
        "exact_pixel_match": bool(same_dimensions and np.array_equal(reference, reconstructed)),
        "mean_absolute_pixel_error": float(diff.mean()),
        "max_absolute_pixel_error": int(diff.max()),
        "per_frame_mae": [float(x) for x in mae_by_frame],
        "note": (
            "Frame-to-frame pixel difference may reflect face-box shifts, "
            "temporal sampling differences, decoding, JPEG encoding, or crop size; "
            "it is not a measure of classifier accuracy."
        ),
    }


def compare_original_to_stored(raw_video: Path, source_sequence: Path,
                               face_size: int = 260,
                               max_sampled_frames: int = 300) -> dict:
    if not source_sequence.is_file():
        raise FileNotFoundError(source_sequence)
    reference = np.load(source_sequence, allow_pickle=False)
    if (reference.ndim != 4 or reference.shape[0] != 30
            or reference.shape[-1] != 3 or reference.dtype != np.uint8):
        raise ValueError("Reference sequence must be 30 uint8 RGB frames")
    if np.count_nonzero(reference) == 0:
        raise ValueError("Reference is an all-zero placeholder, refusing audit")
    reconstructed, preprocessing = mp4_to_sequence(
        raw_video, face_size=face_size,
        max_sampled_frames=max_sampled_frames,
    )
    return {
        "audit_type": "raw_mp4_to_stored_sequence_diagnostic",
        "source_video": str(raw_video),
        "stored_sequence": str(source_sequence),
        "candidate_face_size": face_size,
        "preprocessing": preprocessing,
        "pixel_comparison": frame_differences(reference, reconstructed),
        "historical_settings_proven": False,
        "model_inference_performed": False,
        "model_weights_changed": False,
        "interpretation": (
            "Even an exact match would verify only this specific video and "
            "processing environment. A mismatch does not, on its own, "
            "establish the cause of an external-video false positive."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare original FF++ MP4 preprocessing to existing stored .npy"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--face-size", type=int, default=260)
    parser.add_argument("--max-sampled-frames", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.face_size < 32 or args.max_sampled_frames < 30:
        parser.error("--face-size must be >=32; --max-sampled-frames must be >=30")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Audit report exists and cannot be overwritten: {args.output}")
    report = compare_original_to_stored(
        args.video, args.sequence,
        face_size=args.face_size,
        max_sampled_frames=args.max_sampled_frames,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
