"""
=============================================================
DeepVision AI

Inference CLI Tool

Author : Arvindh Babu
=============================================================
"""

import argparse
from pathlib import Path
from src.utils.config import load_config
from src.inference.predictor import DeepfakePredictor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DeepVision AI Single Video / Multi-Clip Deepfake Inference"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to model configuration YAML",
    )

    parser.add_argument(
        "--clips",
        type=int,
        default=5,
        help="Number of temporal clips to sample from video",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    config = load_config(args.config)

    print("=" * 60)
    print("DeepVision AI — Inference Tool")
    print("=" * 60)
    print(f"Video File  : {video_path.name}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Model       : {config['model']['name']}")
    print(f"Clips       : {args.clips}")
    print("=" * 60)

    predictor = DeepfakePredictor(
        config=config,
        checkpoint_path=args.checkpoint,
    )

    label, confidence, clip_probs = predictor.predict_video(
        video_path=video_path,
        num_clips=args.clips,
    )

    print("\nClip-wise Deepfake Probabilities:")
    for i, prob in enumerate(clip_probs, 1):
        print(f"  Clip {i:02d}: {prob * 100:.2f}%")

    print("\n" + "=" * 60)
    print(f"FINAL PREDICTION : {label}")
    print(f"CONFIDENCE       : {confidence:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
