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
        description="DeepVision AI Single Video / Multi-Clip Deepfake Inference CLI"
    )

    parser.add_argument(
        "--input",
        "--video",
        dest="input",
        type=str,
        required=True,
        help="Path to input video file or image sequence",
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

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    config = load_config(args.config)

    print("=" * 60)
    print("DeepVision AI — Inference Engine")
    print("=" * 60)
    print(f"Input File  : {input_path.name}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Model       : {config['model']['name']}")
    print(f"Clips       : {args.clips}")
    print("=" * 60)

    predictor = DeepfakePredictor(
        config=config,
        checkpoint_path=checkpoint_path,
    )

    label, confidence, clip_probs = predictor.predict_video(
        video_path=input_path,
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
