"""
=============================================================
DeepVision AI

Model Evaluation CLI

Author : Arvindh Babu
=============================================================
"""

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.evaluation.evaluator import (
    get_device,
    load_model,
    run_evaluation_inference,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc,
    save_predictions,
    evaluate,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DeepVision AI Model Evaluation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file path",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Allow synthetic dummy sequence generation if sequence files are missing",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate(
        config=config,
        checkpoint_path=Path(args.checkpoint),
        output_dir=output_dir,
        synthetic_fallback=args.synthetic_fallback,
    )

    print("\n" + "=" * 60)
    print("Evaluation Completed Successfully")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
