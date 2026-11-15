"""
DeepVision AI Evaluation Package.
"""

from src.evaluation.evaluator import (
    load_model,
    run_evaluation_inference,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc,
    save_predictions,
    evaluate,
    get_device,
)

__all__ = [
    "load_model",
    "run_evaluation_inference",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_roc",
    "save_predictions",
    "evaluate",
    "get_device",
]
