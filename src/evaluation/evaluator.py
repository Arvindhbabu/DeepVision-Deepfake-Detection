"""
DeepVision AI — Model Evaluation Engine.

Provides core evaluation inference, metric computation, confusion matrix plotting,
ROC curve plotting, and prediction recording.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)

from src.models.model_factory import ModelFactory


def get_device() -> torch.device:
    """Detect and return available compute device (CUDA / CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=" * 60)
        print("CUDA AVAILABLE")
        print("=" * 60)
        print("GPU :", torch.cuda.get_device_name(0))
        print("=" * 60)
    else:
        device = torch.device("cpu")
        print("=" * 60)
        print("Running on CPU")
        print("=" * 60)

    return device


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Instantiate model via ModelFactory and load weights from checkpoint safely."""
    print("\nInstantiating model via ModelFactory...")
    model = ModelFactory.create(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Checkpoint Loaded Successfully")
    return model


def run_evaluation_inference(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Run model inference over the test dataset."""
    print("\nRunning test set inference...")

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_video_names = []
    all_dataset_names = []

    if len(test_loader) == 0:
        print("[WARNING] Test dataloader is empty.")
        return {
            "labels": [],
            "predictions": [],
            "probabilities": [],
            "video_names": [],
            "dataset_names": [],
        }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            sequences = batch["sequence"].to(device)
            labels = batch["label"].to(device)
            dataset_names = batch.get("dataset", ["Unknown"] * len(labels))
            video_names = batch.get("video_name", [f"video_{i}" for i in range(len(labels))])

            logits = model(sequences)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            fake_probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(fake_probabilities.cpu().numpy().tolist())
            all_video_names.extend(video_names)
            all_dataset_names.extend(dataset_names)

    print("Inference Complete")
    return {
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "video_names": all_video_names,
        "dataset_names": all_dataset_names,
    }


def compute_metrics(results: Dict[str, Any], output_dir: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute and save core classification metrics, confusion matrix, and report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nComputing evaluation metrics...")

    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]

    if len(labels) == 0:
        print("[WARNING] No predictions available to compute metrics.")
        return {}, np.zeros((2, 2))

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # ROC AUC safety check (requires both classes present in labels)
    unique_labels = set(labels)
    if len(unique_labels) > 1:
        roc_auc = float(roc_auc_score(labels, probabilities))
    else:
        roc_auc = 0.0
        print("[NOTE] Only one class present in test set labels; ROC AUC defaulting to 0.0")

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((recall + specificity) / 2.0)

    cm_df = pd.DataFrame(
        cm,
        index=["Real", "Fake"],
        columns=["Pred Real", "Pred Fake"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    report = classification_report(
        labels,
        predictions,
        target_names=["Real", "Fake"] if len(unique_labels) > 1 else None,
        digits=4,
        zero_division=0,
    )

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "num_samples": len(labels),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n==============================")
    print("Evaluation Results")
    print("==============================")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"ROC AUC          : {roc_auc:.4f}")
    print(f"Specificity      : {specificity:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Samples Evaluated: {len(labels)}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return metrics, cm


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path):
    """Plot and save confusion matrix heatmap."""
    if cm.sum() == 0:
        return

    output_dir = Path(output_dir)
    print("Generating Confusion Matrix plot...")
    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"],
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    plt.title("Confusion Matrix — DeepVision AI")
    plt.tight_layout()

    save_path = output_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc(results: Dict[str, Any], output_dir: Path):
    """Plot and save ROC curve."""
    labels = results["labels"]
    probabilities = results["probabilities"]

    if len(set(labels)) <= 1:
        print("[NOTE] Skipping ROC plot because target labels contain only one class.")
        return

    output_dir = Path(output_dir)
    print("Generating ROC Curve plot...")
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — DeepVision AI")
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = output_dir / "roc_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def save_predictions(results: Dict[str, Any], output_dir: Path) -> Optional[pd.DataFrame]:
    """Save clip/video level predictions to a CSV file."""
    if len(results["labels"]) == 0:
        return None

    output_dir = Path(output_dir)
    print("Saving predictions CSV...")
    df = pd.DataFrame({
        "video_name": results["video_names"],
        "dataset": results["dataset_names"],
        "true_label": results["labels"],
        "predicted_label": results["predictions"],
        "fake_probability": results["probabilities"],
    })
    df["correct"] = df["true_label"] == df["predicted_label"]

    csv_path = output_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return df


def evaluate(
    config: dict,
    checkpoint_path: Path,
    output_dir: Path,
    synthetic_fallback: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run the end-to-end evaluation pipeline.
    """
    from src.datasets.dataloader import create_dataloaders

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device()

    _, _, test_loader = create_dataloaders(
        config,
        synthetic_fallback=synthetic_fallback,
    )

    model = load_model(config, Path(checkpoint_path), device)
    results = run_evaluation_inference(model, test_loader, device)
    save_predictions(results, output_dir)
    metrics, cm = compute_metrics(results, output_dir)
    plot_confusion_matrix(cm, output_dir)
    plot_roc(results, output_dir)

    return metrics

