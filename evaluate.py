"""
=============================================================
DeepVision AI

Model Evaluation Script

Author : Arvindh Babu
=============================================================
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.config import load_config
from src.datasets.dataloader import create_dataloaders
from src.models.model_factory import ModelFactory
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


# ============================================================
# Argument Parser
# ============================================================

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


# ============================================================
# Device Detection
# ============================================================

def get_device():
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


# ============================================================
# Load Model
# ============================================================

def load_model(config, checkpoint_path, device):
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


# ============================================================
# Evaluation Inference Loop
# ============================================================

def run_evaluation_inference(model, test_loader, device):
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


# ============================================================
# Metrics & Report Generation
# ============================================================

def compute_metrics(results, output_dir):
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
        "num_samples": len(labels),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n==============================")
    print("Evaluation Results")
    print("==============================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return metrics, cm


# ============================================================
# Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(cm, output_dir):
    if cm.sum() == 0:
        return

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


# ============================================================
# Plot ROC Curve
# ============================================================

def plot_roc(results, output_dir):
    labels = results["labels"]
    probabilities = results["probabilities"]

    if len(set(labels)) <= 1:
        print("[NOTE] Skipping ROC plot because target labels contain only one class.")
        return

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


# ============================================================
# Save Prediction CSV
# ============================================================

def save_predictions(results, output_dir):
    if len(results["labels"]) == 0:
        return None

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


# ============================================================
# Main Entry Point
# ============================================================

def main():
    args = parse_arguments()
    config = load_config(args.config)
    device = get_device()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = create_dataloaders(
        config,
        synthetic_fallback=args.synthetic_fallback,
    )

    model = load_model(config, args.checkpoint, device)

    results = run_evaluation_inference(model, test_loader, device)

    save_predictions(results, output_dir)

    metrics, cm = compute_metrics(results, output_dir)

    plot_confusion_matrix(cm, output_dir)

    plot_roc(results, output_dir)

    print("\n" + "=" * 60)
    print("Evaluation Completed Successfully")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
