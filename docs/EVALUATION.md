# DeepVision AI — Evaluation Pipeline Specification

## 1. Overview

The evaluation pipeline (`evaluate.py`) evaluates model checkpoints against the test set split, computing classification metrics and generating visual evaluation artifacts.

---

## 2. Command Execution

```bash
python evaluate.py \
    --config configs/default.yaml \
    --checkpoint outputs/runs/run_20260913_120000/best_model.pth \
    --output outputs/evaluation
```

---

## 3. Evaluated Metrics & Outputs

- **Accuracy**: Overall classification accuracy on test sequences.
- **Precision**: Proportion of true deepfakes among predicted deepfakes.
- **Recall**: Proportion of actual deepfakes correctly identified.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Area Under Receiver Operating Characteristic Curve (with automatic single-class edge case handling).
- **Generated Artifacts**:
  - `metrics.json`: JSON file with all scalar metrics.
  - `confusion_matrix.csv` & `confusion_matrix.png`: Confusion matrix data and display plot.
  - `roc_curve.png`: ROC curve plot.
  - `predictions.csv`: Per-video ground truth vs predicted labels and fake probability scores.
