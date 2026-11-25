# Locked held-out FF++ test evaluation

The selected pretrained EfficientNet-B2 + cached BiLSTM checkpoint was picked by validation loss at epoch 16. The comparison with a mean-pooled logistic regression baseline used only training and validation data. Both models now undergo **one** evaluation on the reserved 22-video test split. This is a small, internal FF++ test set, not an external generalization claim.

From the project root in PowerShell:

```powershell
python -m unittest tests.test_locked_evaluation
python scripts/evaluate_locked_test.py
```

The script loads cached train/validation/test features, rejects duplicate sequence paths across splits, loads the locked checkpoint, fits the logistic baseline on training only, and evaluates both methods at the fixed 0.5 threshold. It writes `outputs/experiments/locked_test_evaluation_v1.json` once, refusing to overwrite an existing report. It records checkpoint and split-index SHA256 hashes, confusion matrices, and multiple performance metrics. Do not change thresholds or checkpoint selections based on the test result; do not run repeated tests searching for a more favorable number.

The existing training checkpoint is `outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt`. If this file is absent, do not substitute a different checkpoint after inspecting the test. The results should be reported with the sample size (22), limitations and no claims of calibrated probabilities. A future cross-dataset study would require new, independent data.
