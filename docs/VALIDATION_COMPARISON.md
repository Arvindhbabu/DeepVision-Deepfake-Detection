# Validation-only BiLSTM versus logistic-regression baseline

The prior 25-epoch experiment ran 21 epochs with early stopping. Its best
validation-loss checkpoint was selected at epoch 16, with a reported validation
accuracy of 95% on **20** videos. These are experiment observations, not
independent test accuracy or a generalization guarantee.

This step checks whether the additional temporal complexity helps beyond a
simple classifier operating on **mean-pooled pretrained EfficientNet-B2 features**.

The analysis:
- fits a `StandardScaler` and class-balanced `LogisticRegression` using only
  **96 training videos**, not the validation/test set;
- evaluates logistic regression and the frozen saved BiLSTM checkpoint on the
  same **20 validation videos**, at the same predefined 0.5 threshold;
- reports the confusion matrix, accuracy, precision, recall, F1, specificity,
  balanced accuracy and ROC-AUC;
- does not touch the 22 held-out test videos.

PowerShell:

```powershell
git fetch origin
git switch --track origin/stage3/validation-baselines-20261125
python -m unittest tests.test_validation_comparison
python scripts/compare_cached_models.py
```

The default checkpoint is
`outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt`.
The JSON output is written to
`outputs/experiments/validation_comparison_v1.json`.
Existing comparison reports will not be overwritten.

**Interpretation:** The validation split has just 20 samples, and we have
already inspected its metrics repeatedly. Treat this comparison as exploratory
development evidence, not an unbiased generalization estimate. Lock the final
protocol before using the test split once. Do not claim 95% test accuracy.
