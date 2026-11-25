# Stage 3 — CPU-friendly EfficientNet feature caching

The verified data split inventory contains 96 train, 20 validation and 22 test
sequences. Do not mix samples across these splits or train on all-zero placeholders.

This script extracts **pretrained**, frozen EfficientNet-B2 frame embeddings.
It does NOT train a temporal head or establish deepfake detection accuracy.

From PowerShell in the repository root, run the fast regression test first:

```powershell
python -m unittest tests.test_feature_cache
```

Inspect just one real training sample, without downloading weights:

```powershell
python scripts/cache_efficientnet_features.py --split train --max-samples 1 --frame-batch-size 1 --untrained-smoke-test
```

This creates a smoke-only cache under `outputs/feature_cache/efficientnet_b2/smoke_train`.
**Never train or evaluate with this random-weight cache.** No full run is needed yet.

To generate meaningful pretrained embeddings later, run on verified split data
(without `--untrained-smoke-test`). For example:

```powershell
python scripts/cache_efficientnet_features.py --split train --max-samples 1 --frame-batch-size 1
```

The pretrained run may download torchvision EfficientNet-B2 weights, uses CPU,
and processes only one or a few frames at a time. Running all sequences may
take substantial CPU time. Obtain confirmation before a full extraction.

Each split produces one `index.csv`, per-sequence `.npy` features of shape
`(T, 1408)`, and `metadata.json`. Existing caches are protected against accidental overwrite. To continue the existing
pretrained train cache containing one video, use:

```powershell
python scripts/cache_efficientnet_features.py --split train --max-samples 3 --frame-batch-size 1 --resume
```

This retains video 1's saved pretrained features, then processes videos 2 and 3.
After validation you can omit `--max-samples` with `--resume` to finish
the remaining training split. The cache checks source metadata and feature
shapes before reuse. Only pretrained caches can resume across sessions;
random-weight smoke caches are deliberately non-resumable.

For other partitions, use `--split val` or `--split test`. Never train on
test features; reserve them for one final, locked-model evaluation. Completed
feature extraction alone does not establish model accuracy. A complete feature extraction stage must be
followed by a separate training/evaluation implementation with strict split
separation; no metrics should be claimed from this extraction script alone.
