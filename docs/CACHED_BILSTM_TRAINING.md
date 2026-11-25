# Stage 3 — Train a cached-feature BiLSTM on CPU

## What is trained?

A **bidirectional LSTM** with a two-class linear head learns video-level
Real/Fake labels from frozen pretrained EfficientNet-B2 embeddings.
The frozen CNN is not loaded during training. Feature input is `(30, 1408)`.

The training split contains **96** videos (48 Real/48 Fake); validation has
**20** (10 Real/10 Fake). The script never reads the **22-video test set**.
This small dataset makes results high variance; compare multiple seeds
and include meaningful baselines before treating conclusions as reliable.

## Requirements

The user must have finished pretrained feature extraction under:
`outputs/feature_cache/efficientnet_b2_v2/train` and `.../val`.
Each cache must include `index.csv`, `metadata.json`, and matching
`.npy` arrays; `metadata.json` must indicate `pretrained: true`.
Random-weight smoke-test features are rejected.

On Windows PowerShell from the project root:

```powershell
git fetch origin
git switch --track origin/stage3/cached-bilstm-training-20261125
python -m unittest tests.test_cached_bilstm_training
```

First conduct a **two-epoch real-data sanity run**:

```powershell
python scripts/train_cached_bilstm.py --epochs 2 --patience 2 --batch-size 4 --hidden-dim 64 --output outputs/experiments/cached_bilstm_sanity
```

This is real training on real cached data, not a synthetic smoke test.
Inspect the training/validation losses before choosing more epochs.
The CPU runs only the lightweight LSTM head. It will produce:
- `best_bilstm.pt` (checkpoint with preprocessing/architecture metadata)
- `history.csv` (training and validation loss/accuracy per epoch)
- `experiment.json` (hyperparameters and counts)

For a longer run **after sanity validation**, choose a different output
directory, e.g.:

```powershell
python scripts/train_cached_bilstm.py --epochs 25 --patience 5 --batch-size 8 --hidden-dim 64 --output outputs/experiments/cached_bilstm_full_v1
```

The checkpoint must be used with the **same frozen pretrained EfficientNet-B2**
feature extractor. It is **not** interchangeable with the original raw-frame
EfficientNet+BiLSTM checkpoint used by earlier code.

## Safe and honest interpretation

- Training accuracy alone does not measure generalization.
- Use validation performance only to select the experiment and checkpoint.
- Keep test features untouched until the model and selection method are locked.
- Do not claim the model is ready for forensic use or that a softmax score is
  a calibrated probability of manipulation.
- The extractor currently checks feature dimensions and finite values; before
  final evaluation, independently audit source-video links and provenance.
