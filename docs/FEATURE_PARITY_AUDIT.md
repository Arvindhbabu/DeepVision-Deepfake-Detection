# Stage 6 — Frozen EfficientNet feature parity

Read-only follow-up to the pixel comparison for **training-only FF++ originals**.
Three FF++ raw originals were already compared against local aligned sequences
at candidate MTCNN image size 260:

| Video | Sampled faces | Sequence pixel MAE on 0–255 scale |
| --- | ---: | ---: |
| 030 | 50 | 1.653968 |
| 033 | 162 | 1.304481 |
| 049 | 60 | 1.582088 |

All three reconstructed sequences had 30 RGB frames at 260x260, but
**none matched byte-for-byte**. These diagnostics do not establish historical
MTCNN/image-codec version parity, and should not be interpreted as accuracy.

## Run new feature-level audit

From the Stage 6 branch:

```powershell
git pull --ff-only
python -m unittest tests.test_efficientnet_feature_parity tests.test_mp4_sequence_parity tests.test_preprocessing_audit
python scripts/compare_efficientnet_feature_parity.py --video "data/raw/ffpp_c23/original/030.mp4" --sequence "data/intermediate/sequences/ffpp/030.npy" --face-size 260 --frame-batch-size 1 --output "outputs/reports/feature_parity_030_260.json"
```

The command loads the same `EfficientNet_B2_Weights.DEFAULT` pretrained
backbone **once** on CPU, with `AdaptiveAvgPool2d(1)`. Both sequences
pass through the same 224x224 RGB resizing, ImageNet normalization and
`encode_sequence` routine used to create Stage 3 training caches.

The report contains mean/min per-frame cosine similarity, relative L2
distance, mean absolute feature difference and extraction provenance.
**It does not invoke the BiLSTM classifier or open test labels.**

If the 030 experiment succeeds, run the same command with video IDs 033
and 049 and **different output paths**. Loading the backbone and encoding
60 frames may be slow on an 8GB CPU laptop; use batch size 1.

## Interpretation

- Cosine similarity near 1 means feature *directions* are similar; this
  does not necessarily imply identical feature magnitudes or identical
  BiLSTM predictions.
- Relative L2 compares feature changes with the original feature norm.
- Neither metric implies real-vs-fake accuracy or can establish why the
  separate Celeb-DF Real video was predicted Fake.
- Pixel parity and feature parity are both diagnostics on **existing
  training examples**. They do not revise the locked FF++ evaluation,
  prove cross-dataset generalization, or authorize threshold tuning.
- When all audits finish, independently review the proposed inference
  configuration, document historical uncertainty, and retain safeguards.
