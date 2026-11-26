# Stage 6 — Optional 260px raw-MP4 alignment

Three original FF++ training videos were reconstructed using the candidate 260px MTCNN face size and compared with their saved uint8 RGB sequences. Results:

| Video | Pixel MAE (0–255) | Mean frozen EfficientNet cosine | Mean relative L2 |
| --- | ---: | ---: | ---: |
| 030 | 1.65397 | 0.993738 | 0.110479 |
| 033 | 1.30448 | 0.995831 | 0.090866 |
| 049 | 1.58209 | 0.995795 | 0.091309 |

These are **preprocessing diagnostics on existing FF++ examples**, not new
classification accuracy measurements. In all three cases the historical
face-sequence crop dimensions are 260x260, but exact pixel and feature
parity were **not** achieved. The old 224px crop experiment on video 030
had pixel MAE 1.67142 after resizing to 260px, compared with 1.65397 at
native 260px; this small difference is not proof of better classification.

Stage 6 adds an **optional** `--face-size {224,260}` argument to
`scripts/predict_cached_bilstm.py`. The default remains **224** to avoid
silently changing Stage 5 behavior. The MP4 prediction metadata includes
the selected face size. Example:

```powershell
python scripts/predict_cached_bilstm.py --input "data/raw/ffpp_c23/original/030.mp4" --face-size 260 --frame-batch-size 1
```

This is a developer diagnostic and should not be used to tune to the
known label of a specific video. The original checkpoint, locked test
results, threshold 0.5, and stored training data remain unchanged.

Before relying on raw MP4 predictions, independently assess temporal
sampling and preprocessing provenance and measure generalization on a
new, predeclared set of videos. A softmax score is **not** a calibrated
probability. The prior Celeb-DF Real video misclassification is not
resolved by this crop-size comparison.
