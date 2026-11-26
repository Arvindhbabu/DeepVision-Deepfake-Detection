# Stage 6 — Reproducible preprocessing audit

## Why this is needed

The trained FF++ model obtained **17/22 = 77.27% accuracy** on a
held-out test partition of already aligned and cached face sequences. A
separate, raw Celeb-DF **Real** MP4 was predicted **Fake** with a softmax
fake score about **0.918**. This is a false positive and an out-of-domain
example, **not evidence of a verified Celeb-DF error rate**.

A review of the repository found a **specific provenance discrepancy**:
the verified 138 source sequences were reported as
`(30, 260, 260, 3)`, while the current
`configs/default.yaml` says `image_size: 224` and the current
MTCNN alignment reads that setting from `src/config.py`.
This cannot establish how the historical 260×260 source data was created.

The pipeline, as currently checked into the repository:

1. Decode MP4, keep every `max(1, round(video_fps / 5))` frame, save as JPEG.
2. `MTCNN(keep_all=False,margin=0,post_process=False)`, save face crops as JPEG.
3. Read valid aligned JPEGs, uniformly sample 30 with
   `np.linspace(0,n-1,30).astype(int)` and save uint8 RGB `.npy`.
4. Resize each frame with OpenCV to 224×224; divide by 255 and normalize
   by ImageNet mean/std; extract frozen EfficientNet-B2 features.

The Stage 5 MP4 path attempts the same logic, but the source materials do
not establish historical library versions, MTCNN size, or exact round-trip
parity. It is also capped at 300 sampled frames; the historical dataset
extractor was not, so long videos can yield different 30-frame samples.

## Phase A: a fast, read-only audit

On Windows PowerShell, from the project root:

```powershell
git fetch origin
git switch --track origin/stage6/preprocessing-audit-20261126
python -m unittest tests.test_preprocessing_audit
python scripts/audit_preprocessing_parity.py --sequence "data/intermediate/sequences/ffpp/030.npy" --output "outputs/reports/preprocessing_audit_030.json"
```

This verifies sequence format and reports whether its dimensions match the
current 224-pixel configuration. It **does not modify source arrays or
run the classifier**.

If the original aligned JPEG directory is present, run:

```powershell
python scripts/audit_preprocessing_parity.py --sequence "data/intermediate/sequences/ffpp/030.npy" --aligned-dir "data/intermediate/aligned_faces/ffpp/030"
```

The report distinguishes:

- `exact_match`: historical JPEG-to-sequence reconstruction is byte-identical.
- `pixel_mismatch`: compare mean and maximum absolute pixel error.
- `shape_mismatch`: selected JPEG dimensions differ.
- `unavailable`: the original aligned JPEG directory is not present.
- `not_checked`: no aligned directory was provided.

**Important:** `exact_match` only validates the stored aligned JPEG files
through sequence creation. It cannot prove that original video frames, JPEG
encoding, or face detection settings are identical. Never relabel the MP4
pipeline as production-parity solely from this result.

## What remains for Stage 6

- Find an original FF++ raw MP4 corresponding to an already-verified sequence,
  if provenance is available. Test the raw MP4 versus its aligned intermediate
  images *without using labels or changing any classifier weights*.
- Record the historical 260px alignment settings, MTCNN/facenet-pytorch
  version, and JPEG encoder provenance if recoverable.
- Add full raw-MP4-to-sequence parity checks only if a verifiable original
  video/sequence pair is available.
- Treat the false-positive Celeb-DF observation as diagnostic only.
  **Do not tune the threshold or classifier to make that example pass.**

The FF++ held-out test report and epoch-16 checkpoint remain frozen.
