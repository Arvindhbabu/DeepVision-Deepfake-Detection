# Stage 5: CPU inference with the trained cached-feature BiLSTM

This is a **new predictor**, not the legacy `src/inference/predictor.py`.
The legacy predictor expects a different checkpoint format and directly
passes RGB frame tensors through a configurable model. The trained Stage 3
checkpoint instead expects **30 frozen EfficientNet-B2 feature vectors**.

## Verify without running the pretrained backbone

From a PowerShell project-root terminal:

```powershell
git fetch origin
git switch --track origin/stage5/cached-video-inference-20261125
python -m unittest tests.test_cached_inference
```

## First full inference: a known-aligned sequence file

Use one existing verified `.npy` aligned-face sequence from your local
FF++ data. Specify its actual path, for example:

```powershell
python scripts/predict_cached_bilstm.py --input "data/intermediate/sequences/ffpp/039.npy" --frame-batch-size 1
```

A sequence file must contain **30 uint8 RGB frames** of the same shape as
training (typically 260×260×3 before resizing). The code rescales to
224×224, divides by 255, applies ImageNet normalization, extracts
pretrained EfficientNet-B2 frame features, and uses the **best saved
BiLSTM checkpoint** with the fixed 0.5 threshold.

The CLI outputs **uncalibrated** model scores. A single inference call is
not new validation or independent test performance.

## Experimental raw MP4 path

```powershell
python scripts/predict_cached_bilstm.py --input "path/to/short_video.mp4" --frame-batch-size 1
```

The experimental decoder samples around 5 FPS, JPEG round-trips frames,
detects/crops the primary face with `MTCNN(post_process=False)`, selects
30 uniformly spaced valid face images, then applies the same 224×224
normalization and EfficientNet feature encoder.

**Limitations:** Dataset production used JPEG intermediates and a
configurable MTCNN output size. The exact settings used when the user's
138 verified sequences were originally produced have not yet been proven.
Thus a raw MP4 result **does not inherit the 77.27% held-out test accuracy**.
Raw-video preprocessing is explicitly marked
`approximate_not_audited` in the output. Face failure rejects input
with fewer than 30 successful crops. Processing is capped at 300 sampled
frames by default (~60s at 5 FPS), truncating longer videos and reporting
that fact. Do not claim forensic confidence, calibrated probability, or
deployment readiness. Never use random-weight backbone features.

No labeled test split or checkpoint is changed by this script.
