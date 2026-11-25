# Stage 2 — Stabilization

This branch preserves the two existing architectures and focuses on safe input handling.

- MTCNN face crops use `post_process=False` and explicit RGB uint8 conversion.
  Existing previously-preprocessed crops are **not automatically repaired**;
  review/rebuild those artifacts from legitimate source videos if affected.
- The legacy row-level split generator refuses unsafe splits. Use
  `python scripts/validate_dataset.py --help` for verified, actor-disjoint splits.
- Dataset sequence loading checks shapes, finite values and zero-filled placeholders
  (outside explicit synthetic smoke testing), with safe NumPy file loading.
- Relative sequence paths resolve against the split CSV, its parent dataset root,
  and the configured sequence root.
- Pretrained models receive ImageNet channel normalization in training loaders
  and raw-video inference. Checkpoints must use matching preprocessing metadata.
- Raw-video inference samples deterministically with bounded clip buffering.
  It requires a usable frame count and seeks into the video.

## CPU-focused checks

```powershell
python -m unittest tests.test_split_generator tests.test_phase4_dataset
python -m unittest tests.test_stage2_stabilization
python -m unittest discover tests
```

These checks are expected to run on a correctly installed environment, but
the GitHub code edits alone do not constitute a successful test run.
No training, Kaggle uploads or model accuracy claims are part of Stage 2.
