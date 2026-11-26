# DeepVision reliability program — before claiming it works on real videos

## Verified baseline
- A pretrained EfficientNet-B2 frozen-feature + BiLSTM classifier was trained on
  **96** verified FF++ sequences and selected on **20** validation sequences.
- Its original locked FF++ test accuracy was **17/22 (77.27%)**, not a reliable
  estimate for arbitrary real-world footage.
- Stage 6 reconstructed FF++ originals 030/033/049 with ~0.994–0.996 mean
  EfficientNet cosine similarity. This is not perfect historical parity.
- A Celeb-DF Real video and a privately recorded authentic family video have
  been called Fake by the app. **Neither is proof of manipulation.**
- Training a larger model on the same 96 samples or tweaking the score threshold
  against these known examples does not solve generalization.

## Step 1: run an independent external-video audit

Use new, permissioned videos, not the already-inspected wedding clip or
Celeb-DF example to tune any choices. Create a CSV named
`data/local_evaluation/external_manifest.csv` **locally**:

```csv
video_path,label,group_id
videos/real_01.mp4,Real,source_A
videos/fake_01.mp4,Fake,source_B
```

Paths are relative to the manifest's parent directory. `group_id`
identifies the source identity or original video so manipulations of the
same subject can be recognized. These example paths are placeholders;
provide actual files and reliable ground-truth labels. Keep identities
and source families separated when later creating train/validation/test
splits. Do not upload personal videos or metadata to the public repo.

Install existing model dependencies and run:

```powershell
python -m unittest tests.test_external_video_evaluation tests.test_streamlit_app
python scripts/evaluate_external_videos.py --manifest "data/local_evaluation/external_manifest.csv" --face-size 260 --frame-batch-size 1 --output "outputs/reports/external_evaluation_v1.json"
```

The evaluation:
- loads pretrained EfficientNet-B2 **once** for all videos (CPU friendly);
- uses the frozen BiLSTM and current 0.5 threshold;
- provides a confusion matrix, real false-positive rate, fake recall,
  balanced accuracy when both classes exist, and per-video processing errors;
- refuses to overwrite existing evaluation reports; does not open the original
  22-video locked test split or change checkpoint weights;
- keeps local video paths out of the JSON report, but stores `group_id`
  values (avoid real names).

Predeclare which videos are held out *before looking at their predictions*.
Include diverse sources, recording eras, codecs, lighting and manipulation
techniques. Measure metrics by source and subgroup if feasible. Do not
declare an accuracy target achieved from the two previously observed failures.

## Step 2: develop a new trained model, separate from the old one

A credible improvement requires:
1. Additional legitimately sourced, correctly labeled **Real and Fake**
   training videos across multiple domains (FF++, Celeb-DF or other
   appropriately licensed datasets). Respect dataset terms and privacy.
2. A manifest with **original-video / actor / identity group IDs**, ensuring
   no original/derived video or subject identity is in multiple splits.
3. New **train, validation, and one untouched external test** partitions;
   preserve the original FF++ locked-test report as a historical baseline.
4. Compute features using identical video preprocessing at training and
   inference. Track MTCNN, torchvision, OpenCV and model-weight versions.
5. Try a simple pooled-feature linear model as a strong baseline and an
   appropriately regularized temporal model. Use validation only for model
   selection and calibration.
6. Test a **single frozen chosen checkpoint** on the untouched new test
   partition once, reporting accuracy, real false-positive rate, fake
   recall, precision, ROC-AUC/PR-AUC when applicable, confidence intervals,
   and performance stratified by domain.
7. Evaluate abstention/manual-review policies only after independent
   validation. Never turn a high fake score into an authentication claim.

This Stage 7 PR is **measurement and honest UI safeguards**, not a
demonstrated increase in accuracy. Do not promote to trusted forensic use.
