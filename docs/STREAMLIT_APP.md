# Stage 7 — Local DeepVision video analysis UI

The experimental Streamlit interface uses the **actual saved** pretrained
EfficientNet-B2 cached-feature BiLSTM checkpoint. This is a local CPU demo,
not a cloud production service or a forensic authenticity guarantee.

## Install and run

From project root on your Windows Python environment:

```powershell
git fetch origin
git switch --track origin/stage7/streamlit-video-app-20261126
python -m pip install -r requirements-app.txt
python -m unittest tests.test_streamlit_app tests.test_cached_inference tests.test_inference_crop_size_cli
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed in the terminal, usually
`http://localhost:8501`. The app requires your previously trained local
checkpoint at `outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt`.

Upload a short video (MP4/MOV/AVI/MKV), select 260 or 224-pixel face crop,
and select **Analyze video**. The app uses CPU-friendly frame batches of 1.
It decodes up to 300 sampled frames and requires at least 30 detected faces.

Temporary uploaded video copies are deleted after a run even when processing
fails. The original browser upload remains in Streamlit's session memory,
so do not upload sensitive personal media to a shared machine or host the
app publicly without separate authentication, access controls, and a
privacy/security review. The app does not store classification output to a
database.

## Interpretation

- Stage 6 compared FF++ originals 030/033/049 to preprocessed sequences
  and found mean cosine similarity around 0.995 for candidate 260px crops.
  This is **not** exact historical parity or accuracy.
- The chosen 260px UI option is a transparent research default. The
  Stage 6 CLI still defaults to 224px for backward compatibility.
- The locked FF++ test result remains **17/22 = 77.27%** on preprocessed
  sequences. This figure must not be presented as raw-video, Celeb-DF,
  or general real-world accuracy.
- An actual Celeb-DF Real video was previously incorrectly labeled Fake.
- The number labeled "Fake class softmax score" is uncalibrated and **not**
  the probability that the video is manipulated.
- This interface does not retrain the model, tune the threshold, modify
  held-out test records, or provide Grad-CAM explanations.

## Limitations and next checks

Run tests and one sample MP4 locally first. Check no-face errors, very short
videos, and upload-size rejection. The application currently runs one local
analysis per button click without a background job queue. Do not deploy it
to a public website without proper upload limits at the server level,
resource isolation, codec hardening and abuse protection.
