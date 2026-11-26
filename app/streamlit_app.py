"""DeepVision local Streamlit interface (experimental; CPU-only).

Run: streamlit run app/streamlit_app.py
All model execution stays local. Uploaded videos are temporary and deleted.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from scripts.predict_cached_bilstm import classify_sequence, mp4_to_sequence

DEFAULT_CHECKPOINT = ROOT / "outputs/experiments/cached_bilstm_full_v1/best_bilstm.pt"
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def check_upload(filename: str, size: int, max_size: int = MAX_VIDEO_BYTES) -> None:
    """Check extension and size before writing a local temporary file."""
    if Path(filename).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise ValueError("Upload an MP4, MOV, AVI or MKV video.")
    if size <= 0 or size > max_size:
        raise ValueError(f"Video must be between 1 byte and {max_size // 1024**2} MiB.")


def run_prediction(upload, checkpoint: Path, face_size: int) -> dict:
    """Process one uploaded video, deleting its temporary copy on exit."""
    check_upload(upload.name, upload.size)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Trained BiLSTM checkpoint not found on this computer. "
            f"Expected: {checkpoint}"
        )
    suffix = Path(upload.name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="deepvision_") as temp_dir:
        path = Path(temp_dir) / ("upload" + suffix)
        upload.seek(0)
        with path.open("wb") as output:
            while True:
                data = upload.read(1024 * 1024)
                if not data:
                    break
                output.write(data)
        sequence, preprocessing = mp4_to_sequence(
            path, face_size=face_size, max_sampled_frames=300
        )
        prediction = classify_sequence(
            sequence, checkpoint, frame_batch_size=1
        )
    return {
        **prediction,
        "preprocessing": {**preprocessing, "input_type": "raw_video_experimental"},
    }


def main() -> None:
    st.set_page_config(
        page_title="DeepVision AI | Video Analysis",
        page_icon="🔎",
        layout="centered",
    )
    st.title("DeepVision AI")
    st.caption("Research prototype · EfficientNet-B2 + BiLSTM · local CPU inference")
    st.info(
        "Experimental deepfake screening only. Scores are uncalibrated, "
        "may be wrong, and are not forensic proof. Cross-dataset accuracy "
        "has not been established."
    )
    uploaded = st.file_uploader(
        "Select a short video containing a visible face",
        type=["mp4", "mov", "avi", "mkv"],
        help="Maximum file size for this app: 100 MiB. Video processed locally.",
    )
    face_size = st.selectbox(
        "Face crop size",
        [260, 224],
        index=0,
        help="260 matches stored FF++ crop dimensions, but exact historical parity is unproven.",
    )
    if uploaded is not None:
        st.video(uploaded)
        st.caption(f"File: {uploaded.name} · {uploaded.size / (1024**2):.1f} MiB")
    if not st.button("Analyze video", type="primary", disabled=uploaded is None):
        return
    if uploaded is None:
        return
    try:
        with st.spinner("Extracting faces and running local CPU inference..."):
            result = run_prediction(uploaded, DEFAULT_CHECKPOINT, face_size)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        st.error(f"Unable to analyze video: {exc}")
        return
    label = result["label"]
    st.subheader(f"Model prediction: {label}")
    st.metric("Fake class softmax score", f"{result['fake_score'] * 100:.1f}%")
    st.progress(float(result["fake_score"]))
    st.warning(
        "This number is a model score, not the probability the video is fake. "
        "A Real/Fake prediction alone cannot establish authenticity."
    )
    st.subheader("Processing details")
    st.json(result["preprocessing"])
    if result["preprocessing"].get("truncated"):
        st.warning("Video was truncated to the first 300 sampled frames (~60 seconds at 5 FPS).")
    st.caption("No model retraining, threshold adjustment or locked-test evaluation is performed.")


if __name__ == "__main__":
    main()
