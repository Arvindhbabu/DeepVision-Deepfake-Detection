# DeepVision AI — Single Video Inference Specification

## 1. Overview

The inference system (`predict.py`, `src/inference/predictor.py`) processes input video files by extracting multiple temporal frame clips, evaluating per-clip deepfake probabilities, and computing an aggregated final prediction.

---

## 2. Command Execution

```bash
python predict.py \
    --input sample_video.mp4 \
    --checkpoint outputs/runs/run_20260913_120000/best_model.pth \
    --config configs/default.yaml \
    --clips 5
```

---

## 3. Inference Logic

1. **Video Decoding**: Reads input video using OpenCV and resizes frames to target `image_size` (e.g. 224x224).
2. **Multi-Clip Sampling**: Samples `num_clips` random temporal sequence clips across video duration.
3. **Model Inference**: Passes sequence clips through `DeepfakePredictor` and computes softmax probabilities for the **DEEPFAKE** class.
4. **Ensemble Aggregation**: Computes average probability across clips and assigns final binary label (`REAL` vs `DEEPFAKE`) with confidence percentage score.
