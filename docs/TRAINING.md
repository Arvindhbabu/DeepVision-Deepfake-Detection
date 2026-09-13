# DeepVision AI — Training Engine Specification

## 1. Overview

The training engine (`train.py`, `src/training/trainer.py`) executes configuration-driven model optimization with deterministic seed control, early stopping, and checkpoint tracking.

---

## 2. Training Execution Commands

### Production Training
```bash
python train.py --config configs/default.yaml
```

### CPU Smoke Testing (CI/Testing Mode)
```bash
python train.py --config configs/smoke_test.yaml --synthetic-fallback
```

### Resuming Training from Checkpoint
```bash
python train.py --config configs/default.yaml --resume outputs/runs/run_20260913_120000/last_model.pth
```

---

## 3. Key Operational Capabilities

1. **Seed Initialization**: Random seeds are explicitly set across Python `random`, `numpy`, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`) for reproducible weight initialization and data sampling.
2. **Device Detection**: Detects CUDA availability automatically and falls back gracefully to CPU.
3. **Class Balancing**: Utilizes `WeightedRandomSampler` to prevent model bias when training on imbalanced dataset splits.
4. **Checkpoint Management**: Saves `best_model.pth` based on validation loss, `last_model.pth` for checkpoint resumption, along with `config.yaml` and `metrics.json` inside unique experiment run folders under `outputs/runs/run_YYYYMMDD_HHMMSS/`.
