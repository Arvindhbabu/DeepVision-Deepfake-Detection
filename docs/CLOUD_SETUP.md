# DeepVision AI — Cloud Environment Setup Guide

> **Target Environments:** Kaggle Notebooks (Free GPU T4 / P100), Google Colab (Free GPU T4), and Local CPU/GPU Development.

---

## 1. Overview & Cloud Portability Design

DeepVision AI is engineered for seamless migration between local development (Windows / Linux CPU) and remote cloud GPU notebook environments (Kaggle / Colab) without modifying repository code.

### Core Portability Principles:
- **No Hardcoded Absolute Paths:** All paths are driven by configuration YAML files and environment variable overrides.
- **Read-Only Input Compatibility:** Cloud inputs (e.g., `/kaggle/input/...` or Google Drive) are treated strictly as read-only.
- **Configurable Writable Outputs:** Checkpoints, metrics, and logs are directed to designated writable persistent directories.
- **Device-Agnostic Execution:** Automatically adapts to CUDA GPUs or falls back gracefully to CPU.

---

## 2. Environment Variables Reference

| Variable | Default Value | Cloud Purpose | Example Setting |
| :--- | :--- | :--- | :--- |
| `DEEPVISION_CONFIG` | `configs/default.yaml` | Overrides the active configuration YAML. | `configs/smoke_test.yaml` |
| `DEEPVISION_DATA_ROOT` | `data/` | Remaps dataset root to cloud input mount. | `/kaggle/input/deepfake-sequences` or `/content/drive/MyDrive/DeepVisionAI/data` |
| `DEEPVISION_OUTPUT_DIR` | `outputs/` | Remaps generated artifacts to writable storage. | `/kaggle/working/outputs` or `/content/drive/MyDrive/DeepVisionAI/outputs` |

---

## 3. Kaggle Notebook Setup Guide

Kaggle provides free access to NVIDIA T4 (x2) or P100 GPUs with a 30-hour weekly quota.

### Step 1: Session Initialization
1. In your Kaggle Notebook settings (right sidebar):
   - **Accelerator:** GPU T4 (or P100)
   - **Persistence:** None (or Variables & Files)
   - **Internet:** Enabled (required for cloning and dependency verification)

### Step 2: Clone Repository or Upload Source
```bash
# Clone project repository
!git clone https://github.com/<your-username>/deepfake-detection.git /kaggle/working/deepfake-detection
%cd /kaggle/working/deepfake-detection
```

### Step 3: Attach Preprocessed Dataset
1. Click **+ Add Data** in the top-right corner.
2. Select or search for your uploaded dataset containing the preprocessed `.npy` sequences (e.g. `deepfake-sequences`).
3. Note the mounted dataset path (typically `/kaggle/input/deepfake-sequences`).

### Step 4: Configure Environment Variables
```python
import os

# Remap data root to Kaggle read-only input
os.environ["DEEPVISION_DATA_ROOT"] = "/kaggle/input/deepfake-sequences"

# Direct all training checkpoints and logs to writable storage
os.environ["DEEPVISION_OUTPUT_DIR"] = "/kaggle/working/outputs"
```

### Step 5: Verify Environment & Hardware
```bash
!python scripts/check_environment.py
```

### Step 6: Run Cloud Smoke Test
```bash
!python scripts/smoke_test_cloud.py --config configs/smoke_test.yaml
```

### Step 7: Launch Training
```bash
!python train.py --config configs/default.yaml
```

---

## 4. Google Colab Setup Guide

Google Colab provides free access to NVIDIA T4 GPUs with session timeouts.

### Step 1: Runtime Configuration
1. Navigate to **Runtime** > **Change runtime type**.
2. Select **T4 GPU** under Hardware accelerator.

### Step 2: Mount Google Drive (Recommended for Checkpoint Safety)
Google Colab local storage (`/content`) is ephemeral and wiped upon session disconnect. Mounting Google Drive ensures trained weights and evaluation logs survive session drops.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Clone Repository
```bash
%cd /content
!git clone https://github.com/<your-username>/deepfake-detection.git
%cd /content/deepfake-detection
```

### Step 4: Configure Environment Variables
```python
import os

# Data path (e.g. extracted on local Colab disk or stored in Drive)
os.environ["DEEPVISION_DATA_ROOT"] = "/content/drive/MyDrive/DeepVisionAI/data"

# Save checkpoints directly to Drive so disconnects do not lose progress
os.environ["DEEPVISION_OUTPUT_DIR"] = "/content/drive/MyDrive/DeepVisionAI/outputs"
```

### Step 5: Verify Environment & Smoke Test
```bash
!python scripts/check_environment.py
!python scripts/smoke_test_cloud.py
```

---

## 5. Storage & Persistence Considerations

### Kaggle Storage Model:
- `/kaggle/input/`: Strictly **read-only**. Never attempt to write logs or checkpoints here.
- `/kaggle/working/`: **Writable**, but ephemeral per notebook execution unless explicitly saved as output artifacts.
- **Preservation Tip:** At the end of training, compress your experiment outputs for easy downloading:
  ```bash
  !tar -czvf /kaggle/working/experiment_outputs.tar.gz -C /kaggle/working/outputs .
  ```

### Google Colab Storage Model:
- `/content/`: **Ephemeral VM storage**. Fast I/O, but completely lost when kernel restarts.
- `/content/drive/`: **Persistent cloud storage**. Slightly higher latency for millions of small files, but guarantees that checkpoints (`best_model.pth`, `last_model.pth`) and metrics are safely retained.

---

## 6. Diagnostic & Smoke Test Tools

### Environment Diagnostic Utility:
```bash
# Terminal output with PASS, WARNING, and FAIL badges
python scripts/check_environment.py

# Machine-readable JSON output (ideal for automation or pipeline checks)
python scripts/check_environment.py --json
```

### Lightweight Smoke Test:
```bash
# Smoke test both ViT and EfficientNet-B2 architectures
python scripts/smoke_test_cloud.py

# Test a specific architecture
python scripts/smoke_test_cloud.py --model vit
python scripts/smoke_test_cloud.py --model efficientnet
```

---

## 7. Troubleshooting & Common Cloud Errors

### 1. CUDA Out of Memory (OOM)
- **Symptom:** `torch.cuda.OutOfMemoryError: CUDA out of memory.`
- **Remedy:** Reduce `dataloader.batch_size` in the YAML configuration (e.g., from 4 to 2 or 1) and increase `training.gradient_accumulation_steps`.

### 2. Session Disconnection / Timeout
- **Symptom:** Colab or Kaggle session terminates unexpectedly during training.
- **Remedy:** Always resume from the latest saved checkpoint using:
  ```bash
  python train.py --config configs/default.yaml --resume outputs/runs/<latest_run>/last_model.pth
  ```

### 3. Matplotlib Cache Directory Warning
- **Symptom:** `UserWarning: /root/.matplotlib is not a writable directory`
- **Remedy:** DeepVision AI scripts automatically set `MPLCONFIGDIR=/tmp` if not already set in the environment.

### 4. Input Path Not Found
- **Symptom:** `FileNotFoundError: Sequence file not found`
- **Remedy:** Run `python scripts/check_environment.py` and check Section 5 ("Configuration & Paths") to ensure `DEEPVISION_DATA_ROOT` matches the actual mounted directory structure.

