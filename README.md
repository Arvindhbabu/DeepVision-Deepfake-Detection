<div align="center">

<<<<<<< HEAD
> An Explainable Deepfake Detection Framework leveraging Vision Transformers, Temporal Pooling, and Grad-CAM Interpretability.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-blueviolet.svg)](#)
[![Deep Learning](https://img.shields.io/badge/Category-Deep%20Learning-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](#)

---

## 1. Overview

**DeepVision AI** is a modular deep learning repository engineered for detecting AI-manipulated facial videos (deepfakes). The framework extracts facial frame sequences from input videos, feeds them through a Vision Transformer (ViT-B/16) spatial feature extractor, aggregates frame embeddings using temporal mean pooling, and predicts whether the content is authentic (**REAL**) or manipulated (**DEEPFAKE**). Visual explainability is integrated via Grad-CAM to highlight facial regions influencing classification decisions.

---

## 2. Problem Statement

With the rapid progression of generative adversarial networks (GANs) and diffusion models, synthetic facial manipulation has achieved high realism. Automated deepfake detection requires spatio-temporal reasoning to identify subtle artifacts across successive video frames while providing interpretable visual explanations to support trustworthy AI deployment.

---

## 3. Key Features

- **Transformer-Based Feature Extraction**: Employs ViT-B/16 pretrained backbones for rich spatial representation of facial artifacts.
- **Temporal Frame Pooling**: Aggregates per-frame embeddings across temporal sequences using parameter-efficient mean/max pooling.
- **Hybrid Architecture Support**: Includes an alternate EfficientNet-B2 + BiLSTM model for spatial-temporal comparative baselines.
- **Grad-CAM Explainability**: Visualizes spatial attention heatmaps overlaid on facial frames to explain model predictions.
- **Reproducible Pipeline**: Full seed initialization, config-driven execution (YAML), structured logging, and automated checkpoint tracking.
- **Extensible Test Suite**: Includes standard unit tests and CPU-compatible integration smoke tests.

---

## 4. System Architecture

```
                                 DeepVision AI

                             Raw Video Dataset
                                    │
                                    ▼
                         Dataset Index Generator
                       (src/datasets/tf_dataset_builder.py)
                                    │
                                    ▼
                         outputs/dataset_index.csv
                                    │
                                    ▼
                       Dataset Split Generator
                       (src/datasets/split_generator.py)
                                    │
                                    ▼
                         outputs/splits/{train,val,test}.csv
                                    │
                                    ▼
                         Sequence Dataset Loader
                       (src/datasets/sequence_dataset.py)
                                    │
                                    ▼
                          Balanced Data Sampler
                       (src/datasets/balanced_sampler.py)
                                    │
                                    ▼
                         PyTorch DataLoader Factory
                       (src/datasets/dataloader.py)
                                    │
                                    ▼
                              Model Factory
                        (src/models/model_factory.py)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
  ViT + Temporal Pooling                         EfficientNet + BiLSTM
(src/models/vit_temporal_pooling.py)           (src/models/efficientnet_bilstm.py)
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                             Training Engine
                         (src/training/trainer.py)
                                    │
                                    ▼
                           Model Checkpoints
                       (outputs/runs/run_*/best_model.pth)
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
    Evaluation Engine                                Inference Engine
       (evaluate.py)                                   (predict.py)
           │                                                 │
           ▼                                                 ▼
Metrics & ROC-AUC Reports                           Grad-CAM Heatmaps
(outputs/evaluation/)                           (src/explainability/gradcam.py)
=======
# 🎭 DeepVision AI
### AI-Powered Deepfake Detection using Vision Transformers and Temporal Learning

<p align="center">
  <img src="https://i.ibb.co/7dtWBspY/logo.png">
</p>

<p align="center">
  <strong>Detecting AI-generated manipulated videos using Vision Transformers, Temporal Pooling, and Deep Learning.</strong>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![Torchvision](https://img.shields.io/badge/Torchvision-Latest-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)
![Research](https://img.shields.io/badge/Research-Final%20Year%20Project-blueviolet)

</p>

</div>

## 🚀 Overview

DeepVision AI is an advanced deepfake detection framework that leverages **Vision Transformers (ViT)** and **Temporal Feature Pooling** to distinguish authentic videos from AI-generated manipulations.

Unlike conventional CNN-based detectors that analyze individual frames independently, DeepVision AI learns both **spatial** and **temporal** inconsistencies across video sequences, making it more robust against modern deepfake generation techniques.

The project is designed for:

- 🎓 Academic Research
- 📄 IEEE Publications
- 💼 AI/ML Portfolio
- 🧠 Computer Vision Research
- 🔒 Media Authenticity Verification

---

# ✨ Features

- ✅ Vision Transformer (ViT-B/16)
- ✅ Temporal Feature Pooling
- ✅ Video Sequence Learning
- ✅ End-to-End Training Pipeline
- ✅ Automatic Checkpoint Saving
- ✅ CUDA GPU Support
- ✅ Mixed Dataset Training
- ✅ Evaluation Metrics
- ✅ Confusion Matrix
- ✅ ROC-AUC Analysis
- ✅ Modular Architecture
- ✅ Production Ready Codebase

---

# 🧠 Architecture

```
               Video Input
                     │
                     ▼
          Frame Extraction
                     │
                     ▼
            Face Detection
                     │
                     ▼
          Face Alignment
                     │
                     ▼
         Frame Sequence Builder
                     │
                     ▼
          Vision Transformer
              (ViT-B/16)
                     │
                     ▼
         Temporal Mean Pooling
                     │
                     ▼
        Fully Connected Layers
                     │
                     ▼
        Real  ←────────→  Fake
>>>>>>> f02c8d26c0d860a7283c4bd14e45b27c7d06ee46
```

---

<<<<<<< HEAD
## 5. Model Architecture

### Canonical Model: `ViTTemporalPooling`

- **Input Tensor**: `(B, T, C, H, W)` where $B$ is batch size, $T$ is sequence length (default 26 frames), $C=3$, $H=224$, $W=224$.
- **Spatial Feature Extractor**: Reshapes tensor to `(B * T, C, H, W)` and passes through ViT-B/16 backbone yielding 768-dimensional per-frame embeddings.
- **Temporal Pooling**: Restores sequence tensor `(B, T, 768)` and applies temporal mean pooling across dimension $T \rightarrow (B, 768)$.
- **Classification Head**:
  - `Linear(768, 256)`
  - `ReLU()`
  - `Dropout(0.3)`
  - `Linear(256, 2)`

```python
import torch
from src.models.vit_temporal_pooling import ViTTemporalPooling

model = ViTTemporalPooling(image_size=224, num_classes=2, pooling="mean")
x = torch.randn(2, 26, 3, 224, 224)  # (Batch=2, Time=26, Channels=3, Height=224, Width=224)
logits = model(x)                     # Output: (2, 2)
```

---

## 6. Dataset

DeepVision AI supports public benchmark datasets:
- **FaceForensics++ (FF++)**: Standard benchmark containing original raw videos and manipulated subsets (Deepfakes, FaceSwap, Face2Face, NeuralTextures).
- **Celeb-DF v2**: High-quality deepfake benchmark dataset with diverse lighting and identity swaps.

---

## 7. Data Processing Pipeline

1. **Frame Extraction**: Sample video frames at target FPS (default 5 FPS) using OpenCV (`src/preprocessing/extract_frames.py`).
2. **Face Detection & Alignment**: Detect and crop face bounding boxes using MTCNN (`src/preprocessing/face_align_mtcnn.py`).
3. **Sequence Construction**: Uniformly sample $T$ frames across video duration and save as compressed `.npy` arrays (`src/preprocessing/make_sequences.py`).
4. **Metadata Indexing & Stratified Splitting**: Generate `dataset_index.csv` and generate balanced 70/15/15 train/val/test splits (`src/datasets/split_generator.py`).

---

## 8. Training

Execute model training using the authoritative configuration file:

```bash
python train.py --config configs/default.yaml
```

For lightweight CPU smoke testing or CI pipeline validation:

```bash
python train.py --config configs/smoke_test.yaml --synthetic-fallback
```

Key features of the training pipeline:
- **Deterministic Seed**: Sets random seeds across `random`, `numpy`, and `torch`.
- **Weighted Class Sampling**: Handles class imbalance between real and synthetic videos.
- **Checkpointing**: Tracks best validation loss and automatically saves `best_model.pth` and `last_model.pth` along with experiment `config.yaml` and `metrics.json`.

---

## 9. Evaluation

Run comprehensive test set evaluation against a trained model checkpoint:

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/runs/run_20260913_120000/best_model.pth --output outputs/evaluation
```

Generated metrics and visual outputs:
- `metrics.json`: Numerical summary (Accuracy, Precision, Recall, F1 Score, ROC-AUC).
- `confusion_matrix.csv` & `confusion_matrix.png`: Confusion matrix table and display plot.
- `roc_curve.png`: Receiver Operating Characteristic curve.
- `predictions.csv`: Per-video ground-truth vs predicted labels and fake probability scores.

---

## 10. Inference

Run single-video multi-clip deepfake inference using the CLI tool:

```bash
python predict.py --video path/to/video.mp4 --checkpoint outputs/runs/run_20260913_120000/best_model.pth --clips 5
```

Example Output:
```text
============================================================
DeepVision AI — Inference Tool
============================================================
Video File  : sample_video.mp4
Checkpoint  : outputs/runs/run_20260913_120000/best_model.pth
Model       : vit_temporal_pooling
Clips       : 5
============================================================

Clip-wise Deepfake Probabilities:
  Clip 01: 94.20%
  Clip 02: 96.10%
  Clip 03: 93.80%
  Clip 04: 95.50%
  Clip 05: 94.90%

============================================================
FINAL PREDICTION : DEEPFAKE
CONFIDENCE       : 94.90%
============================================================
```

---

## 11. Explainability

DeepVision AI integrates **Grad-CAM** (Gradient-weighted Class Activation Mapping) to compute spatial activation maps indicating facial region focus during prediction.

```python
import cv2
import torch
from src.explainability.gradcam import GradCAM
from src.models.vit_temporal_pooling import ViTTemporalPooling

model = ViTTemporalPooling(pretrained=True)
gradcam = GradCAM(model=model, target_layer=model.backbone.conv_proj)

input_tensor = torch.randn(1, 26, 3, 224, 224, requires_grad=True)
heatmap = gradcam.generate_heatmap(input_tensor, target_class=1)

# Overlay heatmap on original facial image
frame_rgb = cv2.imread("face.jpg")[:, :, ::-1]
visualization = GradCAM.overlay_heatmap(frame_rgb, heatmap)
```

---

## 12. Project Structure

```text
deepfake-detection/
├── .github/workflows/ci.yml     # Automated GitHub Actions CI workflow
├── configs/
│   ├── default.yaml            # Default training & model configuration
│   └── smoke_test.yaml         # Lightweight test configuration
├── docs/
│   └── ARCHITECTURE.md         # Detailed technical design document
├── src/
│   ├── datasets/               # Sequence dataset loaders & split generators
│   ├── explainability/         # Grad-CAM heatmap generator
│   ├── inference/              # Multi-clip predictor engine
│   ├── models/                 # ViT and EfficientNet model architectures
│   ├── preprocessing/          # Frame extraction & face alignment
│   ├── training/               # Trainer loop, callbacks & metrics
│   └── utils/                  # Config parser & logging system
├── tests/                      # PyTorch unittest test suite
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
├── predict.py                  # Single-video inference CLI
├── environment.yml             # Conda environment definition
├── requirements.txt            # Python dependencies
├── PROJECT_STATUS.md           # Roadmap and sprint progress tracker
└── README.md                   # Project documentation
=======
# 📂 Project Structure

```text
DeepVision-AI/
│
├── configs/
│
├── data/
│
├── outputs/
│
├── scripts/
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── preprocessing/
│   └── utils/
│
├── tests/
│
├── train.py
├── evaluate.py
├── requirements.txt
├── environment.yml
└── README.md
>>>>>>> f02c8d26c0d860a7283c4bd14e45b27c7d06ee46
```

---

<<<<<<< HEAD
## 13. Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.2.2+ with CUDA support (or CPU mode)

### Environment Setup

#### Option A: Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate deepvision-ai
```

#### Option B: Virtual Environment via Pip
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
=======
# 📊 Dataset

The model is trained on multiple publicly available deepfake datasets.

| Dataset | Purpose |
|---------|----------|
| FaceForensics++ | Training |
| Celeb-DF v2 | Training & Testing |
| DFDC | Future Evaluation |

The datasets contain:

- Original Videos
- DeepFake Videos
- FaceSwap
- Neural Manipulations

---

# ⚙️ Tech Stack

## Programming

- Python

## Deep Learning

- PyTorch
- Torchvision

## Computer Vision

- OpenCV
- facenet-pytorch

## Model

- Vision Transformer (ViT)

## GPU

- NVIDIA CUDA

## Utilities

- NumPy
- Pandas
- tqdm
- YAML

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Arvindhbabu/DeepVision-AI.git

cd DeepVision-AI
```

Create environment

```bash
conda env create -f environment.yml

conda activate deepvision-ai
```

Install dependencies

```bash
>>>>>>> f02c8d26c0d860a7283c4bd14e45b27c7d06ee46
pip install -r requirements.txt
```

---

<<<<<<< HEAD
## 14. Usage

1. **Verify Environment**:
   ```bash
   python -m unittest discover tests
   ```

2. **Index Dataset & Generate Splits**:
   ```bash
   python -m src.datasets.tf_dataset_builder
   python -m src.datasets.split_generator
   ```

3. **Train Model**:
   ```bash
   python train.py --config configs/default.yaml
   ```

4. **Evaluate Model**:
   ```bash
   python evaluate.py --config configs/default.yaml --checkpoint outputs/runs/run_latest/best_model.pth
   ```

5. **Run Single Video Inference**:
   ```bash
   python predict.py --video sample.mp4 --checkpoint outputs/runs/run_latest/best_model.pth
   ```

---

## 15. Configuration

All operational hyperparameters are specified in `configs/default.yaml`:

```yaml
project:
  name: DeepVision AI
  version: 2.0

paths:
  raw_data: data/raw
  processed_data: data/processed
  sequences: data/intermediate/sequences
  outputs: outputs

dataset:
  name: FFPP_CelebDF
  image_size: 224
  sequence_length: 26
  fps: 5

dataloader:
  batch_size: 2
  num_workers: 0
  pin_memory: true
  drop_last: false

training:
  epochs: 50
  learning_rate: 0.00001
  weight_decay: 0.0001
  optimizer: AdamW
  scheduler: CosineAnnealingLR

model:
  name: vit_temporal_pooling
  pretrained: true
  freeze_backbone: false
  num_classes: 2
  pooling: mean
  hidden_dim: 256
  dropout: 0.3
=======
# 🏋️ Training

```bash
python train.py --config configs/default.yaml
>>>>>>> f02c8d26c0d860a7283c4bd14e45b27c7d06ee46
```

---

<<<<<<< HEAD
## 16. Results

> [!NOTE]
> Benchmark experimental results are currently being established through controlled, reproducible experiments across FaceForensics++ and Celeb-DF v2 splits.

| Model Architecture | Test Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **ViT-B/16 + Temporal Pooling** | FF++ / Celeb-DF | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |
| **EfficientNet-B2 + BiLSTM** | FF++ / Celeb-DF | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |

---

## 17. Limitations

- **Hardware Memory Requirements**: Vision Transformers require substantial GPU VRAM for long sequence lengths or large batch sizes.
- **Dataset Bias**: Models trained primarily on FF++ and Celeb-DF may exhibit accuracy drops on unseen commercial deepfake apps without domain adaptation.
- **Compressional Artifacts**: High video compression ratios (e.g. social media re-encoding) can obscure fine-grained facial artifacts.

---

## 18. Future Improvements

- [ ] Implementation of 3D CNN backbones (e.g. ResNet3D, SlowFast) for dynamic motion feature analysis.
- [ ] Cross-dataset evaluation scripts for out-of-distribution domain generalization benchmarking.
- [ ] Model quantization and ONNX runtime export for low-latency edge deployment.
- [ ] Interactive web interface using Streamlit for real-time video upload and heatmap visualization.

---

## 19. Research Direction

Future research will focus on self-supervised pre-training strategies targeting temporal inconsistency along facial boundaries, lip-sync disfluencies, and synthetic eye-blinking patterns.

---

## 20. License

This project is released under the [MIT License](LICENSE).

---

## 21. Author

**Arvindh Babu** — Lead ML Architect & Developer  
*GitHub*: [Arvindhbabu](https://github.com/Arvindhbabu)
=======
# 📈 Evaluation

```bash
python evaluate.py \
--config configs/default.yaml \
--checkpoint outputs/runs/<run>/best_model.pth
```

---

# 📊 Metrics

The framework evaluates using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# 🧩 Workflow

```
Raw Videos
      │
      ▼
Frame Extraction
      │
      ▼
Face Detection
      │
      ▼
Sequence Generation
      │
      ▼
Vision Transformer
      │
      ▼
Temporal Pooling
      │
      ▼
Classification
      │
      ▼
Prediction
```

---

# 🎯 Applications

- Fake News Detection
- Digital Forensics
- Social Media Verification
- Cybersecurity
- Law Enforcement
- Election Integrity
- Identity Protection
- Journalism

---

# 📌 Current Progress

- ✅ Data Pipeline
- ✅ Dataset Builder
- ✅ Sequence Loader
- ✅ Vision Transformer
- ✅ Temporal Pooling
- ✅ GPU Training
- ✅ Evaluation Pipeline
- 🚧 Performance Optimization
- 🚧 IEEE Publication

---

# 📚 Future Improvements

- Spatial Attention
- Temporal Attention
- XAI Explainability
- Grad-CAM Visualization
- Multi-Dataset Benchmarking
- Model Quantization
- Real-Time Webcam Detection
- REST API Deployment
- Mobile Inference
- ONNX Export

---

# 📄 Research Contributions

This project explores:

- Vision Transformers for Deepfake Detection
- Temporal Feature Aggregation
- Video Sequence Representation Learning
- Robust Binary Classification
- Transfer Learning on Large Vision Models

---

# 🤝 Contributing

Contributions are welcome!

Feel free to:

- Fork the repository
- Create feature branches
- Submit Pull Requests
- Report Issues

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Arvindh Babu V

Final Year B.Tech Artificial Intelligence & Data Science

Passionate about

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Generative AI
- MLOps

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

**Made with ❤️ using Python, PyTorch and Vision Transformers**

</div>
>>>>>>> f02c8d26c0d860a7283c4bd14e45b27c7d06ee46
