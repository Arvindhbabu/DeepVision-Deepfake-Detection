# DeepVision AI

> An Explainable Deepfake Detection Framework leveraging Vision Transformers, Temporal Pooling, and Grad-CAM Interpretability.

[![DeepVision AI CI](https://github.com/Arvindhbabu/DeepVision-Deepfake-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Arvindhbabu/DeepVision-Deepfake-Detection/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview

**DeepVision AI** is a modular deep learning framework engineered for detecting AI-manipulated facial videos (deepfakes). The framework extracts facial frame sequences from input videos, feeds them through a Vision Transformer (ViT-B/16) spatial feature extractor, aggregates frame embeddings using temporal mean pooling, and predicts whether the content is authentic (**REAL**) or manipulated (**DEEPFAKE**). Visual explainability is integrated via Grad-CAM to highlight facial regions influencing classification decisions.

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
```

---

## 5. Supported Models

### Canonical Model: `ViTTemporalPooling`

- **Input Tensor**: `(B, T, C, H, W)` where $B$ is batch size, $T$ is sequence length (default 26 frames), $C=3$, $H=224$, $W=224$.
- **Spatial Feature Extractor**: Reshapes tensor to `(B * T, C, H, W)` and passes through ViT-B/16 backbone yielding 768-dimensional per-frame embeddings.
- **Temporal Pooling**: Restores sequence tensor `(B, T, 768)` and applies temporal mean pooling across dimension $T \rightarrow (B, 768)$.
- **Classification Head**: `Linear(768, 256) -> ReLU() -> Dropout(0.3) -> Linear(256, 2)`

### Alternate Model: `EfficientNetBiLSTM`

- **Spatial Extractor**: EfficientNet-B2 backbone returning 1408-dimensional feature vectors per frame.
- **Temporal Sequence Model**: Bidirectional LSTM (`hidden_dim=256`) processing full frame sequences.

---

## 6. Dataset & Pipeline

1. **Frame Extraction**: Sample video frames at target FPS using OpenCV (`src/preprocessing/extract_frames.py`).
2. **Face Alignment**: Detect and crop face bounding boxes using MTCNN (`src/preprocessing/face_align_mtcnn.py`).
3. **Sequence Construction**: Uniformly sample $T$ frames across video duration and save as `.npy` arrays (`src/preprocessing/make_sequences.py`).
4. **Metadata Indexing & Stratified Splitting**: Generate `dataset_index.csv` and generate non-overlapping train/val/test splits without video identity leakage (`src/datasets/split_generator.py`).

---

## 7. Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.2.2+ (CPU or CUDA 12.1)

```bash
git clone https://github.com/Arvindhbabu/DeepVision-Deepfake-Detection.git
cd DeepVision-Deepfake-Detection

# Option A: Conda
conda env create -f environment.yml
conda activate deepvision-ai

# Option B: Pip Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 8. Configuration

All operational parameters are specified in `configs/default.yaml`:

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
```

---

## 9. Training

Execute model training using the authoritative configuration file:

```bash
python train.py --config configs/default.yaml
```

For lightweight CPU smoke testing or CI pipeline validation:

```bash
python train.py --config configs/smoke_test.yaml --synthetic-fallback
```

---

## 10. Evaluation

Run comprehensive test set evaluation against a trained model checkpoint:

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/runs/run_latest/best_model.pth --output outputs/evaluation
```

Outputs generated in `outputs/evaluation/`:
- `metrics.json`: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- `confusion_matrix.png` & `roc_curve.png`: Visual evaluation plots.
- `predictions.csv`: Per-video true vs predicted labels and fake probability scores.

---

## 11. Inference

Run single-video multi-clip deepfake inference using the CLI tool:

```bash
python predict.py --input sample_video.mp4 --checkpoint outputs/runs/run_latest/best_model.pth --clips 5
```

---

## 12. Explainability

Generate Grad-CAM heatmaps to visualize spatial regions influencing model classification:

```python
import cv2
import torch
from src.explainability.gradcam import GradCAM
from src.models.vit_temporal_pooling import ViTTemporalPooling

model = ViTTemporalPooling(pretrained=True)
gradcam = GradCAM(model=model, target_layer=model.backbone.conv_proj)

input_tensor = torch.randn(1, 26, 3, 224, 224, requires_grad=True)
heatmap = gradcam.generate_heatmap(input_tensor, target_class=1)

frame_rgb = cv2.imread("face.jpg")[:, :, ::-1]
visualization = GradCAM.overlay_heatmap(frame_rgb, heatmap)
```

---

## 13. Testing

Run the full unit test suite:

```bash
python -m unittest discover tests
```

---

## 14. Project Structure

```text
DeepVision-Deepfake-Detection/
├── .github/workflows/ci.yml     # Automated GitHub Actions CI workflow
├── configs/
│   ├── default.yaml            # Default training & model configuration
│   └── smoke_test.yaml         # Lightweight test configuration
├── docs/                       # Modular technical documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   ├── INFERENCE.md
│   ├── EXPLAINABILITY.md
│   └── REPRODUCIBILITY.md
├── src/
│   ├── datasets/               # Sequence dataset loaders & split generators
│   ├── explainability/         # Grad-CAM heatmap generator
│   ├── inference/              # Multi-clip predictor engine
│   ├── models/                 # ViT and EfficientNet model architectures
│   ├── preprocessing/          # Frame extraction & face alignment
│   ├── training/               # Trainer loop, callbacks & metrics
│   └── utils/                  # Config parser & logging system
├── tests/                      # Standardized PyTorch unit test suite
├── train.py                    # Training CLI entry point
├── evaluate.py                 # Evaluation CLI entry point
├── predict.py                  # Single-video inference CLI
├── environment.yml             # Conda environment definition
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable configuration template
├── PROJECT_STATUS.md           # Roadmap and sprint progress tracker
└── README.md                   # Primary project documentation
```

---

## 15. Results

> [!NOTE]
> Benchmark experimental results are currently being established through controlled, reproducible experiments across FaceForensics++ and Celeb-DF v2 splits.

| Model Architecture | Test Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **ViT-B/16 + Temporal Pooling** | FF++ / Celeb-DF | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |
| **EfficientNet-B2 + BiLSTM** | FF++ / Celeb-DF | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |

---

## 16. Limitations

- **Hardware Memory Requirements**: Vision Transformers require GPU VRAM for sequence lengths $>30$ frames at large batch sizes.
- **Compressional Artifacts**: Heavy social media re-encoding can obscure visual facial manipulation cues.

---

## 17. Reproducibility

Seed initialization (`set_seed`), configuration locking (YAML), deterministic CUDA backends, and tracked run artifacts ensure reproducible experiments. Detailed specifications are provided in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## 18. License

This project is released under the [MIT License](LICENSE).

---

## 19. Author

**Arvindh Babu V** — AI & Data Science Student  
*GitHub*: [Arvindhbabu](https://github.com/Arvindhbabu)  
*Repository*: [DeepVision AI](https://github.com/Arvindhbabu/DeepVision-Deepfake-Detection)
