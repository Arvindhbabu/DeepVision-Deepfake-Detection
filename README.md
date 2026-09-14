# DeepVision AI

> An explainable deepfake detection framework leveraging Vision Transformers, temporal pooling, and Grad-CAM interpretability.

[![DeepVision AI CI](https://github.com/Arvindhbabu/DeepVision-Deepfake-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Arvindhbabu/DeepVision-Deepfake-Detection/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview

**DeepVision AI** is a modular deep learning framework engineered for detecting AI-manipulated facial videos (deepfakes). The framework extracts facial frame sequences from input videos, processes them through spatial feature extractors (Vision Transformer ViT-B/16 or EfficientNet-B2), aggregates frame embeddings across temporal sequence dimensions, and predicts whether the video is authentic (**REAL**) or manipulated (**DEEPFAKE**). Visual explainability is integrated via Grad-CAM to highlight facial regions influencing model classification decisions.

---

## 2. Problem Statement

With the rapid progression of generative adversarial networks (GANs) and diffusion-based synthetic video tools, deepfake facial manipulation has achieved high visual realism. Automated deepfake detection requires robust spatio-temporal modeling to identify subtle artifacts across successive video frames while providing interpretable visual explanations to support trustworthy AI deployment and digital media forensics.

---

## 3. Key Features

- **Transformer-Based Feature Extraction**: Employs ViT-B/16 backbones for rich spatial representation of facial manipulation artifacts.
- **Temporal Frame Pooling**: Aggregates per-frame embeddings across temporal sequences using parameter-efficient mean/max pooling.
- **Hybrid Architecture Support**: Includes an alternate EfficientNet-B2 + BiLSTM model for spatial-temporal baseline comparisons.
- **Grad-CAM Explainability**: Visualizes spatial attention heatmaps overlaid on facial frames to explain model predictions.
- **Reproducible Pipeline**: Full seed initialization, config-driven execution (YAML), structured logging, and automated checkpoint tracking.
- **Automated Verification**: Complete 32-test unit test suite and CPU-compatible synthetic fallback execution for CI/offline validation.

---

## 4. Why This Project Matters

- **Interpretable Media Forensics**: Beyond binary classification, Grad-CAM attention heatmaps pinpoint spatial facial regions influenced by digital editing.
- **Modularity & Extensibility**: Decoupled dataset indexers, model factories, and trainer loops allow rapid prototyping of new backbones and datasets.
- **Reproducible ML Engineering**: Standardized seed initialization, YAML configuration management, and CPU synthetic fallbacks ensure reliable execution across edge, local, and CI environments.

---

## 5. System Architecture

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

## 6. Supported Models

### Canonical Model: `ViTTemporalPooling`

- **Input Tensor**: `(B, T, C, H, W)` where $B$ is batch size, $T$ is sequence length (default 26 frames), $C=3$, $H=224$, $W=224$.
- **Spatial Feature Extractor**: Reshapes tensor to `(B * T, C, H, W)` and passes through ViT-B/16 backbone yielding 768-dimensional per-frame embeddings.
- **Temporal Pooling**: Restores sequence tensor `(B, T, 768)` and applies temporal mean pooling across dimension $T \rightarrow (B, 768)$.
- **Classification Head**: `Linear(768, 256) -> ReLU() -> Dropout(0.3) -> Linear(256, 2)`

### Alternate Model: `EfficientNetBiLSTM`

- **Spatial Extractor**: EfficientNet-B2 backbone returning 1408-dimensional feature vectors per frame.
- **Temporal Sequence Model**: Bidirectional LSTM (`hidden_dim=256`) processing full frame sequences.

---

## 7. Dataset & Data Pipeline

1. **Frame Extraction**: Sample video frames at target FPS using OpenCV (`src/preprocessing/extract_frames.py`).
2. **Face Alignment**: Detect and crop face bounding boxes using MTCNN (`src/preprocessing/face_align_mtcnn.py`).
3. **Sequence Construction**: Uniformly sample $T$ frames across video duration and save as `.npy` arrays (`src/preprocessing/make_sequences.py`).
4. **Metadata Indexing & Stratified Splitting**: Generate `dataset_index.csv` and generate non-overlapping train/val/test splits without video identity leakage (`src/datasets/split_generator.py`).

---

## 8. Installation

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

## 9. Configuration

All operational parameters are specified in `configs/default.yaml` and `configs/smoke_test.yaml`:

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
  pretrained: false
  freeze_backbone: false
  num_classes: 2
  pooling: mean
  hidden_dim: 256
  dropout: 0.3
```

---

## 10. Training

Execute model training using the authoritative configuration file:

```bash
python train.py --config configs/default.yaml
```

For lightweight CPU smoke testing or CI pipeline validation:

```bash
python train.py --config configs/smoke_test.yaml --synthetic-fallback
```

---

## 11. Evaluation

Run test set evaluation against a trained model checkpoint:

```bash
python evaluate.py --config configs/smoke_test.yaml --checkpoint outputs/runs/run_latest/best_model.pth --synthetic-fallback
```

Outputs generated in `outputs/evaluation/`:
- `metrics.json`: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- `confusion_matrix.png` & `roc_curve.png`: Visual evaluation plots.
- `predictions.csv`: Per-video true vs predicted labels and fake probability scores.

---

## 12. Inference

Run single-video multi-clip deepfake inference using the CLI tool:

```bash
python predict.py --input sample_video.mp4 --checkpoint outputs/runs/run_latest/best_model.pth --clips 5
```

For CLI help and arguments list:

```bash
python predict.py --help
```

---

## 13. Explainability

Generate Grad-CAM heatmaps to visualize spatial regions influencing model classification:

```python
import cv2
import torch
from src.explainability.gradcam import GradCAM
from src.models.vit_temporal_pooling import ViTTemporalPooling

model = ViTTemporalPooling(pretrained=False)
model.eval()

target_layer = model.backbone.conv_proj
gradcam = GradCAM(model=model, target_layer=target_layer)

input_tensor = torch.randn(1, 26, 3, 224, 224, requires_grad=True)
heatmap = gradcam.generate_heatmap(input_tensor, target_class=1)

frame_rgb = cv2.imread("face.jpg")[:, :, ::-1]
visualization = GradCAM.overlay_heatmap(frame_rgb, heatmap)
```

---

## 14. Testing & Verification

### Verified Engineering Validation

The repository currently supports:

- **Automated Unit Tests**: `32/32 tests passing` across 23 test modules covering config, dataset loaders, sampler, models, trainer, evaluator, predictor, and Grad-CAM.
- **CPU Training Smoke Test**: 1-epoch execution on CPU without GPU or network dependencies.
- **CPU Evaluation Smoke Test**: Metric logging, confusion matrix, and ROC plot generation via `evaluate.py`.
- **Inference Pipeline Validation**: Multi-clip sequence tensor inference via `predict.py`.
- **Grad-CAM Explainability Validation**: Spatial attention map shape and layer hook execution verified.
- **Synthetic Fallback Execution**: Complete offline pipeline execution using `--synthetic-fallback`.

To run unit test suite:

```bash
python -m unittest discover tests -v
```

> [!NOTE]
> Synthetic smoke testing metrics validate pipeline integrity, tensor shapes, and control-flow correctness; they do not represent real deepfake detection benchmark performance.

---

## 15. Results / Benchmark Status

> [!NOTE]
> Benchmark experimental results across FaceForensics++ and Celeb-DF v2 splits are currently in progress / not yet established in the repository archive. Real benchmark accuracy, precision, recall, F1, and ROC-AUC metrics must not be inferred from synthetic smoke tests.

| Model Architecture | Benchmark Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
|---|---|---|---|---|---|---|---|
| **ViT-B/16 + Temporal Pooling** | FaceForensics++ / Celeb-DF v2 | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* | Under Evaluation |
| **EfficientNet-B2 + BiLSTM** | FaceForensics++ / Celeb-DF v2 | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* | Under Evaluation |

---

## 16. Limitations

- **Hardware Memory Requirements**: Vision Transformers require GPU VRAM for sequence lengths $>30$ frames at large batch sizes.
- **Compressional Artifacts**: Heavy social media re-encoding can obscure subtle visual facial manipulation cues.

---

## 17. Project Structure

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
├── PROJECT_STATUS.md           # Implementation status and roadmap
└── README.md                   # Primary project documentation
```

---

## 18. Reproducibility

Seed initialization (`set_seed`), configuration locking (YAML), deterministic PyTorch backends, and tracked run artifacts ensure reproducible experiments. Detailed specifications are provided in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## 19. Future Roadmap

- [ ] Streamlit interactive web deployment interface.
- [ ] 3D CNN spatio-temporal backbones (ResNet3D, SlowFast).
- [ ] Model quantization & ONNX runtime export for real-time edge inference.

---

## 20. License

This project is released under the [MIT License](LICENSE).

---

## 21. Author

**Arvindh Babu V** — AI & Data Science Student  
*GitHub*: [Arvindhbabu](https://github.com/Arvindhbabu)  
*Repository*: [DeepVision-Deepfake-Detection](https://github.com/Arvindhbabu/DeepVision-Deepfake-Detection)
