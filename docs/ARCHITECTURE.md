# DeepVision AI — System Architecture & Technical Design

> A Modular, Explainable Deepfake Detection Framework leveraging Vision Transformers, Temporal Pooling, and Grad-CAM Interpretation.

---

## 1. System Overview

**DeepVision AI** is an open-source research and engineering framework designed to detect AI-generated deepfake videos across benchmark datasets (FaceForensics++ and Celeb-DF v2).

The repository is built around clean software engineering principles:
- **Modular Pipeline Design**: Decoupled preprocessing, dataset loading, model architectures, training, evaluation, explainability, and inference.
- **Config Driven Execution**: Centralized configuration management using YAML.
- **Reproducibility**: Deterministic seed initialization, checkpoint management, and detailed experiment history logging.
- **Explainability**: Integrated Grad-CAM visualization for interpretability of spatial facial feature anomalies.

---

## 2. Overall Pipeline Architecture

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

## 3. Core Model Architecture: ViT-B/16 + Temporal Pooling

The primary canonical model architecture processes video sequences as follows:

```
Input Video Sequence: (B, T, C, H, W)
         │
         ▼
Flatten Batch & Time: (B * T, C, H, W)
         │
         ▼
Vision Transformer Backbone (ViT-B/16 ImageNet-1k)
         │
         ▼
Per-Frame Feature Embedding: (B * T, 768)
         │
         ▼
Restore Temporal Dimension: (B, T, 768)
         │
         ▼
Temporal Mean Pooling: (B, 768)
         │
         ▼
Classification Head: Linear(768 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
         │
         ▼
Output Logits: (B, 2)
```

---

## 4. Alternate Model Architecture: EfficientNet-B2 + BiLSTM

```
Input Video Sequence: (B, T, C, H, W)
         │
         ▼
Flatten Batch & Time: (B * T, C, H, W)
         │
         ▼
EfficientNet-B2 Feature Extractor -> Adaptive Average Pooling: (B * T, 1408)
         │
         ▼
Reshape Temporal Sequence: (B, T, 1408)
         │
         ▼
Bidirectional LSTM (hidden_dim=256): (B, T, 512)
         │
         ▼
Extract Final Timestep: (B, 512)
         │
         ▼
Classification Head: Dropout(0.3) -> Linear(512 -> 2)
         │
         ▼
Output Logits: (B, 2)
```

---

## 5. Repository Structure

```
deepfake-detection/
├── configs/
│   ├── default.yaml            # Authoritative configuration file
│   └── smoke_test.yaml         # Lightweight configuration for smoke testing
├── docs/
│   └── ARCHITECTURE.md         # System design documentation
├── src/
│   ├── datasets/
│   │   ├── balanced_sampler.py # WeightedRandomSampler for class balancing
│   │   ├── dataloader.py       # DataLoader factory function
│   │   ├── sequence_dataset.py # PyTorch Dataset for .npy sequences
│   │   ├── split_generator.py  # Stratified split generator
│   │   └── tf_dataset_builder.py # Metadata dataset indexer
│   ├── explainability/
│   │   └── gradcam.py          # Grad-CAM heatmap overlay visualizer
│   ├── inference/
│   │   └── predictor.py        # Multi-clip video predictor engine
│   ├── models/
│   │   ├── base_model.py       # Abstract base model class
│   │   ├── efficientnet_bilstm.py # Hybrid EfficientNet-BiLSTM model
│   │   ├── model_factory.py    # Dynamic model creation factory
│   │   └── vit_temporal_pooling.py # Vision Transformer with Temporal Mean Pooling
│   ├── preprocessing/
│   │   ├── extract_frames.py   # Video frame extraction tool
│   │   ├── face_align_mtcnn.py # Face detection and alignment via MTCNN
│   │   └── make_sequences.py   # Uniform sequence sampling and .npy exporter
│   ├── training/
│   │   ├── base_trainer.py     # Base trainer interface
│   │   ├── callbacks.py        # Callback manager system
│   │   ├── checkpoint.py       # Checkpoint and artifact manager
│   │   ├── early_stopping.py   # Early stopping handler
│   │   ├── history.py          # Training metrics recorder
│   │   ├── metrics.py          # Batch & epoch metrics computer
│   │   ├── optimizer_factory.py# Optimizer creation factory
│   │   ├── scheduler_factory.py# Learning rate scheduler factory
│   │   ├── state.py            # Training state tracking data class
│   │   └── trainer.py          # Main training execution engine
│   └── utils/
│       ├── config.py           # Configuration YAML parser
│       └── logger.py           # Standardized logger creator
├── tests/                      # PyTorch & system unit test suite
├── train.py                    # Main training execution script
├── evaluate.py                 # Evaluation & metrics generation script
├── predict.py                  # Single video inference CLI tool
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python dependencies
└── README.md                   # Primary repository portfolio documentation
```

---

## 6. Development & Verification Principles

- **Reproducibility**: All dataset splitting and training initialization support explicit random seeds.
- **Fail-Safe Processing**: Data loading and evaluation handle single-class datasets, missing files, and CPU fallbacks gracefully.
- **Test-Driven Rigor**: Standardized unit tests verify configuration, dataset loading, model factory instantiation, forward passes, metric calculations, checkpointing, and inference.