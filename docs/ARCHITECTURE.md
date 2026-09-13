# DeepVision AI — System Architecture

> Modular, Explainable Deepfake Detection Architecture using Vision Transformers, Temporal Pooling, and Grad-CAM Interpretability.

---

## 1. System Overview & Data Flow

```mermaid
flowchart TD
    A[Raw Video Dataset] --> B[Dataset Indexer<br/>src/datasets/tf_dataset_builder.py]
    B --> C[outputs/dataset_index.csv]
    C --> D[Split Generator<br/>src/datasets/split_generator.py]
    D --> E[outputs/splits/train.csv, val.csv, test.csv]
    E --> F[Sequence Dataset Loader<br/>src/datasets/sequence_dataset.py]
    F --> G[Balanced Sampler<br/>src/datasets/balanced_sampler.py]
    G --> H[DataLoader Factory<br/>src/datasets/dataloader.py]
    H --> I[Model Factory<br/>src/models/model_factory.py]
    I --> J{Model Choice}
    J -->|vit_temporal_pooling| K[ViT-B/16 + Temporal Mean Pooling]
    J -->|efficientnet_bilstm| L[EfficientNet-B2 + BiLSTM]
    K --> M[Trainer Engine<br/>src/training/trainer.py]
    L --> M
    M --> N[Outputs & Checkpoints<br/>outputs/runs/run_*/best_model.pth]
    N --> O[Evaluation Engine<br/>evaluate.py]
    N --> P[Inference Engine<br/>predict.py]
    N --> Q[Grad-CAM Interpreter<br/>src/explainability/gradcam.py]
```

---

## 2. Core Model Architecture: ViT-B/16 + Temporal Mean Pooling

```mermaid
sequenceDiagram
    autonumber
    participant Input as Input Video Tensor (B, T, C, H, W)
    participant Reshape as Reshape (B*T, C, H, W)
    participant ViT as ViT-B/16 Backbone
    participant Restore as Restore Temporal Dim (B, T, 768)
    participant Pool as Temporal Mean Pooling (B, 768)
    participant Head as Classifier (Linear -> ReLU -> Dropout -> Linear)
    participant Output as Prediction Logits (B, 2)

    Input->>Reshape: Flatten Batch and Time
    Reshape->>ViT: Spatial Feature Extraction
    ViT->>Restore: Extract 768-dim Frame Embeddings
    Restore->>Pool: Aggregate Embeddings Over Time
    Pool->>Head: Classify Video Representation
    Head->>Output: Logits [Real, Fake]
```

---

## 3. Alternate Model Architecture: EfficientNet-B2 + BiLSTM

```mermaid
sequenceDiagram
    autonumber
    participant Input as Input Video Tensor (B, T, C, H, W)
    participant CNN as EfficientNet-B2 Features + AvgPool
    participant Sequence as Temporal Reshape (B, T, 1408)
    participant LSTM as Bidirectional LSTM (hidden_dim=256)
    participant Last as Extract Final Timestep (B, 512)
    participant Head as Linear Classifier Head
    participant Output as Prediction Logits (B, 2)

    Input->>CNN: Extract Spatial Features per Frame
    CNN->>Sequence: 1408-dim Feature Vectors
    Sequence->>LSTM: Temporal Sequence Learning
    LSTM->>Last: BiLSTM State Vector
    Last->>Head: Fully Connected Layer
    Head->>Output: Logits [Real, Fake]
```

---

## 4. Source Code Component Mapping

- **Datasets Layer (`src/datasets/`)**: Handles video indexing, stratified splitting, sequence loading (`.npy`), balanced sampling, and DataLoader creation.
- **Models Layer (`src/models/`)**: Abstract `BaseModel`, `ViTTemporalPooling`, `EfficientNetBiLSTM`, and `ModelFactory`.
- **Training Layer (`src/training/`)**: Handles training/validation loops, early stopping, optimizer/scheduler creation, checkpoint management, state tracking, and logging.
- **Evaluation Layer (`evaluate.py`)**: Metrics calculator (Accuracy, Precision, Recall, F1, ROC-AUC), confusion matrix generator, classification report exporter, and prediction CSV serializer.
- **Explainability Layer (`src/explainability/`)**: Grad-CAM visual heatmap generator and overlay renderer.
- **Inference Layer (`predict.py`, `src/inference/`)**: Multi-clip temporal predictor for standalone single video analysis.