# DeepVision AI — Dataset Specification & Pipeline

## 1. Supported Benchmark Datasets

DeepVision AI is designed to train and evaluate on standard deepfake detection benchmark datasets:

1. **FaceForensics++ (FF++)**:
   - Contains 1,000 original pristine videos.
   - Contains manipulated subsets: Deepfakes, FaceSwap, Face2Face, NeuralTextures.
   - Default compression level: `c23` (light compression).
2. **Celeb-DF v2**:
   - Contains 590 real celebrity videos and 5,639 synthetic deepfake videos with high visual realism.

---

## 2. Dataset Pipeline Flow

```text
Raw Videos (data/raw/)
       │
       ▼
1. Frame Extraction (src/preprocessing/extract_frames.py)
       │  Outputs JPEG frames per video at 5 FPS
       ▼
2. Face Detection & Alignment (src/preprocessing/face_align_mtcnn.py)
       │  Detects faces using MTCNN, crops and resizes to 224x224
       ▼
3. Sequence Generation (src/preprocessing/make_sequences.py)
       │  Uniformly samples T=26 frames and saves compressed .npy arrays
       ▼
4. Dataset Indexing (src/datasets/tf_dataset_builder.py)
       │  Scans directories and exports outputs/dataset_index.csv
       ▼
5. Stratified Splitting (src/datasets/split_generator.py)
       │  Exports non-overlapping train/val/test CSV splits in outputs/splits/
       ▼
6. PyTorch SequenceDataset (src/datasets/sequence_dataset.py)
       │  Loads .npy arrays, normalizes RGB to [0,1], applies weighted sampling
       ▼
DataLoader Batching (B, T, C, H, W)
```

---

## 3. Data Leakage Prevention

To ensure research validity and eliminate data leakage:
- **Video-Level Splitting**: Train, validation, and test splits are strictly partitioned by video identity. No frames or sequences from the same original video appear across multiple splits.
- **Stratified Partitioning**: The split generator maintains class ratio balance (Real vs Fake) across train (70%), validation (15%), and test (15%) subsets.
