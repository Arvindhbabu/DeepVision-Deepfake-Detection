"""
DeepVision AI

Sequence Dataset Loader
"""

from pathlib import Path
from typing import Union, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Dataset that loads preprocessed frame sequences stored as .npy files.

    Returns dict with:
        sequence  : (T, C, H, W) normalized tensor [0, 1]
        label     : long scalar tensor (0 for Real, 1 for Fake)
        dataset   : dataset source name string (e.g. 'FFPP' or 'CelebDF')
        video_name: video identifier string
    """

    def __init__(
        self,
        csv_file: Union[str, Path],
        sequence_root: Union[str, Path] = "data/intermediate/sequences",
        image_size: int = 224,
        synthetic_if_missing: bool = False,
        sequence_length: int = 30,
        imagenet_normalize: bool = False,
    ):
        self.csv_path = Path(csv_file)
        self.sequence_root = Path(sequence_root)
        self.image_size = image_size
        self.synthetic_if_missing = synthetic_if_missing
        self.sequence_length = sequence_length
        self.imagenet_normalize = imagenet_normalize

        if self.csv_path.exists():
            self.data = pd.read_csv(self.csv_path)
        elif self.synthetic_if_missing:
            synthetic_rows = [
                {"dataset": "ffpp", "video_name": f"syn_real_{i}", "label": 0} for i in range(10)
            ] + [
                {"dataset": "ffpp", "video_name": f"syn_fake_{i}", "label": 1} for i in range(10)
            ]
            self.data = pd.DataFrame(synthetic_rows)
        else:
            raise FileNotFoundError(f"Dataset split CSV not found: {self.csv_path}")

        if len(self.data) == 0 and self.synthetic_if_missing:
            synthetic_rows = [
                {"dataset": "ffpp", "video_name": f"syn_real_{i}", "label": 0} for i in range(10)
            ] + [
                {"dataset": "ffpp", "video_name": f"syn_fake_{i}", "label": 1} for i in range(10)
            ]
            self.data = pd.DataFrame(synthetic_rows)

    def __len__(self) -> int:
        return len(self.data)

    def _load_sequence(self, row: pd.Series) -> np.ndarray:
        dataset = str(row["dataset"]).lower()
        folder = "ffpp" if "ffpp" in dataset else "celebdf"

        video_name = str(row["video_name"])
        clean_name = video_name[:-4] if video_name.endswith(".npy") else video_name

        # Support direct sequence_path if available in row
        if "sequence_path" in row and pd.notnull(row["sequence_path"]):
            direct_p = Path(str(row["sequence_path"]))
            # Allow paths relative to the split CSV, its dataset parent, or the sequence root.
            candidates = [direct_p, self.csv_path.parent / direct_p,
                          self.csv_path.parent.parent / direct_p,
                          self.sequence_root / direct_p]
            resolved = next((p for p in candidates if p.is_file()), None)
            if resolved is not None:
                npy_path = resolved
            elif (self.sequence_root / folder / f"{clean_name}.npy").exists():
                npy_path = self.sequence_root / folder / f"{clean_name}.npy"
            elif (self.sequence_root / f"{clean_name}.npy").exists():
                npy_path = self.sequence_root / f"{clean_name}.npy"
            else:
                npy_path = direct_p
        else:
            # Support both subfolder hierarchy (root/folder/name.npy) and flat root (root/name.npy)
            npy_path = self.sequence_root / folder / f"{clean_name}.npy"
            if not npy_path.exists() and (self.sequence_root / f"{clean_name}.npy").exists():
                npy_path = self.sequence_root / f"{clean_name}.npy"

        if not npy_path.exists():
            if self.synthetic_if_missing:
                # Return dummy sequence (T, H, W, C)
                return np.random.randint(
                    0, 256, (self.sequence_length, self.image_size, self.image_size, 3), dtype=np.uint8
                )
            raise FileNotFoundError(f"Sequence file not found: {npy_path}")

        try:
            sequence = np.load(npy_path, allow_pickle=False)
        except Exception as e:
            if self.synthetic_if_missing:
                return np.random.randint(
                    0, 256, (self.sequence_length, self.image_size, self.image_size, 3), dtype=np.uint8
                )
            raise RuntimeError(f"Failed loading sequence file: {npy_path}") from e

        if sequence.ndim != 4 or sequence.shape[-1] != 3 or sequence.shape[0] == 0:
            raise ValueError(f"Invalid sequence shape: {sequence.shape} in {npy_path}")
        if not np.issubdtype(sequence.dtype, np.number) or not np.isfinite(sequence).all():
            raise ValueError(f"Invalid sequence values in {npy_path}")
        # Guard against zero-filled placeholder arrays in real training
        if not self.synthetic_if_missing and sequence.max() == 0 and sequence.min() == 0:
            raise ValueError(
                f"Invalid sequence data in '{npy_path}': array is entirely zero-filled placeholder data. "
                "Ensure only validated non-zero sequences are loaded for training."
            )

        return sequence

    def _resize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        resized = []
        for frame in sequence:
            if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
                frame = cv2.resize(
                    frame,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            resized.append(frame)
        return np.stack(resized)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        sequence = self._load_sequence(row)
        sequence = self._resize_sequence(sequence)

        # Normalize [0, 1]
        sequence = sequence.astype(np.float32) / 255.0
        tensor_seq = torch.from_numpy(sequence).permute(0, 3, 1, 2).contiguous()
        if self.imagenet_normalize:
            mean = tensor_seq.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = tensor_seq.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            tensor_seq = (tensor_seq - mean) / std

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "sequence": tensor_seq,
            "label": label,
            "dataset": str(row.get("dataset", "Unknown")),
            "video_name": str(row.get("video_name", f"video_{idx}")),
        }