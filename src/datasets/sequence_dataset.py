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
        sequence_length: int = 26,
    ):
        self.csv_path = Path(csv_file)
        self.sequence_root = Path(sequence_root)
        self.image_size = image_size
        self.synthetic_if_missing = synthetic_if_missing
        self.sequence_length = sequence_length

        if self.csv_path.exists():
            self.data = pd.read_csv(self.csv_path)
        else:
            self.data = pd.DataFrame(columns=["dataset", "video_name", "label"])

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

        video_name = row["video_name"]
        npy_path = self.sequence_root / folder / f"{video_name}.npy"

        if not npy_path.exists():
            if self.synthetic_if_missing:
                # Return dummy sequence (T, H, W, C)
                return np.random.randint(
                    0, 256, (self.sequence_length, self.image_size, self.image_size, 3), dtype=np.uint8
                )
            raise FileNotFoundError(f"Sequence file not found: {npy_path}")

        try:
            sequence = np.load(npy_path)
        except Exception as e:
            if self.synthetic_if_missing:
                return np.random.randint(
                    0, 256, (self.sequence_length, self.image_size, self.image_size, 3), dtype=np.uint8
                )
            raise RuntimeError(f"Failed loading sequence file: {npy_path}") from e

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

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "sequence": tensor_seq,
            "label": label,
            "dataset": str(row.get("dataset", "Unknown")),
            "video_name": str(row.get("video_name", f"video_{idx}")),
        }