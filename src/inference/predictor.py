"""
DeepVision AI

Inference Predictor System
"""

from pathlib import Path
from typing import Tuple, List, Union
import cv2
import numpy as np
import torch
import torch.nn as nn

from src.models.model_factory import ModelFactory


class DeepfakePredictor:
    """
    Predictor class for inferring deepfake probabilities from video files or preprocessed sequence tensors.
    """

    def __init__(
        self,
        config: dict,
        checkpoint_path: Union[str, Path],
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = config
        self.image_size = config["dataset"].get("image_size", 224)
        self.seq_len = config["dataset"].get("sequence_length", 30)
        self.imagenet_normalize = bool(config.get("model", {}).get("pretrained", False))

        self.model = ModelFactory.create(config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_clips_from_video(
        self,
        video_path: Union[str, Path],
        num_clips: int = 5,
    ) -> torch.Tensor:
        """
        Extract multiple sequence clips from a video.

        Args:
            video_path (str/Path): Path to input video.
            num_clips (int): Number of random sequence clips to extract.

        Returns:
            torch.Tensor: Tensor of shape (num_clips, T, C, H, W)
        """
        if num_clips < 1 or self.seq_len < 1:
            raise ValueError("num_clips and sequence_length must be positive")
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError(f"Video has no usable frame count: {path}")
            # Deterministic positions, decode only sampled frames; bounded by clips * seq_len.
            max_start = max(0, count - self.seq_len)
            starts = np.linspace(0, max_start, num_clips, dtype=np.int64)
            clip_tensors = []
            for start in starts:
                frames = []
                for frame_idx in range(int(start), int(start) + self.seq_len):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, count - 1))
                    ok, frame = cap.read()
                    if not ok:
                        if not frames:
                            raise ValueError(f"Unable to decode frame {frame_idx}: {path}")
                        frame = frames[-1]
                        frames.append(frame)
                        continue
                    frame = cv2.cvtColor(
                        cv2.resize(frame, (self.image_size, self.image_size)),
                        cv2.COLOR_BGR2RGB,
                    )
                    frames.append(frame)
                arr = np.stack(frames).astype(np.float32) / 255.0
                clip_tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
                if self.imagenet_normalize:
                    mean = clip_tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
                    std = clip_tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
                    clip_tensor = (clip_tensor - mean) / std
                clip_tensors.append(clip_tensor)
            return torch.stack(clip_tensors)
        finally:
            cap.release()

    def predict_video(
        self,
        video_path: Union[str, Path],
        num_clips: int = 5,
    ) -> Tuple[str, float, List[float]]:
        """
        Predict deepfake status for a video.

        Returns:
            label (str): 'DEEPFAKE' or 'REAL'
            confidence (float): Percentage score (0-100%)
            clip_probabilities (List[float]): Per-clip fake probability scores
        """
        clips = self.extract_clips_from_video(video_path, num_clips=num_clips)

        clip_probs = []
        with torch.no_grad():
            for clip in clips:
                clip_batch = clip.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
                logits = self.model(clip_batch)
                prob = torch.softmax(logits, dim=1)[0, 1]  # Fake class prob
                clip_probs.append(prob.item())

        avg_fake_prob = float(np.mean(clip_probs))
        label = "DEEPFAKE" if avg_fake_prob > 0.5 else "REAL"
        confidence = avg_fake_prob * 100.0 if label == "DEEPFAKE" else (1.0 - avg_fake_prob) * 100.0

        return label, confidence, clip_probs

    def predict_sequence_tensor(self, sequence_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Predict from a single sequence tensor of shape (B, T, C, H, W) or (T, C, H, W).
        """
        if sequence_tensor.ndim == 4:
            sequence_tensor = sequence_tensor.unsqueeze(0)

        sequence_tensor = sequence_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            fake_prob = float(probs[1].item())

        label = "DEEPFAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob * 100.0 if label == "DEEPFAKE" else (1.0 - fake_prob) * 100.0

        return label, confidence
