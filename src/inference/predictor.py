"""
DeepVision AI

Inference Predictor System
"""

from pathlib import Path
from typing import Tuple, List, Union
import random
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
        self.seq_len = config["dataset"].get("sequence_length", 26)

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
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from video: {video_path}")

        # If video is shorter than sequence length, pad with last frame
        if len(frames) < self.seq_len:
            pad_count = self.seq_len - len(frames)
            frames = frames + [frames[-1]] * pad_count

        clips = []
        max_start = max(0, len(frames) - self.seq_len)

        for _ in range(num_clips):
            start = random.randint(0, max_start) if max_start > 0 else 0
            clip_frames = frames[start:start + self.seq_len]
            clip_arr = np.stack(clip_frames).astype(np.float32) / 255.0
            clip_tensor = torch.from_numpy(clip_arr).permute(0, 3, 1, 2)  # (T, C, H, W)
            clips.append(clip_tensor)

        return torch.stack(clips)  # (num_clips, T, C, H, W)

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
        clips = self.extract_clips_from_video(video_path, num_clips=num_clips).to(self.device)

        clip_probs = []
        with torch.no_grad():
            for clip in clips:
                clip_batch = clip.unsqueeze(0)  # (1, T, C, H, W)
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
