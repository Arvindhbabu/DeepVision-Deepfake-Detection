"""
DeepVision AI

EfficientNet-B2 + BiLSTM Model Architecture for Video Deepfake Detection.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.models.base_model import BaseModel


class EfficientNetBiLSTM(BaseModel):
    """
    Hybrid spatial-temporal architecture using EfficientNet-B2 feature extractor
    followed by a Bidirectional LSTM and a classification head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        self.cnn = backbone.features

        if freeze_backbone:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1408  # Feature output dimension for EfficientNet-B2

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        feats = self.cnn(x)
        feats = self.pool(feats).reshape(B, T, -1)

        lstm_out, _ = self.lstm(feats)
        out = lstm_out[:, -1, :]  # Extract last hidden state
        logits = self.classifier(out)

        return logits