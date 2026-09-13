"""
DeepVision AI

Model Factory
"""

from src.models.vit_temporal_pooling import ViTTemporalPooling
from src.models.efficientnet_bilstm import EfficientNetBiLSTM


class ModelFactory:
    """
    Factory for instantiating deepfake detection models based on configuration.
    """

    @staticmethod
    def create(config: dict):
        """
        Create a model instance based on the configuration dictionary.

        Args:
            config (dict): Configuration options containing 'model' and 'dataset' sections.

        Returns:
            nn.Module: Instantiated PyTorch model.
        """
        model_name = config["model"]["name"].lower()

        if model_name in ("vit_temporal_pooling", "vit"):
            return ViTTemporalPooling(
                image_size=config["dataset"].get("image_size", 224),
                num_classes=config["model"].get("num_classes", 2),
                pretrained=config["model"].get("pretrained", True),
                freeze_backbone=config["model"].get("freeze_backbone", True),
                pooling=config["model"].get("pooling", "mean"),
                hidden_dim=config["model"].get("hidden_dim", 256),
                dropout=config["model"].get("dropout", 0.3),
            )

        elif model_name in ("efficientnet_bilstm", "efficientnet"):
            return EfficientNetBiLSTM(
                num_classes=config["model"].get("num_classes", 2),
                hidden_dim=config["model"].get("hidden_dim", 256),
                dropout=config["model"].get("dropout", 0.3),
                freeze_backbone=config["model"].get("freeze_backbone", True),
            )

        raise ValueError(f"Unsupported model name: '{model_name}'. Options: 'vit_temporal_pooling', 'efficientnet_bilstm'")