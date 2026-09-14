import unittest
import torch
from src.utils.config import load_config
from src.models.model_factory import ModelFactory
from src.models.vit_temporal_pooling import ViTTemporalPooling
from src.models.efficientnet_bilstm import EfficientNetBiLSTM


class TestModelFactory(unittest.TestCase):

    def setUp(self):
        self.config_vit = load_config("configs/smoke_test.yaml")
        self.config_effnet = {
            "model": {
                "name": "efficientnet_bilstm",
                "num_classes": 2,
                "hidden_dim": 128,
                "dropout": 0.3,
                "pretrained": False,
                "freeze_backbone": True,
            },
            "dataset": {
                "image_size": 224,
            }
        }

    def test_create_vit_temporal_pooling(self):
        model = ModelFactory.create(self.config_vit)
        self.assertIsInstance(model, ViTTemporalPooling)
        self.assertEqual(model.name, "ViTTemporalPooling")

    def test_create_efficientnet_bilstm(self):
        model = ModelFactory.create(self.config_effnet)
        self.assertIsInstance(model, EfficientNetBiLSTM)
        self.assertEqual(model.name, "EfficientNetBiLSTM")

    def test_invalid_model_raises(self):
        invalid_config = {"model": {"name": "invalid_model_xyz"}}
        with self.assertRaises(ValueError):
            ModelFactory.create(invalid_config)


if __name__ == "__main__":
    unittest.main()