import unittest
import torch
from src.models.vit_temporal_pooling import ViTTemporalPooling


class TestViTTemporalPooling(unittest.TestCase):

    def setUp(self):
        self.model = ViTTemporalPooling(
            image_size=224,
            num_classes=2,
            pretrained=False,
            freeze_backbone=True,
            pooling="mean",
            hidden_dim=128,
            dropout=0.1,
        )

    def test_construction(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.feature_dim, 768)

    def test_forward_pass_shape(self):
        self.model.eval()
        B, T, C, H, W = 2, 4, 3, 224, 224
        x = torch.randn(B, T, C, H, W)
        with torch.no_grad():
            output = self.model(x)
        self.assertEqual(output.shape, (B, 2))

    def test_max_pooling(self):
        model_max = ViTTemporalPooling(
            image_size=224,
            num_classes=2,
            pretrained=False,
            pooling="max",
            hidden_dim=128,
        )
        model_max.eval()
        x = torch.randn(1, 3, 3, 224, 224)
        with torch.no_grad():
            output = model_max(x)
        self.assertEqual(output.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
