import unittest
import torch
from src.models.efficientnet_bilstm import EfficientNetBiLSTM


class TestEfficientNetBiLSTM(unittest.TestCase):

    def setUp(self):
        self.model = EfficientNetBiLSTM(
            num_classes=2,
            hidden_dim=64,
            dropout=0.1,
            pretrained=False,
            freeze_backbone=True,
        )

    def test_construction(self):
        self.assertIsNotNone(self.model)

    def test_forward_pass_shape(self):
        self.model.eval()
        B, T, C, H, W = 2, 3, 3, 224, 224
        x = torch.randn(B, T, C, H, W)
        with torch.no_grad():
            output = self.model(x)
        self.assertEqual(output.shape, (B, 2))


if __name__ == "__main__":
    unittest.main()
