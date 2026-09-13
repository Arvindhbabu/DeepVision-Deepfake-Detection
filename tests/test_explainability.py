import unittest
import torch
import numpy as np
from src.explainability.gradcam import GradCAM
from src.models.vit_temporal_pooling import ViTTemporalPooling


class TestGradCAM(unittest.TestCase):

    def test_gradcam_heatmap_generation(self):
        model = ViTTemporalPooling(
            image_size=224,
            num_classes=2,
            pretrained=False,
            freeze_backbone=False,
        )

        gradcam = GradCAM(model=model, target_layer=model.backbone.conv_proj)

        # Input shape (B=1, T=2, C=3, H=224, W=224)
        x = torch.randn(1, 2, 3, 224, 224, requires_grad=True)
        heatmap = gradcam.generate_heatmap(x, target_class=1)

        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.ndim, 2)


if __name__ == "__main__":
    unittest.main()
