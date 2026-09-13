import unittest
import torch
import torch.nn as nn
from src.utils.config import load_config
from src.models.vit_temporal_pooling import ViTTemporalPooling


class TestTrainingPipeline(unittest.TestCase):

    def test_cpu_forward_backward_pass(self):
        config = load_config("configs/smoke_test.yaml")
        model = ViTTemporalPooling(
            image_size=config["dataset"]["image_size"],
            num_classes=2,
            pretrained=False,
            freeze_backbone=True,
            hidden_dim=64,
        )

        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)

        # Batch shape (B=2, T=3, C=3, H=224, W=224)
        dummy_input = torch.randn(2, 3, 3, 224, 224)
        dummy_target = torch.tensor([0, 1])

        optimizer.zero_grad()
        outputs = model(dummy_input)
        self.assertEqual(outputs.shape, (2, 2))

        loss = criterion(outputs, dummy_target)
        loss.backward()
        optimizer.step()

        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()