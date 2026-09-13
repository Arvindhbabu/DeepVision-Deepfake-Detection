import unittest
import tempfile
import torch
from pathlib import Path
from src.models.model_factory import ModelFactory
from src.inference.predictor import DeepfakePredictor


class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.temp_dir.name) / "test_model.pth"

        self.config = {
            "model": {
                "name": "vit_temporal_pooling",
                "num_classes": 2,
                "pretrained": False,
                "freeze_backbone": True,
                "hidden_dim": 64,
                "dropout": 0.1,
            },
            "dataset": {
                "image_size": 224,
                "sequence_length": 4,
            }
        }

        model = ModelFactory.create(self.config)
        torch.save({"model_state_dict": model.state_dict()}, self.ckpt_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_predict_sequence_tensor(self):
        predictor = DeepfakePredictor(
            config=self.config,
            checkpoint_path=self.ckpt_path,
            device="cpu",
        )

        dummy_seq = torch.randn(1, 4, 3, 224, 224)
        label, confidence = predictor.predict_sequence_tensor(dummy_seq)

        self.assertIn(label, ["REAL", "DEEPFAKE"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 100.0)


if __name__ == "__main__":
    unittest.main()
