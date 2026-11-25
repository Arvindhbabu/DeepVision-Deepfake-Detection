"""Small synthetic test of frame-feature batching; no pretrained weights needed."""
import unittest

import numpy as np
import torch

from scripts.cache_efficientnet_features import encode_sequence


class ConstantBackbone(torch.nn.Module):
    def forward(self, inputs):
        # Shape matches EfficientNet-B2 final feature channels.
        return inputs.mean(dim=1, keepdim=True).expand(-1, 1408, -1, -1)


class TestFeatureCache(unittest.TestCase):
    def test_bounded_batches_preserve_frame_order(self):
        seq = torch.stack([torch.full((3, 8, 8), value) for value in (1., 2., 3.)])
        result = encode_sequence(seq, ConstantBackbone(),
                                 torch.nn.AdaptiveAvgPool2d(1), frame_batch_size=1)
        self.assertEqual(result.shape, (3, 1408))
        np.testing.assert_allclose(result[:, 0], [1., 2., 3.], rtol=1e-6)

    def test_invalid_batch_size_rejected(self):
        with self.assertRaises(ValueError):
            encode_sequence(torch.zeros(2, 3, 8, 8), ConstantBackbone(),
                            torch.nn.AdaptiveAvgPool2d(1), frame_batch_size=0)


if __name__ == "__main__":
    unittest.main()
