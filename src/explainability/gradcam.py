"""
DeepVision AI

Grad-CAM Explainability Module for Vision Transformer and CNN Backbones.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM implementation for visualizing attention / feature map activations
    driving deepfake model decisions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap (0.0 to 1.0) for an input tensor.

        Args:
            input_tensor (torch.Tensor): Tensor of shape (B, T, C, H, W) or (B, C, H, W)
            target_class (int, optional): Class index to visualize. Default uses predicted class.

        Returns:
            np.ndarray: Heatmap array of shape (H, W) normalized to [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[:, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks were not triggered (e.g. Identity layer)
            H, W = input_tensor.shape[-2:]
            return np.zeros((H, W), dtype=np.float32)

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        if gradients.ndim == 4:  # (B*T, C, H, W)
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = torch.clamp(cam, min=0)
            cam = cam.squeeze().cpu().numpy()
        else:
            cam = torch.clamp(activations, min=0).squeeze().cpu().numpy()

        if cam.ndim > 2:
            cam = np.mean(cam, axis=0)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heat-map on input RGB image.

        Args:
            image (np.ndarray): Original image (H, W, 3) in uint8 [0-255]
            heatmap (np.ndarray): Heatmap (H, W) in float [0-1]
            alpha (float): Blending factor
            colormap: OpenCV colormap constant

        Returns:
            np.ndarray: Blended RGB image uint8
        """
        H, W = image.shape[:2]
        resized_heatmap = cv2.resize(heatmap, (W, H))
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image, 1.0 - alpha, colored_heatmap, alpha, 0)
        return overlay
