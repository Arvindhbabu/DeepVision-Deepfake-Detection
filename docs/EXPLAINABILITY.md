# DeepVision AI — Grad-CAM Explainability Specification

## 1. Overview

DeepVision AI incorporates **Grad-CAM** (Gradient-weighted Class Activation Mapping) in `src/explainability/gradcam.py` to extract spatial attention heatmaps indicating facial feature anomalies influencing deepfake detection decisions.

---

## 2. Python API Usage

```python
import cv2
import torch
from src.explainability.gradcam import GradCAM
from src.models.vit_temporal_pooling import ViTTemporalPooling

# Load model
model = ViTTemporalPooling(pretrained=True)
model.eval()

# Target spatial feature layer
target_layer = model.backbone.conv_proj

# Instantiate GradCAM
gradcam = GradCAM(model=model, target_layer=target_layer)

# Generate heatmap for input sequence tensor (B, T, C, H, W)
input_tensor = torch.randn(1, 26, 3, 224, 224, requires_grad=True)
heatmap = gradcam.generate_heatmap(input_tensor, target_class=1)

# Overlay heatmap on original facial image
frame_rgb = cv2.imread("face.jpg")[:, :, ::-1]
visualization = GradCAM.overlay_heatmap(frame_rgb, heatmap)
```

---

## 3. Methodological Disclosures & Limitations

- **Interpretability Tool**: Grad-CAM visual heatmaps provide model feature interpretation by highlighting spatial gradients.
- **Not Causal Proof**: Visual heatmaps highlight spatial regions associated with classification decisions; they do not constitute mathematical or legal proof of video forgery.
