# DeepVision AI — CPU Inference

Use `python predict.py --input video.mp4 --checkpoint PATH --config CONFIG --clips 3`.

The current predictor samples clips at deterministic temporal positions and decodes
only the requested frames, avoiding full-video buffering. The video container must
report a valid frame count, and OpenCV must be able to seek/decode those frames.

Model input is RGB resized to the configured size. If `model.pretrained: true`,
ImageNet channel normalization is applied after scaling to [0, 1].
A checkpoint must match the model architecture and preprocessing configuration
used for its training.

The score is a softmax model output, **not calibrated forensic certainty**.
Real/fake predictions from untrained or synthetic-smoke-test checkpoints have
no real-world meaning. Long videos may still incur significant seeking costs.
