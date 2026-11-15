"""
DeepVision AI — Legacy Configuration Bridge.

Preserves backward compatibility for preprocessing and standalone scripts
by synchronizing with the canonical configuration system in `src.utils.config`.
"""

import os
from pathlib import Path
from src.utils.config import load_config

# Attempt to load canonical configuration
try:
    _cfg = load_config()
except Exception:
    _cfg = {}

_paths = _cfg.get("paths", {})
_dataset = _cfg.get("dataset", {})
_dataloader = _cfg.get("dataloader", {})

# Base directories
BASE_DIR = str(Path(os.getcwd()).resolve())
DATA_DIR = str(Path(_paths.get("data_root", Path(BASE_DIR) / "data")).resolve())
RAW_FFPP = str(Path(_paths.get("raw_data", Path(DATA_DIR) / "raw")) / "ffpp_c23")
RAW_CELEBDF = str(Path(_paths.get("raw_data", Path(DATA_DIR) / "raw")) / "celebdf_v2_sampled")

INTERMEDIATE_DIR = str(Path(DATA_DIR) / "intermediate")
FRAMES_DIR = str(Path(INTERMEDIATE_DIR) / "frames")
ALIGNED_DIR = str(Path(INTERMEDIATE_DIR) / "aligned_faces")
SEQS_DIR = str(Path(_paths.get("sequences", Path(INTERMEDIATE_DIR) / "sequences")).resolve())

# Preprocessing parameters
FPS = int(_dataset.get("fps", 5))
SEQ_LEN = int(_dataset.get("sequence_length", 30))
INPUT_SIZE = int(_dataset.get("image_size", 224))
MIN_FRAMES = 40
NUM_WORKERS = int(_dataloader.get("num_workers", 0))

# Output formats
SEQ_SAVE_FORMAT = "npy"

