"""
DeepVision AI — Cloud Dataset & Path Discovery Utility.

Provides robust, cross-platform path resolution for Kaggle, Colab, and local environments,
ensuring strict validation of mounted datasets and preventing silent placeholder fallbacks.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


def is_kaggle_environment() -> bool:
    """Return True if executing within a Kaggle notebook environment."""
    return Path("/kaggle/working").exists() or Path("/kaggle/input").exists()


def is_colab_environment() -> bool:
    """Return True if executing within a Google Colab notebook environment."""
    return Path("/content").exists() and ("COLAB_GPU" in os.environ or Path("/content/sample_data").exists())


def discover_kaggle_dataset(
    kaggle_input_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Inspects /kaggle/input to discover an attached dataset containing sequences.

    Returns:
        tuple: (dataset_root, sequences_dir) if discovered, else (None, None).
    Raises:
        RuntimeError: If executing on Kaggle but /kaggle/input is completely empty.
    """
    input_dir = Path(kaggle_input_dir) if kaggle_input_dir else Path("/kaggle/input")

    if not input_dir.exists():
        return None, None

    entries = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not entries:
        if is_kaggle_environment():
            raise RuntimeError(
                "Kaggle execution detected, but '/kaggle/input' is EMPTY!\n"
                "Please attach the DeepVision dataset in Kaggle (e.g., 'deepvision-ffpp-138') "
                "via the '+ Add Input' panel before executing."
            )
        return None, None

    # Search for sequences directory across attached datasets
    for candidate_dir in entries:
        # Check direct sequences folder
        direct_seq = candidate_dir / "sequences"
        if direct_seq.exists() and direct_seq.is_dir():
            return candidate_dir, direct_seq

        # Check nested intermediate/sequences folder
        nested_seq = candidate_dir / "intermediate" / "sequences"
        if nested_seq.exists() and nested_seq.is_dir():
            return candidate_dir, nested_seq

        # Check glob search for any folder containing .npy
        any_npy = list(candidate_dir.glob("**/*.npy"))
        if any_npy:
            return candidate_dir, any_npy[0].parent

    # Fallback to first attached dataset if present
    return entries[0], entries[0]


def resolve_split_paths(
    splits_dir: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Locates and validates train.csv, val.csv, and test.csv.

    Searches in:
    1. Explicit splits_dir
    2. dataset_root / "splits"
    3. repo_root / "outputs" / "splits"
    4. ./outputs/splits
    """
    if splits_dir is not None:
        p = Path(splits_dir)
        if (p / "train.csv").exists() and (p / "val.csv").exists() and (p / "test.csv").exists():
            return {
                "splits_dir": p,
                "train_csv": p / "train.csv",
                "val_csv": p / "val.csv",
                "test_csv": p / "test.csv",
            }
        raise FileNotFoundError(
            f"Explicitly specified splits_dir '{splits_dir}' is missing required split CSV files (train.csv, val.csv, test.csv)."
        )

    candidates = []
    if dataset_root and (Path(dataset_root) / "splits").exists():
        candidates.append(Path(dataset_root) / "splits")
    if repo_root and (Path(repo_root) / "outputs" / "splits").exists():
        candidates.append(Path(repo_root) / "outputs" / "splits")
    if Path("outputs/splits").exists():
        candidates.append(Path("outputs/splits"))

    resolved_splits_dir = None
    for c in candidates:
        if (c / "train.csv").exists() and (c / "val.csv").exists() and (c / "test.csv").exists():
            resolved_splits_dir = c
            break

    if resolved_splits_dir is None:
        raise FileNotFoundError(
            "Could not locate verified split CSVs (train.csv, val.csv, test.csv). "
            f"Searched candidates: {[str(c) for c in candidates]}"
        )

    return {
        "splits_dir": resolved_splits_dir,
        "train_csv": resolved_splits_dir / "train.csv",
        "val_csv": resolved_splits_dir / "val.csv",
        "test_csv": resolved_splits_dir / "test.csv",
    }


def verify_real_sequences_exist(
    split_csv: Path,
    sequence_root: Path,
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validates that every row in the split CSV points to a real, readable,
    non-zero sequence on disk.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(split_csv)
    if expected_count is not None and len(df) != expected_count:
        raise ValueError(f"Expected {expected_count} rows in {split_csv}, but found {len(df)}")

    verified_count = 0
    zero_placeholder_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        vname = str(row["video_name"])
        clean_name = vname[:-4] if vname.endswith(".npy") else vname

        # Resolve path
        direct_p = Path(str(row.get("sequence_path", "")))
        if direct_p.exists():
            npy_path = direct_p
        elif (sequence_root / "ffpp" / f"{clean_name}.npy").exists():
            npy_path = sequence_root / "ffpp" / f"{clean_name}.npy"
        elif (sequence_root / f"{clean_name}.npy").exists():
            npy_path = sequence_root / f"{clean_name}.npy"
        else:
            missing_count += 1
            continue

        arr = np.load(npy_path)
        if arr.max() == 0 and arr.min() == 0:
            zero_placeholder_count += 1
        else:
            verified_count += 1

    return {
        "split_file": str(split_csv),
        "total_rows": len(df),
        "verified_real_count": verified_count,
        "zero_placeholder_count": zero_placeholder_count,
        "missing_count": missing_count,
        "is_valid": (verified_count == len(df) and zero_placeholder_count == 0 and missing_count == 0),
    }
