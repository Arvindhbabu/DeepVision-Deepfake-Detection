"""
DeepVision AI — Verified Dataset Exporter for Kaggle.

Prepares, validates, and packages the verified 138 non-zero FF++ sequences,
leak-free splits, manifest, README, and SHA-256 checksums into:
outputs/kaggle_bundle/deepvision_ffpp_verified.zip
"""

import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_actors(vname: str) -> set:
    clean = str(vname)[:-4] if str(vname).endswith(".npy") else str(vname)
    return set(clean.split("_"))


def build_readme(
    train_count: int,
    val_count: int,
    test_count: int,
    total_count: int,
) -> str:
    return f"""# DeepVision AI — Verified FaceForensics++ (FF++) Dataset

## Overview
This dataset contains **{total_count} verified, non-zero facial frame sequences** from FaceForensics++ (FF++ c23 compression).
All zero-filled development placeholder files have been strictly excluded.

- **Total Sequences**: {total_count} (69 Real, 69 Fake)
- **Tensor Shape**: `(30, 260, 260, 3)` (30 frames, 260x260 resolution, 3 RGB channels)
- **Data Type**: `uint8` (pixel values range `[0, 255]`)
- **Label Invariant**: `0 = REAL`, `1 = FAKE`

## Partitions & Leakage Prevention
Partitions were engineered using **Disjoint Set Union (DSU) actor-level clustering**:
- **Train Split**: {train_count} sequences (48 Real, 48 Fake)
- **Validation Split**: {val_count} sequences (10 Real, 10 Fake)
- **Test Split**: {test_count} sequences (11 Real, 11 Fake)

**Leakage Audit**:
- Sequence path overlap: **0**
- Video ID overlap: **0**
- Actor identity overlap: **0**

## Directory Layout
```
deepvision_ffpp_verified/
├── sequences/
│   └── ffpp/
│       ├── 000.npy
│       ├── 000_003.npy
│       └── ... (138 .npy files)
├── splits/
│   ├── train.csv (96 samples)
│   ├── val.csv   (20 samples)
│   └── test.csv  (22 samples)
├── dataset_manifest.csv (complete metadata audit)
├── README.md
└── checksums.sha256
```

## How to Load in Python / PyTorch

```python
import numpy as np
import pandas as pd
from pathlib import Path

# When mounted in Kaggle:
DATASET_ROOT = Path("/kaggle/input/deepvision-ffpp-verified") # or your dataset slug

# Load split
train_df = pd.read_csv(DATASET_ROOT / "splits" / "train.csv")

# Load first sequence
first_row = train_df.iloc[0]
seq_path = DATASET_ROOT / first_row["sequence_path"]
sequence = np.load(seq_path) # Shape: (30, 260, 260, 3), dtype: uint8
label = int(first_row["label"]) # 0 for Real, 1 for Fake
```

## Integrity Verification
Verify file integrity using the provided `checksums.sha256`:
```bash
sha256sum -c checksums.sha256
```
"""


def export_kaggle_bundle():
    print("=" * 65)
    print(" DeepVision AI — Kaggle Dataset Bundle Exporter")
    print("=" * 65)

    manifest_path = PROJECT_ROOT / "outputs" / "reports" / "dataset_manifest.csv"
    train_path = PROJECT_ROOT / "outputs" / "splits" / "train.csv"
    val_path = PROJECT_ROOT / "outputs" / "splits" / "val.csv"
    test_path = PROJECT_ROOT / "outputs" / "splits" / "test.csv"

    for p in [manifest_path, train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required source file: {p}")

    df_manifest = pd.read_csv(manifest_path)
    df_valid_manifest = df_manifest[df_manifest["validation_status"] == "VALID"].copy()

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print(f"Loaded Manifest  : {len(df_valid_manifest)} VALID sequences")
    print(f"Loaded Splits    : Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    total_splits = len(df_train) + len(df_val) + len(df_test)
    assert len(df_valid_manifest) == 138, f"Expected 138 valid, found {len(df_valid_manifest)}"
    assert total_splits == 138, f"Expected 138 split rows, found {total_splits}"

    # Verify zero overlap
    train_seqs = set(df_train["sequence_path"])
    val_seqs = set(df_val["sequence_path"])
    test_seqs = set(df_test["sequence_path"])

    assert len(train_seqs & val_seqs) == 0, "Sequence overlap between train and val!"
    assert len(train_seqs & test_seqs) == 0, "Sequence overlap between train and test!"
    assert len(val_seqs & test_seqs) == 0, "Sequence overlap between val and test!"

    train_actors = set().union(*[get_actors(v) for v in df_train["video_name"]])
    val_actors = set().union(*[get_actors(v) for v in df_val["video_name"]])
    test_actors = set().union(*[get_actors(v) for v in df_test["video_name"]])

    assert len(train_actors & val_actors) == 0, "Actor overlap between train and val!"
    assert len(train_actors & test_actors) == 0, "Actor overlap between train and test!"
    assert len(val_actors & test_actors) == 0, "Actor overlap between val and test!"
    print("[PASS] Zero sequence, video, and actor overlap confirmed across all splits.")

    # Create staging directory
    bundle_dir = PROJECT_ROOT / "outputs" / "kaggle_bundle"
    staging_dir = bundle_dir / "staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    seq_dest_dir = staging_dir / "sequences" / "ffpp"
    splits_dest_dir = staging_dir / "splits"
    seq_dest_dir.mkdir(parents=True, exist_ok=True)
    splits_dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy sequence files and assert non-zero
    print("\nCopying and validating 138 sequence files into staging...")
    copied_files = 0
    for _, row in df_valid_manifest.iterrows():
        src_path = PROJECT_ROOT / row["sequence_path"]
        if not src_path.exists():
            raise FileNotFoundError(f"Source sequence file missing: {src_path}")

        arr = np.load(src_path)
        assert arr.shape == (30, 260, 260, 3), f"Bad shape for {src_path}: {arr.shape}"
        assert arr.dtype == np.uint8, f"Bad dtype for {src_path}: {arr.dtype}"
        assert arr.max() > 0, f"Encountered all-zero placeholder at {src_path}!"

        dest_file = seq_dest_dir / src_path.name
        shutil.copy2(src_path, dest_file)
        copied_files += 1

    print(f"[PASS] Successfully copied and verified {copied_files} .npy sequence files.")

    # 2. Prepare exported splits with relative sequence paths: sequences/ffpp/<name>.npy
    def format_split_df(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out["sequence_path"] = out["video_name"].apply(
            lambda v: f"sequences/ffpp/{v[:-4] if v.endswith('.npy') else v}.npy"
        )
        return out

    exp_train = format_split_df(df_train)
    exp_val = format_split_df(df_val)
    exp_test = format_split_df(df_test)

    exp_train.to_csv(splits_dest_dir / "train.csv", index=False)
    exp_val.to_csv(splits_dest_dir / "val.csv", index=False)
    exp_test.to_csv(splits_dest_dir / "test.csv", index=False)
    print("[PASS] Exported train.csv, val.csv, and test.csv with relative paths.")

    # 3. Export filtered manifest
    exp_manifest = df_valid_manifest.copy()
    exp_manifest["sequence_path"] = exp_manifest["video_name"].apply(
        lambda v: f"sequences/ffpp/{v[:-4] if v.endswith('.npy') else v}.npy"
    )
    exp_manifest.to_csv(staging_dir / "dataset_manifest.csv", index=False)
    print("[PASS] Exported filtered dataset_manifest.csv (138 rows).")

    # 4. Generate README.md
    readme_content = build_readme(
        train_count=len(exp_train),
        val_count=len(exp_val),
        test_count=len(exp_test),
        total_count=copied_files,
    )
    with open(staging_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("[PASS] Generated README.md documentation.")

    # 5. Generate SHA-256 Checksums
    print("\nGenerating SHA-256 checksums...")
    checksum_lines = []
    for root, _, files in os.walk(staging_dir):
        for f in sorted(files):
            if f == "checksums.sha256":
                continue
            file_p = Path(root) / f
            rel_p = file_p.relative_to(staging_dir).as_posix()
            digest = compute_sha256(file_p)
            checksum_lines.append(f"{digest}  {rel_p}\n")

    checksum_file = staging_dir / "checksums.sha256"
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)
    print(f"[PASS] Computed SHA-256 for {len(checksum_lines)} exported files -> checksums.sha256")

    # 6. Create ZIP archive
    zip_path = bundle_dir / "deepvision_ffpp_verified.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"\nCompressing staging directory into {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging_dir):
            for f in sorted(files):
                file_p = Path(root) / f
                arc_name = file_p.relative_to(staging_dir).as_posix()
                zf.write(file_p, arcname=arc_name)

    # 7. Reopen & Verify ZIP Archive
    print("\nReopening and validating ZIP archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        npy_in_zip = [n for n in namelist if n.endswith(".npy")]
        csv_in_zip = [n for n in namelist if n.endswith(".csv")]
        md_in_zip = [n for n in namelist if n.endswith(".md")]
        sha_in_zip = [n for n in namelist if n.endswith(".sha256")]

        assert len(npy_in_zip) == 138, f"Expected 138 .npy files in zip, found {len(npy_in_zip)}"
        assert len(csv_in_zip) == 4, f"Expected 4 .csv files in zip, found {len(csv_in_zip)}"
        assert len(md_in_zip) == 1, f"Expected README.md in zip, found {len(md_in_zip)}"
        assert len(sha_in_zip) == 1, f"Expected checksums.sha256 in zip, found {len(sha_in_zip)}"

        # Test archive integrity
        bad_file = zf.testzip()
        if bad_file is not None:
            raise RuntimeError(f"Corrupted file detected inside zip: {bad_file}")

        # Check random .npy readability from zip
        with zf.open("sequences/ffpp/000.npy") as f:
            test_arr = np.load(f)
            assert test_arr.shape == (30, 260, 260, 3)
            assert test_arr.max() > 0

    zip_size_bytes = zip_path.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    zip_sha256 = compute_sha256(zip_path)

    print("=" * 65)
    print(" KAGGLE EXPORT BUNDLE COMPLETED & VERIFIED")
    print("=" * 65)
    print(f"Archive Path     : {zip_path}")
    print(f"Archive Size     : {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
    print(f"Archive SHA-256  : {zip_sha256}")
    print(f"Total Files      : {len(namelist)} (138 .npy, 4 .csv, 1 .md, 1 .sha256)")
    print(f"Zero Placeholders: EXCLUDED (0)")
    print("=" * 65)

    # Clean up staging directory to save local disk space
    shutil.rmtree(staging_dir)
    print("[INFO] Cleaned temporary staging directory.")

    return {
        "zip_path": str(zip_path),
        "size_mb": round(zip_size_mb, 2),
        "sha256": zip_sha256,
        "total_files": len(namelist),
        "npy_count": len(npy_in_zip),
        "train_count": len(exp_train),
        "val_count": len(exp_val),
        "test_count": len(exp_test),
    }


if __name__ == "__main__":
    export_kaggle_bundle()

