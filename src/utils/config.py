"""
Configuration utilities for DeepVision AI.

Loads, validates, and manages configuration settings with support for
environment variable overrides and cross-platform path resolution.

Configuration Precedence:
-------------------------
1. Explicit CLI / programmatic argument (highest priority)
2. Environment variables (DEEPVISION_CONFIG, DEEPVISION_DATA_ROOT, DEEPVISION_OUTPUT_DIR)
3. YAML configuration file (e.g. configs/default.yaml)
4. Built-in defaults (lowest priority)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path cleanly using pathlib, normalizing cross-platform separators.

    Args:
        path: Path string or Path object.
        base_dir: Optional base directory to prepend if path is relative.

    Returns:
        Path: Normalized Path object.
    """
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    return p


def _resolve_data_root_paths(config: Dict[str, Any], data_root: Path):
    """Internal helper to resolve data subpaths given a base data_root."""
    config["paths"]["data_root"] = str(data_root)

    # Raw data resolution
    if (data_root / "raw").exists():
        config["paths"]["raw_data"] = str(data_root / "raw")
    else:
        config["paths"]["raw_data"] = str(data_root / "raw")

    # Processed data resolution
    config["paths"]["processed_data"] = str(data_root / "processed")

    # Sequences resolution (support hierarchical, flat sequences, or direct root)
    if (data_root / "intermediate" / "sequences").exists():
        config["paths"]["sequences"] = str(data_root / "intermediate" / "sequences")
    elif (data_root / "sequences").exists():
        config["paths"]["sequences"] = str(data_root / "sequences")
    else:
        config["paths"]["sequences"] = str(data_root / "intermediate" / "sequences")

    # Splits directory resolution (detect if mounted dataset includes splits)
    if (data_root / "splits").exists():
        config["paths"]["splits"] = str(data_root / "splits")
    elif (data_root / "outputs" / "splits").exists():
        config["paths"]["splits"] = str(data_root / "outputs" / "splits")


def _resolve_output_dir_paths(config: Dict[str, Any], output_dir: Path):
    """Internal helper to resolve output subpaths given an output_dir."""
    config["paths"]["outputs"] = str(output_dir)
    config["paths"]["checkpoints"] = str(output_dir / "checkpoints")
    config["paths"]["logs"] = str(output_dir / "logs")
    config["paths"]["metrics"] = str(output_dir / "metrics")
    config["paths"]["predictions"] = str(output_dir / "predictions")
    config["paths"]["runs"] = str(output_dir / "runs")


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply supported environment variable overrides to configuration.

    Supported Environment Variables:
        DEEPVISION_DATA_ROOT: Overrides base dataset directory paths.
        DEEPVISION_OUTPUT_DIR: Overrides base output directory paths.

    Args:
        config: Configuration dictionary.

    Returns:
        dict: Updated configuration dictionary.
    """
    if "paths" not in config:
        config["paths"] = {}

    data_root_env = os.environ.get("DEEPVISION_DATA_ROOT")
    if data_root_env:
        _resolve_data_root_paths(config, Path(data_root_env))

    output_dir_env = os.environ.get("DEEPVISION_OUTPUT_DIR")
    if output_dir_env:
        _resolve_output_dir_paths(config, Path(output_dir_env))

    return config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load YAML configuration file with support for environment variable overrides.

    Precedence:
        1. Explicitly passed arguments (`config_path`, `data_root`, `output_dir`).
        2. Environment variables (`DEEPVISION_CONFIG`, `DEEPVISION_DATA_ROOT`, `DEEPVISION_OUTPUT_DIR`).
        3. YAML file contents.
        4. Built-in defaults.

    Args:
        config_path: Optional path to YAML configuration. Defaults to env var or configs/default.yaml.
        data_root: Optional explicit override for dataset root.
        output_dir: Optional explicit override for output directory.

    Returns:
        dict: Configuration dictionary.
    """
    if config_path is None:
        config_path = os.environ.get("DEEPVISION_CONFIG", "configs/default.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    # Ensure required sections exist
    config.setdefault("paths", {})
    config.setdefault("dataset", {})
    config.setdefault("dataloader", {})
    config.setdefault("training", {})
    config.setdefault("model", {})
    config.setdefault("evaluation", {})

    # Apply environment variable overrides
    config = apply_env_overrides(config)

    # Apply explicit programmatic overrides (highest priority)
    if data_root is not None:
        _resolve_data_root_paths(config, Path(data_root))

    if output_dir is not None:
        _resolve_output_dir_paths(config, Path(output_dir))

    return config


def validate_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configured paths for existence, readability, and writability.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        dict: Detailed path validation report.
    """
    paths = config.get("paths", {})
    report = {
        "valid": True,
        "input_paths": {},
        "output_paths": {},
        "issues": [],
    }

    # Input paths (must exist and be readable)
    input_keys = ["raw_data", "processed_data", "sequences", "splits"]
    for key in input_keys:
        if key in paths:
            p = Path(paths[key])
            exists = p.exists()
            readable = os.access(p, os.R_OK) if exists else False
            report["input_paths"][key] = {
                "path": str(p),
                "exists": exists,
                "readable": readable,
            }
            if not exists:
                report["issues"].append(f"Input path '{key}' does not exist: {p}")
            elif not readable:
                report["issues"].append(f"Input path '{key}' exists but is not readable: {p}")

    # Output paths (must be writable, created if missing)
    out_dir_path = Path(paths.get("outputs", "outputs"))
    try:
        out_dir_path.mkdir(parents=True, exist_ok=True)
        writable = os.access(out_dir_path, os.W_OK)
    except Exception as e:
        writable = False
        report["issues"].append(f"Could not create output directory: {out_dir_path} ({e})")

    report["output_paths"]["outputs"] = {
        "path": str(out_dir_path),
        "exists": out_dir_path.exists(),
        "writable": writable,
    }

    if not writable:
        report["valid"] = False
        report["issues"].append(f"Output directory is not writable: {out_dir_path}")

    return report


def get_path(config: Dict[str, Any], key: str) -> Path:
    """
    Return project path from configuration as a Path object.

    Args:
        config: Configuration dictionary.
        key: Key inside config['paths'].

    Returns:
        Path: Path object.
    """
    if "paths" not in config or key not in config["paths"]:
        raise KeyError(f"Path key '{key}' not found in configuration 'paths' section.")
    return Path(config["paths"][key])


def print_config(config: Dict[str, Any]):
    """
    Nicely print configuration.
    """
    print("=" * 60)
    print(" DeepVision AI Configuration")
    print("=" * 60)

    for section, values in config.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"{k:20} : {v}")
        else:
            print(values)

    print("=" * 60)
