#!/usr/bin/env python3
"""
DeepVision AI — Environment & Cloud Diagnostic Utility.

Inspects the runtime environment and reports:
- Python, OS, platform, and hardware details
- Installed deep learning and ML packages
- CUDA, cuDNN, and GPU hardware metrics
- Cloud notebook environment detection (Kaggle, Google Colab, Google Drive)
- Configured dataset and output paths (readability, writability, disk space)
- Generates human-readable terminal reports and machine-readable JSON
"""

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure project root is in sys.path when invoked directly from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure MPLCONFIGDIR is set to a writable temporary directory
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()


def get_system_ram() -> Tuple[Optional[float], Optional[float]]:
    """
    Get total and available system RAM in gigabytes (GB).
    Uses psutil if available, with OS fallbacks.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / (1024 ** 3), 2), round(vm.available / (1024 ** 3), 2)
    except Exception:
        pass

    # Windows fallback via ctypes
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return (
                    round(stat.ullTotalPhys / (1024 ** 3), 2),
                    round(stat.ullAvailPhys / (1024 ** 3), 2),
                )
        except Exception:
            pass

    # Linux fallback via /proc/meminfo
    if platform.system() == "Linux":
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = parts[1].strip()
            total_kb = float(meminfo.get("MemTotal", "0 kB").split()[0])
            avail_kb = float(meminfo.get("MemAvailable", "0 kB").split()[0])
            return round(total_kb / (1024 ** 2), 2), round(avail_kb / (1024 ** 2), 2)
        except Exception:
            pass

    return None, None


def detect_cloud_environment() -> Dict[str, Any]:
    """Detect whether running in Kaggle, Google Colab, or a local environment."""
    is_kaggle = (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
        or Path("/kaggle/working").exists()
    )

    is_colab = (
        "google.colab" in sys.modules
        or "COLAB_GPU" in os.environ
        or "COLAB_RELEASE_TAG" in os.environ
        or Path("/content").exists()
    )

    google_drive_mounted = (
        Path("/content/drive/MyDrive").exists() or Path("/content/drive").exists()
    )

    if is_kaggle:
        platform_name = "Kaggle Notebook"
    elif is_colab:
        platform_name = "Google Colab"
    elif platform.system() == "Windows":
        platform_name = "Local Windows"
    else:
        platform_name = "Local / Remote Linux"

    return {
        "platform_name": platform_name,
        "is_kaggle": bool(is_kaggle),
        "is_colab": bool(is_colab),
        "google_drive_mounted": bool(google_drive_mounted),
        "kaggle_input_exists": Path("/kaggle/input").exists(),
        "kaggle_working_exists": Path("/kaggle/working").exists(),
        "colab_content_exists": Path("/content").exists(),
    }


def check_packages() -> Dict[str, Dict[str, Any]]:
    """Check availability and versions of core and optional packages without crashing."""
    packages_to_check = [
        ("torch", True),
        ("torchvision", True),
        ("numpy", True),
        ("cv2", True),
        ("sklearn", True),
        ("yaml", True),
        ("timm", False),
        ("facenet_pytorch", False),
        ("matplotlib", False),
        ("pandas", True),
        ("tqdm", False),
    ]

    results = {}
    for pkg_name, required in packages_to_check:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "Installed (no __version__)")
            results[pkg_name] = {
                "installed": True,
                "version": str(ver),
                "required": required,
                "status": "PASS",
            }
        except ImportError:
            status = "FAIL" if required else "WARNING"
            results[pkg_name] = {
                "installed": False,
                "version": None,
                "required": required,
                "status": status,
            }
        except Exception as e:
            results[pkg_name] = {
                "installed": False,
                "version": None,
                "required": required,
                "status": "FAIL",
                "error": str(e),
            }

    return results


def check_cuda_and_gpu() -> Dict[str, Any]:
    """Inspect CUDA runtime and GPU availability via PyTorch."""
    cuda_info: Dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "status": "WARNING",  # Default to WARNING on CPU (not FAIL)
    }

    try:
        import torch

        if torch.cuda.is_available():
            cuda_info["cuda_available"] = True
            cuda_info["status"] = "PASS"
            cuda_info["cuda_version"] = str(torch.version.cuda)
            try:
                cuda_info["cudnn_version"] = str(torch.backends.cudnn.version())
            except Exception:
                cuda_info["cudnn_version"] = None

            count = torch.cuda.device_count()
            cuda_info["device_count"] = count

            for i in range(count):
                dev_name = torch.cuda.get_device_name(i)
                try:
                    total_mem = round(
                        torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2
                    )
                except Exception:
                    total_mem = None

                cuda_info["devices"].append({
                    "index": i,
                    "name": dev_name,
                    "memory_gb": total_mem,
                })
        else:
            cuda_info["status"] = "WARNING"
            cuda_info["message"] = "CUDA is not available. Execution will run in CPU mode."
    except Exception as e:
        cuda_info["status"] = "FAIL"
        cuda_info["message"] = f"Error querying PyTorch CUDA state: {e}"

    return cuda_info


def check_disk_space(path: Union[str, Path]) -> Dict[str, Any]:
    """Get total and free disk space in GB for a given path."""
    try:
        p = Path(path)
        target = p if p.exists() else p.parent
        usage = shutil.disk_usage(str(target))
        return {
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
            "status": "PASS",
        }
    except Exception as e:
        return {
            "total_gb": None,
            "used_gb": None,
            "free_gb": None,
            "status": "WARNING",
            "error": str(e),
        }


def collect_diagnostics(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run full environment diagnostic collection.

    Returns:
        dict: Complete diagnostic metrics and checks.
    """
    total_ram, avail_ram = get_system_ram()
    cloud_env = detect_cloud_environment()
    pkg_status = check_packages()
    cuda_info = check_cuda_and_gpu()

    # Load configuration
    config_dict = {}
    config_file = config_path or os.environ.get("DEEPVISION_CONFIG", "configs/default.yaml")
    config_exists = Path(config_file).exists()

    paths_status: Dict[str, Any] = {}
    if config_exists:
        try:
            from src.utils.config import load_config, validate_paths
            config_dict = load_config(config_file)
            paths_status = validate_paths(config_dict)
        except Exception as e:
            paths_status = {"valid": False, "error": str(e), "issues": [str(e)]}
    else:
        paths_status = {
            "valid": False,
            "error": f"Config file not found: {config_file}",
            "issues": [f"Config file not found: {config_file}"],
        }

    # Disk usage
    disk_info = check_disk_space(".")
    output_disk_info = None
    if config_dict and "paths" in config_dict and "outputs" in config_dict["paths"]:
        output_disk_info = check_disk_space(config_dict["paths"]["outputs"])

    # Determine critical failure status
    critical_failure = False
    critical_errors: List[str] = []

    # PyTorch is required
    if not pkg_status.get("torch", {}).get("installed", False):
        critical_failure = True
        critical_errors.append("PyTorch ('torch') is not installed. Training and inference cannot run.")

    if not config_exists:
        critical_errors.append(f"Specified configuration file '{config_file}' was not found.")

    return {
        "status": "FAIL" if critical_failure else ("WARNING" if cuda_info["status"] == "WARNING" else "PASS"),
        "critical_failure": critical_failure,
        "critical_errors": critical_errors,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "Unknown",
            "cpu_count": os.cpu_count(),
        },
        "hardware": {
            "total_ram_gb": total_ram,
            "available_ram_gb": avail_ram,
            "disk": disk_info,
            "output_disk": output_disk_info,
        },
        "cloud": cloud_env,
        "packages": pkg_status,
        "cuda": cuda_info,
        "configuration": {
            "file": str(config_file),
            "exists": config_exists,
            "paths": config_dict.get("paths", {}),
            "validation": paths_status,
        },
    }


def format_report(diag: Dict[str, Any]) -> str:
    """Format diagnostic dictionary into a human-readable string with PASS/WARN/FAIL indicators."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(" DeepVision AI — Environment & Cloud Diagnostics")
    lines.append("=" * 70)

    # Status banner
    status_label = diag["status"]
    if status_label == "PASS":
        badge = "[PASS]"
    elif status_label == "WARNING":
        badge = "[WARNING]"
    else:
        badge = "[FAIL]"

    lines.append(f"Overall Environment Status: {badge} {status_label}")
    lines.append("-" * 70)

    # 1. Python & System
    py = diag["python"]
    plat = diag["platform"]
    lines.append("1. System & Runtime:")
    lines.append(f"  • Platform           : {plat['system']} {plat['release']} ({plat['machine']})")
    lines.append(f"  • Python Version     : {py['version']} [PASS]")
    lines.append(f"  • Python Executable  : {py['executable']}")
    lines.append(f"  • CPU Model          : {plat['processor']}")
    lines.append(f"  • CPU Logical Cores  : {plat['cpu_count']}")

    hw = diag["hardware"]
    ram_total = f"{hw['total_ram_gb']} GB" if hw["total_ram_gb"] is not None else "Unknown"
    ram_avail = f"{hw['available_ram_gb']} GB" if hw["available_ram_gb"] is not None else "Unknown"
    lines.append(f"  • System RAM         : Total {ram_total} | Available {ram_avail}")

    disk = hw.get("disk", {})
    if disk and disk.get("free_gb") is not None:
        lines.append(f"  • Workspace Disk     : {disk['free_gb']} GB free of {disk['total_gb']} GB")

    lines.append("")

    # 2. Cloud Environment Detection
    cloud = diag["cloud"]
    lines.append("2. Cloud Notebook Detection:")
    lines.append(f"  • Detected Platform  : {cloud['platform_name']}")
    lines.append(f"  • Kaggle Runtime     : {'YES' if cloud['is_kaggle'] else 'NO'}")
    if cloud["is_kaggle"]:
        lines.append(f"    - /kaggle/input    : {'Found [PASS]' if cloud['kaggle_input_exists'] else 'Not Found'}")
        lines.append(f"    - /kaggle/working  : {'Found [PASS]' if cloud['kaggle_working_exists'] else 'Not Found'}")

    lines.append(f"  • Google Colab       : {'YES' if cloud['is_colab'] else 'NO'}")
    if cloud["is_colab"]:
        lines.append(f"    - Google Drive     : {'Mounted [PASS]' if cloud['google_drive_mounted'] else 'Not Mounted [WARNING]'}")

    lines.append("")

    # 3. Core Packages
    lines.append("3. Python Packages:")
    for pkg_name, info in diag["packages"].items():
        tag = f"[{info['status']}]"
        ver_str = info["version"] or "NOT INSTALLED"
        req_str = "(Required)" if info["required"] else "(Optional)"
        lines.append(f"  • {pkg_name:<16}: {ver_str:<18} {req_str:<11} {tag}")

    lines.append("")

    # 4. Acceleration & GPU
    cuda = diag["cuda"]
    lines.append("4. Hardware Acceleration (CUDA / GPU):")
    if cuda["cuda_available"]:
        lines.append(f"  • CUDA Available     : YES [PASS]")
        lines.append(f"  • CUDA Version       : {cuda['cuda_version']}")
        lines.append(f"  • cuDNN Version      : {cuda.get('cudnn_version') or 'N/A'}")
        lines.append(f"  • GPU Device Count   : {cuda['device_count']}")
        for dev in cuda["devices"]:
            lines.append(f"    - GPU {dev['index']}: {dev['name']} ({dev['memory_gb']} GB VRAM)")
    else:
        lines.append("  • CUDA Available     : NO [WARNING]")
        lines.append("  • Compute Device     : CPU (Suitable for development, CPU tests, and smoke tests)")
        lines.append("    Note: Training on full video datasets requires a GPU environment (Kaggle/Colab).")

    lines.append("")

    # 5. Configuration & Paths
    cfg = diag["configuration"]
    lines.append("5. Configuration & Paths:")
    lines.append(f"  • Active Config      : {cfg['file']} ({'Found [PASS]' if cfg['exists'] else 'Missing [FAIL]'})")

    paths = cfg.get("paths", {})
    if paths:
        lines.append(f"  • Raw Data Path      : {paths.get('raw_data', 'N/A')}")
        lines.append(f"  • Sequences Path     : {paths.get('sequences', 'N/A')}")
        lines.append(f"  • Output Root        : {paths.get('outputs', 'N/A')}")
        lines.append(f"  • Checkpoints Path   : {paths.get('checkpoints', 'N/A')}")

    val = cfg.get("validation", {})
    if val:
        lines.append(f"  • Paths Validated    : {'YES [PASS]' if val.get('valid') else 'ISSUES FOUND [WARNING]'}")
        for issue in val.get("issues", []):
            lines.append(f"    - Issue: {issue}")

    # Critical errors if any
    if diag["critical_errors"]:
        lines.append("")
        lines.append("CRITICAL ISSUES:")
        for err in diag["critical_errors"]:
            lines.append(f"  [FAIL] {err}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="DeepVision AI Environment & Cloud Diagnostic Utility"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration to validate (defaults to DEEPVISION_CONFIG or configs/default.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostics in machine-readable JSON format",
    )

    args = parser.parse_args()

    diagnostics = collect_diagnostics(config_path=args.config)

    if args.json:
        print(json.dumps(diagnostics, indent=2))
    else:
        report = format_report(diagnostics)
        print(report)

    # Return nonzero ONLY for critical failures (e.g. PyTorch not installed)
    if diagnostics["critical_failure"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
