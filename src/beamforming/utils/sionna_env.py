"""Optional Sionna environment helpers for smoke demos."""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any

import torch


def _safe_gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return "unavailable"


def collect_sionna_env_info() -> dict[str, Any]:
    """Collect a compact environment summary without raising noisy tracebacks."""
    info: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "python_version_tuple": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": _safe_gpu_name(),
        "sionna_import_ok": False,
        "sionna_version": None,
        "sionna_error": None,
        "install_hint": "pip install sionna-no-rt or pip install sionna",
    }
    try:
        module = importlib.import_module("sionna")
        info["sionna_import_ok"] = True
        info["sionna_version"] = getattr(module, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - script-path behavior
        info["sionna_error"] = f"{type(exc).__name__}: {exc}"
    return info


def format_sionna_env_lines(info: dict[str, Any]) -> list[str]:
    """Return human-readable status lines for CLI scripts."""
    lines = [
        f"Python version: {info['python_version']}",
        f"PyTorch version: {info['torch_version']}",
        f"CUDA available: {info['cuda_available']}",
        f"GPU name: {info['gpu_name'] or 'N/A'}",
        f"Sionna import: {'OK' if info['sionna_import_ok'] else 'MISSING'}",
    ]
    if info["sionna_import_ok"]:
        lines.append(f"Sionna version: {info['sionna_version']}")
    else:
        lines.append(f"Sionna error: {info['sionna_error']}")
        lines.append(f"Install hint: {info['install_hint']}")
    return lines
