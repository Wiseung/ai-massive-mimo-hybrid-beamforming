#!/usr/bin/env python
"""Check the local Python/CUDA/beamforming environment."""

from __future__ import annotations

import importlib
import platform
import sys

from _bootstrap import add_src_to_path
import torch


def _module_status(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, str(version)
    except Exception as exc:  # pragma: no cover - exercised in script mode
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    add_src_to_path()
    print("Python version:", sys.version.replace("\n", " "))
    print("Python executable:", sys.executable)
    print("Python version tuple:", sys.version_info[:3])
    print("Platform:", platform.platform())
    if sys.version_info < (3, 11):
        print("DeepMIMO note: DeepMIMO PyPI may require Python >= 3.11.")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    deepmimo_ok = False
    for name in ("sionna", "DeepMIMO", "deepmimo"):
        ok, detail = _module_status(name)
        if name == "deepmimo":
            deepmimo_ok = ok
        print(f"{name}:", "OK" if ok else "MISSING", detail)
    if not deepmimo_ok:
        print("Install DeepMIMO with: pip install deepmimo")


if __name__ == "__main__":
    main()
