"""Helpers for optional Sionna PHY smoke demos with explicit fallback behavior."""

from __future__ import annotations

from typing import Any

import torch


def try_import_sionna_phy() -> dict[str, Any]:
    """Return available Sionna PHY classes/functions without hard failing."""
    try:
        from sionna.phy.channel import AWGN
        from sionna.phy.mapping import BinarySource, Demapper, Mapper

        return {
            "import_ok": True,
            "AWGN": AWGN,
            "BinarySource": BinarySource,
            "Mapper": Mapper,
            "Demapper": Demapper,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {
            "import_ok": False,
            "AWGN": None,
            "BinarySource": None,
            "Mapper": None,
            "Demapper": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def add_awgn_torch(signal: torch.Tensor, snr_db: float) -> tuple[torch.Tensor, float]:
    """Torch AWGN fallback with unit average symbol power assumption."""
    noise_var = float(10.0 ** (-snr_db / 10.0))
    noise_scale = (noise_var / 2.0) ** 0.5
    noise = noise_scale * (torch.randn_like(signal) + 1j * torch.randn_like(signal))
    return signal + noise, noise_var
