"""Fully-digital reference baselines.

These methods are references for the current digital-only MU-MISO setup.
They are not claimed as strict information-theoretic upper bounds.
"""

from __future__ import annotations

import torch

from beamforming.baselines.rzf import rzf_precoder
from beamforming.baselines.zf import zf_precoder


def fd_zf_precoder(channel: torch.Tensor, total_power: float = 1.0) -> torch.Tensor:
    """Fully-digital ZF reference."""
    return zf_precoder(channel, total_power=total_power)


def fd_rzf_precoder(
    channel: torch.Tensor,
    noise_var: float | torch.Tensor = 1.0,
    total_power: float = 1.0,
) -> torch.Tensor:
    """Fully-digital RZF reference."""
    return rzf_precoder(channel, noise_var=noise_var, total_power=total_power)
