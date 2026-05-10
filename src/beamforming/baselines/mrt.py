"""Maximum-ratio transmission baseline."""

from __future__ import annotations

import torch

from beamforming.models.constraints import power_normalization


def mrt_precoder(channel: torch.Tensor, total_power: float = 1.0) -> torch.Tensor:
    """Compute MRT precoder for channel shape (B, K, Nt)."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    precoder = channel.transpose(-2, -1).conj()
    return power_normalization(precoder, total_power=total_power)
