"""Zero-forcing baseline."""

from __future__ import annotations

import torch

from beamforming.models.constraints import power_normalization


def zf_precoder(channel: torch.Tensor, total_power: float = 1.0, reg: float = 1e-9) -> torch.Tensor:
    """Compute ZF precoder for channel shape (B, K, Nt)."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    gram = channel @ channel.transpose(-2, -1).conj()
    eye = torch.eye(gram.size(-1), dtype=gram.dtype, device=gram.device).unsqueeze(0)
    inv = torch.linalg.inv(gram + reg * eye)
    precoder = channel.transpose(-2, -1).conj() @ inv
    return power_normalization(precoder, total_power=total_power)
