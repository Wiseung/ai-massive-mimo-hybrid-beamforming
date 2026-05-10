"""Regularized zero-forcing baseline."""

from __future__ import annotations

import torch

from beamforming.models.constraints import power_normalization


def rzf_precoder(channel: torch.Tensor, noise_var: float = 1.0, total_power: float = 1.0) -> torch.Tensor:
    """Compute RZF precoder for channel shape (B, K, Nt)."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    num_users = channel.size(-2)
    noise = torch.as_tensor(noise_var, dtype=torch.float32, device=channel.device)
    if noise.ndim == 0:
        alpha = num_users * noise
    else:
        alpha = (num_users * noise).view(-1, 1, 1)
    gram = channel @ channel.transpose(-2, -1).conj()
    eye = torch.eye(num_users, dtype=gram.dtype, device=gram.device).unsqueeze(0)
    inv = torch.linalg.inv(gram + alpha * eye)
    precoder = channel.transpose(-2, -1).conj() @ inv
    return power_normalization(precoder, total_power=total_power)
