"""Synthetic CSI generators for narrowband and simple OFDM channels."""

from __future__ import annotations

import math

import torch


def rayleigh_narrowband_channel(
    num_samples: int,
    num_users: int,
    num_bs_ant: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Generate IID Rayleigh MU-MISO channels with shape (N, K, Nt)."""
    real = torch.randn(num_samples, num_users, num_bs_ant, device=device, dtype=torch.float32)
    imag = torch.randn(num_samples, num_users, num_bs_ant, device=device, dtype=torch.float32)
    return torch.complex(real, imag).to(dtype) / math.sqrt(2.0)


def sparse_geometric_mmwave_channel(
    num_samples: int,
    num_users: int,
    num_bs_ant: int,
    num_paths: int,
    angle_spread: float = math.pi,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Generate a simple sparse geometric mmWave channel for a ULA BS."""
    n = torch.arange(num_bs_ant, device=device, dtype=torch.float32)
    channels = []
    for _ in range(num_samples):
        user_channels = []
        for _user in range(num_users):
            h = torch.zeros(num_bs_ant, device=device, dtype=dtype)
            aoas = (torch.rand(num_paths, device=device) - 0.5) * angle_spread
            gains = torch.complex(
                torch.randn(num_paths, device=device),
                torch.randn(num_paths, device=device),
            ).to(dtype) / math.sqrt(2.0 * num_paths)
            for gain, angle in zip(gains, aoas):
                steering = torch.exp(1j * math.pi * n * torch.sin(angle)) / math.sqrt(num_bs_ant)
                h = h + gain * steering
            user_channels.append(h)
        channels.append(torch.stack(user_channels, dim=0))
    return torch.stack(channels, dim=0)


def wideband_ofdm_channel(
    num_samples: int,
    num_users: int,
    num_bs_ant: int,
    num_paths: int,
    num_subcarriers: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Generate a simple frequency-selective channel with shape (N, S, K, Nt)."""
    delays = torch.rand(num_samples, num_users, num_paths, device=device)
    path_channels = sparse_geometric_mmwave_channel(
        num_samples=num_samples * num_paths,
        num_users=num_users,
        num_bs_ant=num_bs_ant,
        num_paths=1,
        device=device,
        dtype=dtype,
    ).reshape(num_samples, num_paths, num_users, num_bs_ant)
    subcarrier_idx = torch.arange(num_subcarriers, device=device, dtype=torch.float32)
    channels = []
    for sample_idx in range(num_samples):
        sample_subcarriers = []
        for sc in subcarrier_idx:
            phase = torch.exp(-1j * 2.0 * math.pi * sc * delays[sample_idx]).to(dtype)
            h_sc = (phase.unsqueeze(-1) * path_channels[sample_idx].transpose(0, 1)).sum(dim=1)
            sample_subcarriers.append(h_sc)
        channels.append(torch.stack(sample_subcarriers, dim=0))
    return torch.stack(channels, dim=0)

