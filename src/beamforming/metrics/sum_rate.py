"""Sum-rate and spectral-efficiency metrics."""

from __future__ import annotations

import math

import torch

from beamforming.utils.complex_ops import hermitian


def noise_variance_from_snr(snr_db: float | torch.Tensor, signal_power: float = 1.0) -> torch.Tensor:
    """Convert SNR in dB to noise variance assuming unit signal power."""
    snr_tensor = torch.as_tensor(snr_db, dtype=torch.float32)
    return torch.as_tensor(signal_power, dtype=torch.float32) / torch.pow(10.0, snr_tensor / 10.0)


def _effective_channel(channel: torch.Tensor, precoder: torch.Tensor) -> torch.Tensor:
    return torch.matmul(channel, precoder)


def single_user_mimo_rate(channel: torch.Tensor, precoder: torch.Tensor, noise_var: float | torch.Tensor) -> torch.Tensor:
    """Achievable rate for a single-user MIMO link with Gaussian signaling."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    if precoder.ndim == 2:
        precoder = precoder.unsqueeze(0)
    noise = torch.as_tensor(noise_var, dtype=torch.float32, device=channel.device)
    heff = _effective_channel(channel, precoder)
    rx_ant = channel.size(-2)
    eye = torch.eye(rx_ant, dtype=channel.dtype, device=channel.device).unsqueeze(0).expand(channel.size(0), -1, -1)
    cov = eye + torch.matmul(heff, hermitian(heff)) / noise.view(-1, 1, 1)
    sign, logdet = torch.linalg.slogdet(cov)
    if not torch.all(sign > 0):
        raise RuntimeError("Covariance matrix must be positive definite.")
    return logdet / math.log(2.0)


def multi_user_downlink_sum_rate(channel: torch.Tensor, precoder: torch.Tensor, noise_var: float | torch.Tensor) -> torch.Tensor:
    """Downlink MU-MISO sum-rate with one stream per user."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    if precoder.ndim == 2:
        precoder = precoder.unsqueeze(0)
    if channel.ndim != 3 or precoder.ndim != 3:
        raise ValueError("channel and precoder must have shape (B, K, Nt) and (B, Nt, K).")

    noise = torch.as_tensor(noise_var, dtype=torch.float32, device=channel.device)
    if noise.ndim == 0:
        noise = noise.repeat(channel.size(0))

    heff = torch.matmul(channel, precoder)
    signal = torch.abs(torch.diagonal(heff, dim1=-2, dim2=-1)) ** 2
    total = torch.abs(heff) ** 2
    interference = total.sum(dim=-1) - signal
    sinr = signal / (interference + noise.unsqueeze(-1))
    rates = torch.log2(1.0 + sinr)
    return rates.sum(dim=-1)


def spectral_efficiency(channel: torch.Tensor, precoder: torch.Tensor, noise_var: float | torch.Tensor) -> torch.Tensor:
    """Alias for per-sample downlink sum-rate in bit/s/Hz."""
    return multi_user_downlink_sum_rate(channel, precoder, noise_var)
