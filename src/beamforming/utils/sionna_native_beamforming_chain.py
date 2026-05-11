"""Helpers for inserting project precoders into the Sionna-native OFDM chain."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import torch

from beamforming.baselines.common import get_digital_precoder
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device


def build_frequency_domain_channel(
    batch_size: int,
    num_subcarriers: int,
    num_users: int,
    num_bs_ant: int,
    device: torch.device,
    resource_grid: Any | None = None,
    noise_var: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return H_f with shape (B, Nsc, K, Nt)."""
    sionna_device = resolve_sionna_device(device)
    OFDMChannel, _, _ = load_component("OFDMChannel")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    if resource_grid is not None and OFDMChannel is not None and RayleighBlockFading is not None:
        try:
            channel_model = RayleighBlockFading(
                num_rx=num_users,
                num_rx_ant=1,
                num_tx=1,
                num_tx_ant=num_bs_ant,
                device=sionna_device,
            )
            channel = OFDMChannel(channel_model, resource_grid, return_channel=True, device=sionna_device)
            dummy_x = torch.zeros(
                batch_size,
                1,
                num_bs_ant,
                resource_grid.num_ofdm_symbols,
                resource_grid.fft_size,
                dtype=torch.complex64,
                device=device,
            )
            _, h_freq = channel(dummy_x, no=torch.full((batch_size, num_users, 1), float(noise_var or 0.0), device=device))
            h_freq = h_freq.squeeze(2).squeeze(3).squeeze(3).permute(0, 3, 1, 2).contiguous()
            return h_freq, {
                "used_sionna_channel_tensor": True,
                "fallback_used": False,
                "notes": ["Extracted frequency-domain channel tensor from real Sionna OFDMChannel."],
            }
        except Exception as exc:  # pragma: no cover - optional runtime path
            pass

    real = torch.randn(batch_size, num_subcarriers, num_users, num_bs_ant, device=device)
    imag = torch.randn(batch_size, num_subcarriers, num_users, num_bs_ant, device=device)
    h_f = (real + 1j * imag) / math.sqrt(2.0)
    return h_f.to(torch.complex64), {
        "used_sionna_channel_tensor": False,
        "fallback_used": True,
        "notes": ["Used synthetic Rayleigh frequency-domain H_f fallback because direct Sionna channel extraction was unavailable."],
    }


def compute_project_precoder_per_subcarrier(
    method: str,
    channel_f: torch.Tensor,
    noise_var: float | torch.Tensor,
) -> torch.Tensor:
    """Compute F_f with shape (B, Nsc, Nt, K)."""
    precoders = []
    for sc in range(channel_f.size(1)):
        precoders.append(get_digital_precoder(method, channel_f[:, sc, :, :], noise_var=noise_var))
    return torch.stack(precoders, dim=1)


def apply_precoder_to_resource_grid(stream_symbols: torch.Tensor, precoder_f: torch.Tensor) -> torch.Tensor:
    """Apply per-subcarrier precoding.

    stream_symbols: (B, Nsc, K)
    precoder_f: (B, Nsc, Nt, K)
    returns: (B, Nt, Nsc)
    """
    tx = torch.matmul(precoder_f, stream_symbols.unsqueeze(-1)).squeeze(-1)
    return tx.transpose(1, 2).contiguous()


def evaluate_ofdm_beamforming_outputs(
    channel_f: torch.Tensor,
    precoder_f: torch.Tensor,
    stream_symbols: torch.Tensor,
    noise_var: float,
) -> dict[str, float]:
    """Compute proxy metrics for the beamformed OFDM chain."""
    heff = torch.matmul(channel_f, precoder_f)
    diag = torch.diagonal(heff, dim1=-2, dim2=-1)
    signal_power = torch.abs(diag) ** 2
    total_power = torch.abs(heff) ** 2
    interference_power = total_power.sum(dim=-1) - signal_power
    sinr = signal_power / (interference_power + noise_var)
    noisy_streams = diag * stream_symbols
    mse = torch.mean(torch.abs(noisy_streams - stream_symbols) ** 2).item()
    power_norm = torch.mean((torch.abs(precoder_f) ** 2).sum(dim=(-2, -1))).item()
    power_violation = torch.mean(torch.abs((torch.abs(precoder_f) ** 2).sum(dim=(-2, -1)) - 1.0)).item()
    sinr_mean = sinr.mean().clamp_min(1e-12)
    sum_rate = torch.log2(1.0 + sinr).sum(dim=-1).mean().item()
    return {
        "ber_if_available": None,
        "symbol_mse": float(mse),
        "effective_sinr_db": float(10.0 * torch.log10(sinr_mean).item()),
        "approximate_sum_rate": float(sum_rate),
        "power_norm": float(power_norm),
        "power_violation": float(power_violation),
    }


def time_function(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = perf_counter()
    out = fn(*args, **kwargs)
    return out, (perf_counter() - start) * 1000.0
