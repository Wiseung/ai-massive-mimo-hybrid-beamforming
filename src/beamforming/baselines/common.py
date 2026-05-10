"""Shared baseline evaluation helpers."""

from __future__ import annotations

import time
from typing import Any

import torch

from beamforming.baselines.dft_codebook import dft_hybrid_precoder
from beamforming.baselines.mrt import mrt_precoder
from beamforming.baselines.omp import omp_hybrid_precoder
from beamforming.baselines.rzf import rzf_precoder
from beamforming.baselines.upper_bound import fd_rzf_precoder, fd_zf_precoder
from beamforming.baselines.wmmse import wmmse_precoder
from beamforming.baselines.zf import zf_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr


def get_digital_precoder(
    method: str,
    channel: torch.Tensor,
    noise_var: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a digital precoder for a supported baseline method."""
    if method == "mrt":
        return mrt_precoder(channel)
    if method == "zf":
        return zf_precoder(channel)
    if method == "rzf":
        if noise_var is None:
            raise ValueError("noise_var is required for rzf.")
        return rzf_precoder(channel, noise_var=noise_var)
    if method == "fd_zf":
        return fd_zf_precoder(channel)
    if method == "fd_rzf":
        if noise_var is None:
            raise ValueError("noise_var is required for fd_rzf.")
        return fd_rzf_precoder(channel, noise_var=noise_var)
    if method == "wmmse":
        if noise_var is None:
            raise ValueError("noise_var is required for wmmse.")
        return wmmse_precoder(channel, noise_var=noise_var)
    raise ValueError(f"Unsupported digital precoder method: {method}")


def evaluate_baseline(
    method: str,
    channel: torch.Tensor,
    snr_db: float,
    num_rf_chains: int | None = None,
) -> dict[str, Any]:
    """Run one baseline and return precoder plus aggregate metrics."""
    start = time.perf_counter()
    analog = None
    digital = None
    noise_var = float(noise_variance_from_snr(snr_db).item())
    if method == "mrt":
        precoder = get_digital_precoder(method, channel)
    elif method == "zf":
        precoder = get_digital_precoder(method, channel)
    elif method == "rzf":
        precoder = get_digital_precoder(method, channel, noise_var=noise_var)
    elif method == "fd_zf":
        precoder = get_digital_precoder(method, channel)
    elif method == "fd_rzf":
        precoder = get_digital_precoder(method, channel, noise_var=noise_var)
    elif method == "wmmse":
        precoder = get_digital_precoder(method, channel, noise_var=noise_var)
    elif method == "dft":
        if num_rf_chains is None:
            raise ValueError("num_rf_chains is required for dft.")
        analog, digital, precoder = dft_hybrid_precoder(channel, num_rf_chains=num_rf_chains)
    elif method == "omp":
        if num_rf_chains is None:
            raise ValueError("num_rf_chains is required for omp.")
        analog, digital, precoder = omp_hybrid_precoder(channel, num_rf_chains=num_rf_chains)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    runtime = time.perf_counter() - start
    noise_var_tensor = noise_variance_from_snr(snr_db).to(channel.device)
    sum_rate = multi_user_downlink_sum_rate(channel, precoder, noise_var_tensor)
    return {
        "method": method,
        "precoder": precoder,
        "analog_precoder": analog,
        "digital_precoder": digital,
        "sum_rate": sum_rate,
        "se": sum_rate,
        "runtime_sec": runtime,
    }
