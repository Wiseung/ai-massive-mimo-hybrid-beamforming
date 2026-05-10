"""Shared baseline evaluation helpers."""

from __future__ import annotations

import time
from typing import Any

import torch

from beamforming.baselines.dft_codebook import dft_hybrid_precoder
from beamforming.baselines.mrt import mrt_precoder
from beamforming.baselines.omp import omp_hybrid_precoder
from beamforming.baselines.rzf import rzf_precoder
from beamforming.baselines.zf import zf_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr


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
    if method == "mrt":
        precoder = mrt_precoder(channel)
    elif method == "zf":
        precoder = zf_precoder(channel)
    elif method == "rzf":
        precoder = rzf_precoder(channel, noise_var=float(noise_variance_from_snr(snr_db).item()))
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
    noise_var = noise_variance_from_snr(snr_db).to(channel.device)
    sum_rate = multi_user_downlink_sum_rate(channel, precoder, noise_var)
    return {
        "method": method,
        "precoder": precoder,
        "analog_precoder": analog,
        "digital_precoder": digital,
        "sum_rate": sum_rate,
        "se": sum_rate,
        "runtime_sec": runtime,
    }
