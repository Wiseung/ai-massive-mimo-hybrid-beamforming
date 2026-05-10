"""Teacher target generation for supervised beamformer pretraining."""

from __future__ import annotations

import torch

from beamforming.baselines.mrt import mrt_precoder
from beamforming.baselines.rzf import rzf_precoder
from beamforming.baselines.zf import zf_precoder
from beamforming.metrics.sum_rate import noise_variance_from_snr


def generate_mrt_target(channel: torch.Tensor) -> torch.Tensor:
    """Generate MRT teacher precoders."""
    return mrt_precoder(channel)


def generate_zf_target(channel: torch.Tensor) -> torch.Tensor:
    """Generate ZF teacher precoders."""
    return zf_precoder(channel)


def generate_rzf_target(channel: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Generate per-sample RZF teacher precoders with matched noise variance."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    noise_var = noise_variance_from_snr(snr_db).view(-1)
    precoders = []
    for idx in range(channel.size(0)):
        precoders.append(rzf_precoder(channel[idx : idx + 1], noise_var=float(noise_var[idx].item())).squeeze(0))
    return torch.stack(precoders, dim=0)


def generate_mixed_rzf_zf_target(channel: torch.Tensor, snr_db: torch.Tensor, threshold_db: float = 10.0) -> torch.Tensor:
    """Use RZF at low SNR and ZF at high SNR."""
    rzf = generate_rzf_target(channel, snr_db)
    zf = generate_zf_target(channel)
    mask = (snr_db.view(-1, 1, 1) >= threshold_db).to(zf.device)
    return torch.where(mask, zf, rzf)


def generate_best_baseline_target(channel: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Pick per-sample target between RZF and ZF using achieved sum-rate."""
    from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate

    rzf = generate_rzf_target(channel, snr_db)
    zf = generate_zf_target(channel)
    noise_var = noise_variance_from_snr(snr_db).to(channel.device)
    rzf_rate = multi_user_downlink_sum_rate(channel, rzf, noise_var)
    zf_rate = multi_user_downlink_sum_rate(channel, zf, noise_var)
    mask = (zf_rate >= rzf_rate).view(-1, 1, 1)
    return torch.where(mask, zf, rzf)


def get_teacher_target(channel: torch.Tensor, snr_db: torch.Tensor, teacher: str) -> torch.Tensor:
    """Dispatch teacher target generation by name."""
    if teacher == "mrt":
        return generate_mrt_target(channel)
    if teacher == "zf":
        return generate_zf_target(channel)
    if teacher == "rzf":
        return generate_rzf_target(channel, snr_db)
    if teacher == "mixed_rzf_zf":
        return generate_mixed_rzf_zf_target(channel, snr_db)
    if teacher == "best_of_rzf_zf":
        return generate_best_baseline_target(channel, snr_db)
    raise ValueError(f"Unsupported teacher: {teacher}")
