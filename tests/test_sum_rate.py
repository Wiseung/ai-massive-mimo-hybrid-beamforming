from __future__ import annotations

import torch

from beamforming.baselines.mrt import mrt_precoder
from beamforming.baselines.zf import zf_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr


def _random_channel(batch: int = 256, users: int = 4, antennas: int = 16) -> torch.Tensor:
    real = torch.randn(batch, users, antennas)
    imag = torch.randn(batch, users, antennas)
    return torch.complex(real, imag) / torch.sqrt(torch.tensor(2.0))


def test_sum_rate_increases_with_snr() -> None:
    channel = _random_channel()
    precoder = zf_precoder(channel)
    low = multi_user_downlink_sum_rate(channel, precoder, noise_variance_from_snr(-5.0)).mean()
    high = multi_user_downlink_sum_rate(channel, precoder, noise_variance_from_snr(20.0)).mean()
    assert high > low


def test_zf_outperforms_mrt_at_high_snr() -> None:
    channel = _random_channel(batch=512)
    noise_var = noise_variance_from_snr(25.0)
    mrt = multi_user_downlink_sum_rate(channel, mrt_precoder(channel), noise_var).mean()
    zf = multi_user_downlink_sum_rate(channel, zf_precoder(channel), noise_var).mean()
    assert zf > mrt
