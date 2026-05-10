import torch

from beamforming.baselines.common import get_digital_precoder
from beamforming.baselines.mrt import mrt_precoder
from beamforming.baselines.wmmse import wmmse_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate


def test_wmmse_shape_and_power() -> None:
    torch.manual_seed(0)
    channel = torch.randn(3, 4, 8, dtype=torch.complex64)
    precoder = wmmse_precoder(channel, noise_var=0.1, max_iter=10)
    assert precoder.shape == (3, 8, 4)
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)


def test_wmmse_sum_rate_is_finite() -> None:
    torch.manual_seed(1)
    channel = torch.randn(2, 4, 8, dtype=torch.complex64)
    precoder = wmmse_precoder(channel, noise_var=0.1, max_iter=8)
    rate = multi_user_downlink_sum_rate(channel, precoder, noise_var=0.1)
    assert torch.isfinite(rate).all()


def test_wmmse_not_worse_than_mrt_on_small_case() -> None:
    torch.manual_seed(2)
    channel = torch.randn(4, 3, 6, dtype=torch.complex64)
    mrt = mrt_precoder(channel)
    wmmse = wmmse_precoder(channel, noise_var=0.1, max_iter=12)
    mrt_rate = multi_user_downlink_sum_rate(channel, mrt, noise_var=0.1).mean()
    wmmse_rate = multi_user_downlink_sum_rate(channel, wmmse, noise_var=0.1).mean()
    assert float(wmmse_rate) >= float(mrt_rate) - 1e-3


def test_wmmse_near_singular_channel_is_finite() -> None:
    torch.manual_seed(3)
    base = torch.randn(2, 1, 8, dtype=torch.complex64)
    channel = base.repeat(1, 4, 1)
    channel[:, 1:] = channel[:, 1:] + 1e-4 * torch.randn_like(channel[:, 1:])
    precoder = wmmse_precoder(channel, noise_var=1e-3, max_iter=20)
    assert torch.isfinite(precoder.real).all()
    assert torch.isfinite(precoder.imag).all()
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)


def test_wmmse_low_and_high_snr_batch_are_finite() -> None:
    torch.manual_seed(4)
    channel = torch.randn(5, 4, 8, dtype=torch.complex64)
    for noise_var in (10.0, 1e-4):
        precoder = wmmse_precoder(channel, noise_var=noise_var, max_iter=15)
        rate = multi_user_downlink_sum_rate(channel, precoder, noise_var=noise_var)
        assert torch.isfinite(rate).all()
        assert not torch.isnan(precoder).any()
        assert not torch.isinf(precoder).any()


def test_wmmse_more_iterations_do_not_catastrophically_degrade() -> None:
    torch.manual_seed(5)
    channel = torch.randn(3, 4, 8, dtype=torch.complex64)
    precoder_short = wmmse_precoder(channel, noise_var=0.1, max_iter=2)
    precoder_long = wmmse_precoder(channel, noise_var=0.1, max_iter=20)
    rate_short = multi_user_downlink_sum_rate(channel, precoder_short, noise_var=0.1).mean()
    rate_long = multi_user_downlink_sum_rate(channel, precoder_long, noise_var=0.1).mean()
    assert float(rate_long) >= float(rate_short) - 0.2


def test_wmmse_iter_alias_matches_direct_call() -> None:
    torch.manual_seed(6)
    channel = torch.randn(2, 4, 8, dtype=torch.complex64)
    direct = wmmse_precoder(channel, noise_var=0.1, max_iter=5)
    alias = get_digital_precoder("wmmse_iter_5", channel, noise_var=0.1)
    assert torch.allclose(alias, direct, atol=1e-5, rtol=1e-5)
