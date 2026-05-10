import torch

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
