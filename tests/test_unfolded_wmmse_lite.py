import torch

from beamforming.models.unfolded_wmmse_lite import UnfoldedWMMSELiteBeamformer


def test_unfolded_wmmse_lite_output_shape_and_layers() -> None:
    torch.manual_seed(0)
    model = UnfoldedWMMSELiteBeamformer(
        num_users=4,
        num_bs_ant=8,
        num_rf_chains=4,
        num_layers=3,
        alpha_init=0.05,
        init_method="rzf",
    )
    channel = torch.randn(2, 4, 8, dtype=torch.complex64)
    channel_real = torch.stack([channel.real, channel.imag], dim=1)
    snr_db = torch.tensor([0.0, 10.0], dtype=torch.float32)
    outputs = model(channel_real, snr_db=snr_db, channel_complex=channel)
    assert outputs["precoder"].shape == (2, 8, 4)
    assert outputs["base_precoder"].shape == (2, 8, 4)
    assert outputs["delta_precoder"].shape == (2, 8, 4)
    assert outputs["layer_sum_rates"].shape == (2, 3)
    power = (torch.abs(outputs["precoder"]) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)


def test_unfolded_wmmse_lite_supports_wmmse_iter_alias_init() -> None:
    torch.manual_seed(1)
    model = UnfoldedWMMSELiteBeamformer(
        num_users=4,
        num_bs_ant=8,
        num_rf_chains=4,
        num_layers=2,
        alpha_init=0.03,
        init_method="wmmse_iter_2",
    )
    channel = torch.randn(2, 4, 8, dtype=torch.complex64)
    channel_real = torch.stack([channel.real, channel.imag], dim=1)
    snr_db = torch.tensor([5.0, 15.0], dtype=torch.float32)
    outputs = model(channel_real, snr_db=snr_db, channel_complex=channel)
    assert outputs["precoder"].shape == (2, 8, 4)
    assert outputs["layer_sum_rates"].shape == (2, 2)
