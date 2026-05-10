import torch

from beamforming.models.unfolded_rzf import UnfoldedRZFBeamformer


def test_unfolded_rzf_output_shape_and_layers() -> None:
    torch.manual_seed(0)
    model = UnfoldedRZFBeamformer(num_users=4, num_bs_ant=8, num_rf_chains=4, num_layers=3, alpha_init=0.05)
    channel = torch.randn(2, 4, 8, dtype=torch.complex64)
    channel_real = torch.stack([channel.real, channel.imag], dim=1)
    snr_db = torch.tensor([0.0, 10.0], dtype=torch.float32)
    outputs = model(channel_real, snr_db=snr_db, channel_complex=channel)
    assert outputs["precoder"].shape == (2, 8, 4)
    assert outputs["layer_sum_rates"].shape == (2, 3)
    power = (torch.abs(outputs["precoder"]) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
