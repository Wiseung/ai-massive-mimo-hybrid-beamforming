from __future__ import annotations

import torch

from beamforming.models.residual_beamformer import ResidualRZFBeamformer


def test_residual_beamformer_forward_shapes() -> None:
    batch, users, antennas = 4, 3, 8
    channel = torch.randn(batch, users, antennas) + 1j * torch.randn(batch, users, antennas)
    channel = channel.to(torch.complex64)
    channel_real = torch.stack((channel.real, channel.imag), dim=1)
    snr_db = torch.tensor([0.0, 5.0, 10.0, 15.0], dtype=torch.float32)
    model = ResidualRZFBeamformer(
        num_users=users,
        num_bs_ant=antennas,
        num_rf_chains=users,
        hidden_dims=[64],
        alpha_init=0.05,
    )
    outputs = model(channel_real, snr_db=snr_db, channel_complex=channel)
    assert outputs["precoder"].shape == (batch, antennas, users)
    assert outputs["base_precoder"].shape == (batch, antennas, users)
    assert outputs["delta_precoder"].shape == (batch, antennas, users)
    power = (torch.abs(outputs["precoder"]) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
