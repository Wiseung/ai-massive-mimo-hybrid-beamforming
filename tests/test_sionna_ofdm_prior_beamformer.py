from __future__ import annotations

import torch

from beamforming.metrics.sum_rate import noise_variance_from_snr
from beamforming.models.sionna_ofdm_prior_beamformer import SionnaOFDMResidualRZFBeamformer, SionnaOFDMUnfoldedLiteBeamformer
from beamforming.utils.sionna_ofdm_training import build_baseline_precoder_stack


def _sample_channel(batch: int = 2, nsc: int = 4, k: int = 4, nt: int = 8) -> torch.Tensor:
    channel = torch.randn(batch, nsc, k, nt, dtype=torch.complex64)
    return (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0))


def test_sionna_ofdm_residual_rzf_shape_power_and_backward() -> None:
    model = SionnaOFDMResidualRZFBeamformer(num_users=4, num_bs_ant=8, hidden_dim=32, condition_on_snr=True)
    channel = _sample_channel()
    snr_db = torch.tensor([0.0, 10.0], dtype=torch.float32)
    outputs = model(channel, snr_db=snr_db)
    precoder = outputs["precoder"]
    assert precoder.shape == (2, 4, 8, 4)
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    loss = torch.abs(precoder).mean()
    loss.backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm().item() > 0 for param in model.parameters())


def test_sionna_ofdm_residual_rzf_init_not_collapse_from_rzf() -> None:
    channel = _sample_channel(batch=2, nsc=3, k=4, nt=8)
    snr_db = torch.tensor([10.0, 10.0], dtype=torch.float32)
    noise_var = noise_variance_from_snr(snr_db)
    model = SionnaOFDMResidualRZFBeamformer(num_users=4, num_bs_ant=8, hidden_dim=16, alpha_init=0.0, learnable_alpha=False)
    outputs = model(channel, snr_db=snr_db)
    base = build_baseline_precoder_stack("rzf", channel, noise_var)
    assert torch.allclose(outputs["precoder"], base, atol=1e-4)


def test_sionna_ofdm_unfolded_lite_forward_runs() -> None:
    model = SionnaOFDMUnfoldedLiteBeamformer(
        num_users=4,
        num_bs_ant=8,
        hidden_dim=32,
        num_layers=2,
        init_method="wmmse_iter_1",
        condition_on_snr=True,
    )
    channel = _sample_channel(batch=2, nsc=4, k=4, nt=8)
    snr_db = torch.tensor([5.0, 15.0], dtype=torch.float32)
    outputs = model(channel, snr_db=snr_db)
    precoder = outputs["precoder"]
    assert precoder.shape == (2, 4, 8, 4)
    assert torch.isfinite(precoder.real).all()
    assert torch.isfinite(precoder.imag).all()
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    assert outputs["layer_sum_rates"].shape == (2, 2)
