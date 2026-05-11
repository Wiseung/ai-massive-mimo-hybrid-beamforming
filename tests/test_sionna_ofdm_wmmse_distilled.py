from __future__ import annotations

import torch

from beamforming.models.sionna_ofdm_prior_beamformer import SionnaOFDMResidualWMMSEDistilledBeamformer


def _sample_channel(batch: int = 2, nsc: int = 4, k: int = 4, nt: int = 8) -> torch.Tensor:
    channel = torch.randn(batch, nsc, k, nt, dtype=torch.complex64)
    return (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0))


def test_sionna_ofdm_residual_wmmse_distill_shape_power_and_backward() -> None:
    model = SionnaOFDMResidualWMMSEDistilledBeamformer(
        num_users=4,
        num_bs_ant=8,
        hidden_dim=32,
        condition_on_snr=True,
        teacher_iter=5,
    )
    channel = _sample_channel()
    snr_db = torch.tensor([0.0, 10.0], dtype=torch.float32)
    outputs = model(channel, snr_db=snr_db)
    precoder = outputs["precoder"]
    assert precoder.shape == (2, 4, 8, 4)
    assert torch.isfinite(precoder.real).all()
    assert torch.isfinite(precoder.imag).all()
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    assert outputs["teacher_used_during_inference"] is False
    assert outputs["teacher_iter"] == 5
    loss = torch.abs(precoder).mean()
    loss.backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm().item() > 0 for param in model.parameters())


def test_sionna_ofdm_residual_wmmse_distill_inference_does_not_call_teacher() -> None:
    model = SionnaOFDMResidualWMMSEDistilledBeamformer(num_users=4, num_bs_ant=8, hidden_dim=16, teacher_iter=5)
    channel = _sample_channel(batch=1, nsc=2, k=4, nt=8)
    snr_db = torch.tensor([10.0], dtype=torch.float32)
    outputs = model(channel, snr_db=snr_db)
    assert "base_precoder" in outputs
    assert "delta_precoder" in outputs
    assert outputs["teacher_used_during_training"] is True
    assert outputs["teacher_used_during_inference"] is False
    assert outputs["inference_inputs"] == ["H_f", "F_rzf", "snr_db"]
