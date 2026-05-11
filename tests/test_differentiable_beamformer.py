from __future__ import annotations

import torch

from beamforming.models.differentiable_beamformer import TinyNeuralBeamformer


def test_differentiable_beamformer_output_shape_and_power() -> None:
    model = TinyNeuralBeamformer(num_users=4, num_bs_ant=8, hidden_dim=32)
    channel = torch.randn(3, 4, 8, dtype=torch.complex64)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0))
    precoder = model(channel)
    assert precoder.shape == (3, 8, 4)
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
    assert torch.isfinite(precoder.real).all()
    assert torch.isfinite(precoder.imag).all()


def test_differentiable_beamformer_backward_has_gradients() -> None:
    model = TinyNeuralBeamformer(num_users=4, num_bs_ant=8, hidden_dim=32)
    channel = torch.randn(2, 5, 4, 8, dtype=torch.complex64)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0))
    precoder = model(channel)
    loss = (torch.abs(precoder) ** 2).mean()
    loss.backward()
    nonzero_grad = False
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        nonzero_grad = nonzero_grad or bool(param.grad.norm().item() > 0)
    assert nonzero_grad


def test_differentiable_beamformer_supports_snr_conditioning() -> None:
    model = TinyNeuralBeamformer(num_users=4, num_bs_ant=8, hidden_dim=32, condition_on_snr=True)
    channel = torch.randn(2, 3, 4, 8, dtype=torch.complex64)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0))
    snr_db = torch.tensor([0.0, 10.0], dtype=torch.float32)
    precoder = model(channel, snr_db=snr_db)
    assert precoder.shape == (2, 3, 8, 4)
    power = (torch.abs(precoder) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4)
