from __future__ import annotations

import torch

from beamforming.utils.complex_ops import (
    complex_matmul,
    complex_norm,
    complex_to_real,
    hermitian,
    normalize_power,
    real_to_complex,
)


def test_complex_round_trip() -> None:
    x = torch.randn(3, 4, 5, dtype=torch.complex64)
    x = x + 1j * torch.randn_like(x)
    restored = real_to_complex(complex_to_real(x))
    assert torch.allclose(x, restored)


def test_hermitian_matches_manual() -> None:
    x = torch.randn(2, 3, 4, dtype=torch.complex64) + 1j * torch.randn(2, 3, 4, dtype=torch.complex64)
    expected = x.transpose(-2, -1).conj()
    assert torch.allclose(hermitian(x), expected)


def test_complex_matmul_matches_torch() -> None:
    a = torch.randn(2, 3, 4, dtype=torch.complex64) + 1j * torch.randn(2, 3, 4, dtype=torch.complex64)
    b = torch.randn(2, 4, 5, dtype=torch.complex64) + 1j * torch.randn(2, 4, 5, dtype=torch.complex64)
    assert torch.allclose(complex_matmul(a, b), a @ b)


def test_complex_norm_and_power_normalization() -> None:
    x = torch.randn(2, 4, 3, dtype=torch.complex64) + 1j * torch.randn(2, 4, 3, dtype=torch.complex64)
    norm = complex_norm(x, dim=(-2, -1))
    assert torch.all(norm > 0)
    y = normalize_power(x, target_power=2.0)
    power = (torch.abs(y) ** 2).sum(dim=(-2, -1))
    assert torch.allclose(power, torch.full_like(power, 2.0), atol=1e-4)
