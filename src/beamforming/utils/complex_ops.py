"""Complex tensor helpers using PyTorch native complex dtypes."""

from __future__ import annotations

import math

import torch


def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """Split a complex tensor into stacked real and imaginary parts."""
    if not torch.is_complex(x):
        raise TypeError("complex_to_real expects a complex tensor.")
    return torch.stack((x.real, x.imag), dim=1)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Merge stacked real and imaginary channels back to a complex tensor."""
    if x.ndim < 2 or x.size(1) != 2:
        raise ValueError("real_to_complex expects shape (B, 2, ...).")
    real = x[:, 0, ...]
    imag = x[:, 1, ...]
    if real.dtype in (torch.float16, torch.bfloat16):
        real = real.float()
        imag = imag.float()
    return torch.complex(real, imag)


def hermitian(x: torch.Tensor) -> torch.Tensor:
    """Conjugate transpose across the last two dimensions."""
    return x.transpose(-2, -1).conj()


def complex_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiply complex tensors."""
    return torch.matmul(a, b)


def complex_norm(x: torch.Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> torch.Tensor:
    """L2 norm for complex tensors."""
    power = torch.abs(x) ** 2
    if dim is None:
        return torch.sqrt(power.sum())
    return torch.sqrt(power.sum(dim=dim, keepdim=keepdim).clamp_min(1e-12))


def normalize_power(x: torch.Tensor, target_power: float = 1.0, dim: tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    """Scale a complex tensor to the target average Frobenius power."""
    if target_power <= 0:
        raise ValueError("target_power must be positive.")
    current_power = (torch.abs(x) ** 2).sum(dim=dim, keepdim=True).clamp_min(1e-12)
    scale = math.sqrt(target_power) / torch.sqrt(current_power)
    return x * scale
