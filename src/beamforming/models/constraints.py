"""Hybrid precoder constraints and projections."""

from __future__ import annotations

import math

import torch

from beamforming.utils.complex_ops import normalize_power


def power_normalization(precoder: torch.Tensor, total_power: float = 1.0) -> torch.Tensor:
    """Normalize the precoder to meet a total transmit-power constraint."""
    return normalize_power(precoder, target_power=total_power)


def constant_modulus_projection(analog_precoder: torch.Tensor) -> torch.Tensor:
    """Project analog precoder entries onto the unit-modulus manifold."""
    phase = torch.angle(analog_precoder)
    projected = torch.exp(1j * phase)
    return projected / math.sqrt(analog_precoder.size(-2))


def compose_hybrid_precoder(analog_precoder: torch.Tensor, digital_precoder: torch.Tensor, total_power: float = 1.0) -> torch.Tensor:
    """Compose analog and digital components into a normalized hybrid precoder."""
    full_precoder = analog_precoder @ digital_precoder
    return power_normalization(full_precoder, total_power=total_power)
