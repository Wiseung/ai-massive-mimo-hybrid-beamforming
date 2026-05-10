"""DFT codebook analog beam selection baseline."""

from __future__ import annotations

import math

import torch

from beamforming.models.constraints import compose_hybrid_precoder, constant_modulus_projection


def ula_dft_codebook(num_bs_ant: int, num_beams: int | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    """Construct a unitary DFT codebook for a ULA."""
    num_beams = num_beams or num_bs_ant
    n = torch.arange(num_bs_ant, device=device, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(num_beams, device=device, dtype=torch.float32).unsqueeze(0)
    codebook = torch.exp(-1j * 2.0 * math.pi * n * k / num_beams) / math.sqrt(num_bs_ant)
    return codebook.to(torch.complex64)


def dft_hybrid_precoder(channel: torch.Tensor, num_rf_chains: int, total_power: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick strongest DFT beams and use matched digital weights."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    _batch_size, _num_users, num_bs_ant = channel.shape
    codebook = ula_dft_codebook(num_bs_ant, device=channel.device)
    gains = torch.abs(channel @ codebook) ** 2
    beam_scores = gains.sum(dim=1)
    indices = torch.topk(beam_scores, k=min(num_rf_chains, codebook.size(1)), dim=-1).indices
    analog = codebook.unsqueeze(0).expand(channel.size(0), -1, -1).gather(
        2, indices.unsqueeze(1).expand(-1, num_bs_ant, -1)
    )
    analog = constant_modulus_projection(analog)
    reduced = channel @ analog
    digital = reduced.transpose(-2, -1).conj()
    full = compose_hybrid_precoder(analog, digital, total_power=total_power)
    return analog, digital, full
