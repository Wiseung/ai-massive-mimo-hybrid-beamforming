"""Simple OMP-style sparse hybrid beamforming baseline."""

from __future__ import annotations

import torch

from beamforming.baselines.dft_codebook import ula_dft_codebook
from beamforming.models.constraints import compose_hybrid_precoder


def omp_hybrid_precoder(channel: torch.Tensor, num_rf_chains: int, total_power: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy sparse beam selection with a DFT dictionary."""
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    batch_size, num_users, num_bs_ant = channel.shape
    dictionary = ula_dft_codebook(num_bs_ant, device=channel.device)
    analog_list = []
    digital_list = []
    full_list = []
    for batch_idx in range(batch_size):
        residual = channel[batch_idx].transpose(-2, -1).conj()
        selected: list[int] = []
        for _ in range(min(num_rf_chains, dictionary.size(1))):
            projections = torch.abs(dictionary.transpose(-2, -1).conj() @ residual).sum(dim=-1)
            best_idx = int(torch.argmax(projections).item())
            if best_idx not in selected:
                selected.append(best_idx)
            analog = dictionary[:, selected]
            digital = torch.linalg.pinv(analog) @ channel[batch_idx].transpose(-2, -1).conj()
            residual = channel[batch_idx].transpose(-2, -1).conj() - analog @ digital
        analog = dictionary[:, selected]
        digital = torch.linalg.pinv(analog) @ channel[batch_idx].transpose(-2, -1).conj()
        full = compose_hybrid_precoder(analog, digital, total_power=total_power)
        analog_list.append(analog)
        digital_list.append(digital)
        full_list.append(full)
    return torch.stack(analog_list), torch.stack(digital_list), torch.stack(full_list)
