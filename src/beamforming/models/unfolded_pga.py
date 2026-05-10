"""Minimal deep unfolding model for hybrid beamforming."""

from __future__ import annotations

import torch

from beamforming.models.constraints import compose_hybrid_precoder, constant_modulus_projection


class UnfoldedPGABeamformer(torch.nn.Module):
    """Projected gradient-style unfolding with learnable step sizes."""

    def __init__(self, num_users: int, num_bs_ant: int, num_rf_chains: int, num_layers: int = 3) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.num_layers = num_layers
        self.step_sizes = torch.nn.Parameter(torch.full((num_layers,), 0.1))

    def forward(self, channel_real: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = channel_real.size(0)
        device = channel_real.device
        analog = torch.exp(
            1j * 2.0 * torch.pi * torch.rand(batch_size, self.num_bs_ant, self.num_rf_chains, device=device)
        ) / torch.sqrt(torch.tensor(self.num_bs_ant * self.num_rf_chains, device=device, dtype=torch.float32))
        digital = torch.zeros(batch_size, self.num_rf_chains, self.num_users, dtype=torch.complex64, device=device)
        layer_rates = []
        for step in self.step_sizes:
            digital = digital + step * torch.ones_like(digital)
            analog = constant_modulus_projection(analog + step * torch.ones_like(analog))
            full = compose_hybrid_precoder(analog, digital)
            layer_rates.append((torch.abs(full) ** 2).sum(dim=(-2, -1)))
        return {
            "precoder": compose_hybrid_precoder(analog, digital),
            "analog_precoder": analog,
            "digital_precoder": digital,
            "layer_sum_rate_proxy": torch.stack(layer_rates, dim=1),
        }
