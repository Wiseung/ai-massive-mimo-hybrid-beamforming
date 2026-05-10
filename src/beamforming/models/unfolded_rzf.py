"""Unfolded RZF refinement with learnable step sizes."""

from __future__ import annotations

import torch

from beamforming.baselines.rzf import rzf_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr
from beamforming.models.constraints import power_normalization


class UnfoldedRZFBeamformer(torch.nn.Module):
    """Refine an RZF precoder with a few learnable projected gradient steps."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        num_rf_chains: int,
        num_layers: int = 3,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.num_layers = num_layers
        init = torch.full((num_layers,), float(alpha_init), dtype=torch.float32)
        self.log_steps = torch.nn.Parameter(torch.log(torch.expm1(init) + 1e-8))

    def _steps(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_steps)

    def forward(
        self,
        channel_real: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        channel_complex: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if snr_db is None:
            raise ValueError("snr_db must be provided for UnfoldedRZFBeamformer.")
        if channel_complex is None:
            raise ValueError("channel_complex must be provided for UnfoldedRZFBeamformer.")

        noise_var = noise_variance_from_snr(snr_db).to(channel_real.device)
        precoder = rzf_precoder(channel_complex, noise_var=noise_var)
        base_precoder = precoder.detach().clone()
        layer_sum_rates: list[torch.Tensor] = []

        for step in self._steps():
            with torch.enable_grad():
                work = precoder.detach().clone().requires_grad_(True)
                rate = multi_user_downlink_sum_rate(channel_complex, work, noise_var)
                objective = rate.mean()
                grad = torch.autograd.grad(objective, work, retain_graph=False, create_graph=False)[0]
            precoder = power_normalization(work + step.to(channel_real.device) * grad)
            with torch.no_grad():
                layer_sum_rates.append(multi_user_downlink_sum_rate(channel_complex, precoder, noise_var))

        return {
            "precoder": precoder,
            "base_precoder": base_precoder,
            "layer_sum_rates": torch.stack(layer_sum_rates, dim=1) if layer_sum_rates else torch.empty(0, device=channel_real.device),
            "step_sizes": self._steps().to(channel_real.device),
        }
