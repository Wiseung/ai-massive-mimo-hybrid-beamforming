"""Unfolded WMMSE-lite refinement with learnable projected updates."""

from __future__ import annotations

import torch

from beamforming.baselines.rzf import rzf_precoder
from beamforming.baselines.wmmse import wmmse_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr
from beamforming.models.constraints import power_normalization


class UnfoldedWMMSELiteBeamformer(torch.nn.Module):
    """Refine a structured initial precoder with learnable projected gradient steps."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        num_rf_chains: int,
        num_layers: int = 3,
        alpha_init: float = 0.05,
        init_method: str = "rzf",
        init_wmmse_iters: int = 2,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.num_layers = num_layers
        self.init_method = init_method
        self.init_wmmse_iters = init_wmmse_iters
        init = torch.full((num_layers,), float(alpha_init), dtype=torch.float32)
        self.log_steps = torch.nn.Parameter(torch.log(torch.expm1(init) + 1e-8))

    def _steps(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_steps)

    def _resolve_init_wmmse_iters(self) -> int | None:
        if self.init_method == "rzf":
            return None
        if self.init_method == "wmmse":
            return self.init_wmmse_iters
        if self.init_method.startswith("wmmse_iter_"):
            suffix = self.init_method.removeprefix("wmmse_iter_")
            try:
                max_iter = int(suffix)
            except ValueError as exc:
                raise ValueError(f"Unsupported init_method: {self.init_method}") from exc
            if max_iter <= 0:
                raise ValueError(f"Unsupported init_method: {self.init_method}")
            return max_iter
        raise ValueError(f"Unsupported init_method: {self.init_method}")

    def _init_precoder(self, channel_complex: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
        if self.init_method == "rzf":
            return rzf_precoder(channel_complex, noise_var=noise_var)
        max_iter = self._resolve_init_wmmse_iters()
        if max_iter is None:
            raise ValueError(f"Unsupported init_method: {self.init_method}")
        return wmmse_precoder(channel_complex, noise_var=noise_var, max_iter=max_iter)

    def forward(
        self,
        channel_real: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        channel_complex: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if snr_db is None:
            raise ValueError("snr_db must be provided for UnfoldedWMMSELiteBeamformer.")
        if channel_complex is None:
            raise ValueError("channel_complex must be provided for UnfoldedWMMSELiteBeamformer.")

        noise_var = noise_variance_from_snr(snr_db).to(channel_real.device)
        precoder = self._init_precoder(channel_complex, noise_var=noise_var)
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
            "delta_precoder": precoder - base_precoder,
            "layer_sum_rates": torch.stack(layer_sum_rates, dim=1) if layer_sum_rates else torch.empty(0, device=channel_real.device),
            "step_sizes": self._steps().to(channel_real.device),
            "init_method": self.init_method,
        }
