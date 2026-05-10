"""Residual refinement beamformer around a classical digital prior."""

from __future__ import annotations

import math

import torch

from beamforming.baselines.common import get_digital_precoder
from beamforming.baselines.rzf import rzf_precoder
from beamforming.metrics.sum_rate import noise_variance_from_snr
from beamforming.models.constraints import power_normalization
from beamforming.utils.complex_ops import complex_to_real, real_to_complex


class ResidualRZFBeamformer(torch.nn.Module):
    """Predict a residual on top of a classical digital precoder."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        num_rf_chains: int,
        base_method: str = "rzf",
        condition_on_snr: bool = True,
        base_channels: int = 32,
        pool_factor: int = 2,
        hidden_dims: list[int] | None = None,
        learnable_alpha: bool = True,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.base_method = base_method
        self.condition_on_snr = condition_on_snr
        self.learnable_alpha = learnable_alpha
        hidden_dims = hidden_dims or [1024, 512]

        in_channels = 4 + (1 if condition_on_snr else 0)
        pooled_nt = max(4, num_bs_ant // pool_factor)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((num_users, pooled_nt)),
        )
        flattened = 2 * base_channels * num_users * pooled_nt
        output_dim = 2 * num_bs_ant * num_users
        head_layers: list[torch.nn.Module] = [torch.nn.Flatten()]
        prev = flattened
        for hidden in hidden_dims:
            head_layers.extend([torch.nn.Linear(prev, hidden), torch.nn.ReLU()])
            prev = hidden
        final_linear = torch.nn.Linear(prev, output_dim)
        torch.nn.init.zeros_(final_linear.weight)
        torch.nn.init.zeros_(final_linear.bias)
        head_layers.append(final_linear)
        self.head = torch.nn.Sequential(*head_layers)

        if learnable_alpha:
            alpha_tensor = torch.tensor(float(alpha_init), dtype=torch.float32)
            self.log_alpha = torch.nn.Parameter(torch.log(torch.expm1(alpha_tensor) + 1e-8))
        else:
            self.register_buffer("fixed_alpha", torch.tensor(float(alpha_init), dtype=torch.float32))

    def _alpha(self) -> torch.Tensor:
        if self.learnable_alpha:
            return torch.nn.functional.softplus(self.log_alpha)
        return self.fixed_alpha

    def forward(
        self,
        channel_real: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        channel_complex: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if snr_db is None:
            raise ValueError("snr_db must be provided for ResidualRZFBeamformer.")
        if channel_complex is None:
            raise ValueError("channel_complex must be provided for ResidualRZFBeamformer.")

        batch_size = channel_real.size(0)
        noise_var = noise_variance_from_snr(snr_db).to(channel_real.device)
        base_precoder = self._base_precoder(channel_complex, noise_var)
        base_feature = complex_to_real(base_precoder.transpose(-2, -1))
        features = torch.cat([channel_real, base_feature], dim=1)
        if self.condition_on_snr:
            snr_map = snr_db.view(batch_size, 1, 1, 1).expand(-1, 1, channel_real.size(-2), channel_real.size(-1)) / 20.0
            features = torch.cat([features, snr_map], dim=1)

        encoded = self.encoder(features)
        delta_real = self.head(encoded).reshape(batch_size, 2, self.num_bs_ant, self.num_users)
        delta_precoder = real_to_complex(delta_real)
        alpha = self._alpha().to(channel_real.device)
        precoder = power_normalization(base_precoder + alpha * delta_precoder)
        return {
            "precoder": precoder,
            "base_precoder": base_precoder,
            "delta_precoder": delta_precoder,
            "alpha": alpha,
            "base_method": self.base_method,
            "base_precoder_norm": torch.mean(torch.sqrt((torch.abs(base_precoder) ** 2).sum(dim=(-2, -1)).clamp_min(1e-12))),
            "delta_precoder_norm": torch.mean(
                torch.sqrt((torch.abs(delta_precoder) ** 2).sum(dim=(-2, -1)).clamp_min(1e-12))
            ),
        }

    def _base_precoder(self, channel_complex: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
        if self.base_method == "rzf":
            return rzf_precoder(channel_complex, noise_var=noise_var)
        if self.base_method in {"wmmse", "zf", "mrt"}:
            return get_digital_precoder(self.base_method, channel_complex, noise_var=noise_var)
        raise ValueError(f"Unsupported base_method for ResidualRZFBeamformer: {self.base_method}")
