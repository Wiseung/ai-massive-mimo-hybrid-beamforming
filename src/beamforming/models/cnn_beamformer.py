"""Lightweight CNN beamformer for narrowband channels."""

from __future__ import annotations

import math

import torch

from beamforming.baselines.mrt import mrt_precoder
from beamforming.models.constraints import compose_hybrid_precoder, constant_modulus_projection, power_normalization
from beamforming.utils.complex_ops import real_to_complex


class CNNBeamformer(torch.nn.Module):
    """CNN beamformer for reproducible single-GPU training."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        num_rf_chains: int,
        hybrid: bool = False,
        condition_on_snr: bool = False,
        snr_embed_dim: int = 16,
        base_channels: int = 32,
        pool_factor: int = 2,
        hidden_dims: list[int] | None = None,
        residual_to_mrt: bool = True,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.hybrid = hybrid
        self.condition_on_snr = condition_on_snr
        self.residual_to_mrt = residual_to_mrt and not hybrid
        hidden_dims = hidden_dims or [1024, 512]

        in_channels = 2 + (1 if condition_on_snr else 0)
        encoder_layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((num_users, max(4, num_bs_ant // pool_factor))),
        ]
        self.encoder = torch.nn.Sequential(*encoder_layers)
        pooled_nt = max(4, num_bs_ant // pool_factor)
        flattened = 2 * base_channels * num_users * pooled_nt
        if hybrid:
            output_dim = 2 * (num_bs_ant * num_rf_chains + num_rf_chains * num_users)
        else:
            output_dim = 2 * num_bs_ant * num_users
        head_layers: list[torch.nn.Module] = [torch.nn.Flatten()]
        prev = flattened
        for hidden in hidden_dims:
            head_layers.extend([torch.nn.Linear(prev, hidden), torch.nn.ReLU()])
            prev = hidden
        head_layers.append(torch.nn.Linear(prev, output_dim))
        self.head = torch.nn.Sequential(*head_layers)

    def forward(
        self,
        channel_real: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        channel_complex: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = channel_real.size(0)
        features = channel_real
        if self.condition_on_snr:
            if snr_db is None:
                raise ValueError("snr_db must be provided when condition_on_snr=True.")
            snr_map = snr_db.view(batch_size, 1, 1, 1).expand(-1, 1, channel_real.size(-2), channel_real.size(-1)) / 20.0
            features = torch.cat([channel_real, snr_map], dim=1)
        encoded = self.encoder(features)
        output = self.head(encoded)
        if self.hybrid:
            split = 2 * self.num_bs_ant * self.num_rf_chains
            analog_real = output[:, :split].reshape(batch_size, 2, self.num_bs_ant, self.num_rf_chains)
            digital_real = output[:, split:].reshape(batch_size, 2, self.num_rf_chains, self.num_users)
            analog = constant_modulus_projection(real_to_complex(analog_real))
            digital = real_to_complex(digital_real)
            precoder = compose_hybrid_precoder(analog, digital)
            return {"precoder": precoder, "analog_precoder": analog, "digital_precoder": digital}

        precoder_real = output.reshape(batch_size, 2, self.num_bs_ant, self.num_users)
        residual = real_to_complex(precoder_real)
        if self.residual_to_mrt:
            if channel_complex is None:
                raise ValueError("channel_complex must be provided when residual_to_mrt=True.")
            base = mrt_precoder(channel_complex)
            precoder = power_normalization(base + residual / math.sqrt(self.num_bs_ant))
        else:
            precoder = power_normalization(residual)
        return {"precoder": precoder}
