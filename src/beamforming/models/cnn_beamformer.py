"""Lightweight CNN beamformer for narrowband channels."""

from __future__ import annotations

import torch

from beamforming.models.constraints import compose_hybrid_precoder, constant_modulus_projection, power_normalization
from beamforming.utils.complex_ops import real_to_complex


class CNNBeamformer(torch.nn.Module):
    """Small convolutional beamformer for reproducible single-GPU training."""

    def __init__(self, num_users: int, num_bs_ant: int, num_rf_chains: int, hybrid: bool = False) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.hybrid = hybrid

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((num_users, max(4, num_bs_ant // 4))),
        )
        flattened = 32 * num_users * max(4, num_bs_ant // 4)
        if hybrid:
            output_dim = 2 * (num_bs_ant * num_rf_chains + num_rf_chains * num_users)
        else:
            output_dim = 2 * num_bs_ant * num_users
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flattened, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
        )

    def forward(self, channel_real: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = channel_real.size(0)
        encoded = self.encoder(channel_real)
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
        precoder = power_normalization(real_to_complex(precoder_real))
        return {"precoder": precoder}
