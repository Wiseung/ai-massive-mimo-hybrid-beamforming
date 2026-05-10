"""Small MLP beamformer for narrowband channels."""

from __future__ import annotations

import math

import torch

from beamforming.baselines.mrt import mrt_precoder
from beamforming.models.constraints import compose_hybrid_precoder, constant_modulus_projection, power_normalization
from beamforming.utils.complex_ops import real_to_complex


class MLPBeamformer(torch.nn.Module):
    """Simple MLP that predicts a digital or hybrid precoder."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        num_rf_chains: int,
        hidden_dims: list[int] | None = None,
        hybrid: bool = False,
        condition_on_snr: bool = False,
        snr_embed_dim: int = 16,
        residual_to_mrt: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [1024, 512]
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.hybrid = hybrid
        self.condition_on_snr = condition_on_snr
        self.residual_to_mrt = residual_to_mrt and not hybrid

        self.snr_embed_dim = snr_embed_dim if condition_on_snr else 0
        if condition_on_snr:
            self.snr_embed = torch.nn.Sequential(
                torch.nn.Linear(1, snr_embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(snr_embed_dim, snr_embed_dim),
                torch.nn.ReLU(),
            )

        input_dim = 2 * num_users * num_bs_ant + self.snr_embed_dim
        if hybrid:
            output_dim = 2 * (num_bs_ant * num_rf_chains + num_rf_chains * num_users)
        else:
            output_dim = 2 * num_bs_ant * num_users

        layers: list[torch.nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([torch.nn.Linear(prev, hidden), torch.nn.ReLU()])
            prev = hidden
        layers.append(torch.nn.Linear(prev, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(
        self,
        channel_real: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        channel_complex: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = channel_real.size(0)
        flat = channel_real.reshape(batch_size, -1)
        if self.condition_on_snr:
            if snr_db is None:
                raise ValueError("snr_db must be provided when condition_on_snr=True.")
            snr_features = self.snr_embed(snr_db.view(batch_size, 1).float())
            flat = torch.cat([flat, snr_features], dim=-1)

        output = self.net(flat)
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
