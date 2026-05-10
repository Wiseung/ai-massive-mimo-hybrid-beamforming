"""Small MLP beamformer for narrowband channels."""

from __future__ import annotations

import torch

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
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.num_rf_chains = num_rf_chains
        self.hybrid = hybrid

        input_dim = 2 * num_users * num_bs_ant
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

    def forward(self, channel_real: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = channel_real.size(0)
        flat = channel_real.reshape(batch_size, -1)
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
        precoder = power_normalization(real_to_complex(precoder_real))
        return {"precoder": precoder}
