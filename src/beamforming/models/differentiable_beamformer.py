"""Minimal differentiable beamformers for optional Sionna smoke demos."""

from __future__ import annotations

import torch

from beamforming.models.constraints import power_normalization


class TinyNeuralBeamformer(torch.nn.Module):
    """Small complex-to-complex neural beamformer with explicit power normalization."""

    def __init__(self, num_users: int, num_bs_ant: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        in_dim = 2 * num_users * num_bs_ant
        out_dim = 2 * num_bs_ant * num_users
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def _forward_single(self, channel: torch.Tensor) -> torch.Tensor:
        batch = channel.size(0)
        features = torch.view_as_real(channel).reshape(batch, -1)
        output = self.net(features).reshape(batch, self.num_bs_ant, self.num_users, 2)
        precoder = torch.complex(output[..., 0], output[..., 1])
        return power_normalization(precoder)

    def forward(self, channel: torch.Tensor) -> torch.Tensor:
        if channel.ndim == 3:
            return self._forward_single(channel)
        if channel.ndim == 4:
            batch, num_sc, _, _ = channel.shape
            precoders = []
            for sc in range(num_sc):
                precoders.append(self._forward_single(channel[:, sc, :, :]))
            return torch.stack(precoders, dim=1).reshape(batch, num_sc, self.num_bs_ant, self.num_users)
        raise ValueError("channel must have shape (B, K, Nt) or (B, Nsc, K, Nt)")
