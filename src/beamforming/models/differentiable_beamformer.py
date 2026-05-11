"""Minimal differentiable beamformers for optional Sionna smoke demos."""

from __future__ import annotations

import torch

from beamforming.models.constraints import power_normalization


class TinyNeuralBeamformer(torch.nn.Module):
    """Small complex-to-complex neural beamformer with optional SNR conditioning."""

    def __init__(
        self,
        num_users: int,
        num_bs_ant: int,
        hidden_dim: int = 128,
        condition_on_snr: bool = False,
        normalize_power_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_bs_ant = num_bs_ant
        self.condition_on_snr = condition_on_snr
        self.normalize_power_output = normalize_power_output
        in_dim = 2 * num_users * num_bs_ant + (1 if condition_on_snr else 0)
        out_dim = 2 * num_bs_ant * num_users
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def _prepare_snr_feature(self, batch: int, channel: torch.Tensor, snr_db: torch.Tensor | None) -> torch.Tensor:
        if not self.condition_on_snr:
            return torch.empty(batch, 0, dtype=channel.real.dtype, device=channel.device)
        if snr_db is None:
            snr_db = torch.zeros(batch, dtype=channel.real.dtype, device=channel.device)
        if snr_db.ndim == 0:
            snr_db = snr_db.repeat(batch)
        snr_feature = snr_db.reshape(batch, 1).to(device=channel.device, dtype=channel.real.dtype)
        return snr_feature / 20.0

    def _forward_single(self, channel: torch.Tensor, snr_db: torch.Tensor | None = None) -> torch.Tensor:
        batch = channel.size(0)
        features = torch.view_as_real(channel).reshape(batch, -1)
        snr_feature = self._prepare_snr_feature(batch, channel, snr_db)
        if snr_feature.numel() > 0:
            features = torch.cat([features, snr_feature], dim=-1)
        output = self.net(features).reshape(batch, self.num_bs_ant, self.num_users, 2)
        precoder = torch.complex(output[..., 0], output[..., 1])
        if self.normalize_power_output:
            return power_normalization(precoder)
        return precoder

    def forward(self, channel: torch.Tensor, snr_db: torch.Tensor | None = None) -> torch.Tensor:
        if channel.ndim == 3:
            return self._forward_single(channel, snr_db=snr_db)
        if channel.ndim == 4:
            batch, num_sc, _, _ = channel.shape
            precoders = []
            for sc in range(num_sc):
                precoders.append(self._forward_single(channel[:, sc, :, :], snr_db=snr_db))
            return torch.stack(precoders, dim=1).reshape(batch, num_sc, self.num_bs_ant, self.num_users)
        raise ValueError("channel must have shape (B, K, Nt) or (B, Nsc, K, Nt)")
