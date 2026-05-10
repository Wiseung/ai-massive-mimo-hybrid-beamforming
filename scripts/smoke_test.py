#!/usr/bin/env python
"""Minimal smoke test for channel metrics and tiny model forward pass."""

from __future__ import annotations

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.baselines.mrt import mrt_precoder
from beamforming.baselines.zf import zf_precoder
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr
from beamforming.utils.complex_ops import complex_to_real


class TinyMLP(torch.nn.Module):
    def __init__(self, num_users: int, num_bs_ant: int) -> None:
        super().__init__()
        in_dim = 2 * num_users * num_bs_ant
        out_dim = 2 * num_bs_ant * num_users
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    batch_size = 8
    num_users = 4
    num_bs_ant = 16
    channel = torch.randn(batch_size, num_users, num_bs_ant, dtype=torch.complex64)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0))
    noise_var = noise_variance_from_snr(10.0)

    mrt = mrt_precoder(channel)
    zf = zf_precoder(channel)
    mrt_rate = multi_user_downlink_sum_rate(channel, mrt, noise_var).mean().item()
    zf_rate = multi_user_downlink_sum_rate(channel, zf, noise_var).mean().item()

    features = complex_to_real(channel).reshape(batch_size, -1)
    mlp = TinyMLP(num_users=num_users, num_bs_ant=num_bs_ant)
    output = mlp(features)

    print("Smoke test succeeded.")
    print(f"MRT sum-rate: {mrt_rate:.4f} bit/s/Hz")
    print(f"ZF sum-rate: {zf_rate:.4f} bit/s/Hz")
    print("TinyMLP output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
