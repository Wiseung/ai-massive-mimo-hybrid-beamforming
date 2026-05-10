#!/usr/bin/env python
"""Run a lightweight gradient check for the optional differentiable beamformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.metrics.sum_rate import noise_variance_from_snr
from beamforming.models.differentiable_beamformer import TinyNeuralBeamformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_users = 4
    num_bs_ant = 16
    model = TinyNeuralBeamformer(num_users=num_users, num_bs_ant=num_bs_ant, hidden_dim=64).to(device)
    channel = torch.randn(batch_size, num_users, num_bs_ant, dtype=torch.complex64, device=device)
    channel = (channel + 1j * torch.randn_like(channel)) / torch.sqrt(torch.tensor(2.0, device=device))
    symbols = (
        torch.randn(batch_size, num_users, 1, dtype=torch.complex64, device=device)
        + 1j * torch.randn(batch_size, num_users, 1, dtype=torch.complex64, device=device)
    ) / torch.sqrt(torch.tensor(2.0, device=device))
    precoder = model(channel)
    tx = torch.matmul(precoder, symbols)
    rx = torch.matmul(channel, tx).squeeze(-1)
    target = symbols.squeeze(-1)
    noise_var = float(noise_variance_from_snr(10.0).item())
    loss = torch.mean(torch.abs(rx - target) ** 2) + 0.01 * torch.mean(torch.abs(precoder) ** 2) * noise_var
    loss.backward()

    params = []
    for name, param in model.named_parameters():
        grad = param.grad
        params.append(
            {
                "name": name,
                "shape": list(param.shape),
                "requires_grad": bool(param.requires_grad),
                "grad_norm": float(grad.norm().item()) if grad is not None else 0.0,
                "grad_has_nan": bool(torch.isnan(grad).any().item()) if grad is not None else False,
            }
        )

    payload = {
        "device": str(device),
        "loss": float(loss.item()),
        "parameters": params,
        "all_gradients_present": all(item["grad_norm"] >= 0.0 for item in params),
        "any_nonzero_gradient": any(item["grad_norm"] > 0.0 for item in params),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved differentiable beamformer gradient check to {out_path}")


if __name__ == "__main__":
    main()
