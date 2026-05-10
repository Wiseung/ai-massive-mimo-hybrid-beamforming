#!/usr/bin/env python
"""Short differentiable OFDM beamforming smoke demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.models.differentiable_beamformer import TinyNeuralBeamformer
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_phy_helpers import add_awgn_torch, try_import_sionna_ofdm, try_import_sionna_phy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _rand_qpsk(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    bits = torch.randint(0, 2, shape + (2,), device=device)
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    return (real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=device))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_subcarriers = 8
    num_users = 4
    num_bs_ant = 16
    snr_db = 10.0
    noise_var = float(10.0 ** (-snr_db / 10.0))
    steps = 5

    ofdm = try_import_sionna_ofdm()
    phy = try_import_sionna_phy()
    used_sionna_ofdm = False
    used_sionna_channel = False
    fallback_used = False
    notes: list[str] = []

    channel_f = torch.randn(batch_size, num_subcarriers, num_users, num_bs_ant, dtype=torch.complex64, device=device)
    channel_f = (channel_f + 1j * torch.randn_like(channel_f)) / torch.sqrt(torch.tensor(2.0, device=device))
    tx_symbols = _rand_qpsk((batch_size, num_subcarriers, num_users), device)

    if ofdm["import_ok"]:
        try:
            rg = ofdm["ResourceGrid"](
                num_ofdm_symbols=1,
                fft_size=num_subcarriers,
                subcarrier_spacing=15_000.0,
                num_tx=1,
                num_streams_per_tx=1,
                num_guard_carriers=(0, 0),
                dc_null=False,
            )
            mapper = ofdm["ResourceGridMapper"](rg)
            sm = ofdm["StreamManagement"](np.array([[1]]), num_streams_per_tx=1)
            demapper = ofdm["ResourceGridDemapper"](rg, sm)
            _ = demapper
            flat_qpsk = tx_symbols[:, :, 0].reshape(batch_size, 1, 1, rg.num_data_symbols)
            _ = mapper(flat_qpsk)
            used_sionna_ofdm = True
            notes.append("Used real Sionna OFDM ResourceGrid/Mapper in differentiable demo setup.")
        except Exception as exc:  # pragma: no cover
            fallback_used = True
            notes.append(f"Sionna OFDM grid setup failed; using torch fallback: {type(exc).__name__}: {exc}")
    else:
        fallback_used = True
        notes.append(f"Sionna OFDM components unavailable; using torch fallback: {ofdm['error']}")

    model = TinyNeuralBeamformer(num_users=num_users, num_bs_ant=num_bs_ant, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses: list[float] = []
    last_grad_norm = 0.0

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        precoder = model(channel_f)
        tx = tx_symbols.unsqueeze(-1)
        tx_signal = torch.matmul(precoder, tx).squeeze(-1)
        noiseless = torch.einsum("bsku,bsu->bsk", channel_f, tx_signal)
        if phy["import_ok"]:
            try:
                awgn = phy["AWGN"]()
                rx = awgn(noiseless, no=torch.full((batch_size, 1, 1), noise_var, dtype=torch.float32, device=device))
                used_sionna_channel = True
            except Exception as exc:  # pragma: no cover
                rx, _ = add_awgn_torch(noiseless, snr_db)
                fallback_used = True
                notes.append(f"Sionna AWGN failed; using torch fallback: {type(exc).__name__}: {exc}")
        else:
            rx, _ = add_awgn_torch(noiseless, snr_db)
            fallback_used = True
            notes.append(f"Sionna AWGN unavailable; using torch fallback: {phy['error']}")
        loss = torch.mean(torch.abs(rx - tx_symbols) ** 2)
        loss.backward()
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.norm().item())
        last_grad_norm = grad_norm
        optimizer.step()
        losses.append(float(loss.item()))

    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_decreased = final_loss < initial_loss
    status = "ok" if loss_decreased else "backward_ok_but_no_clear_improvement"

    payload = {
        "sionna_import_ok": env_info["sionna_import_ok"],
        "used_sionna_ofdm": used_sionna_ofdm,
        "used_sionna_channel": used_sionna_channel,
        "fallback_used": fallback_used,
        "torch_version": env_info["torch_version"],
        "sionna_version": env_info["sionna_version"],
        "device": str(device),
        "beamformer_type": "TinyNeuralBeamformer",
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "grad_norm": last_grad_norm,
        "demo_status": status,
        "notes": notes
        + [
            "This is a differentiable beamformer smoke demo only.",
            "It is not a full Sionna end-to-end training pipeline, not Sionna RT, and not a 5G NR full-stack system.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved differentiable OFDM beamforming summary to {out_path}")


if __name__ == "__main__":
    main()
