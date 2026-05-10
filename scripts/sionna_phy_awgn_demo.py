#!/usr/bin/env python
"""Minimal Sionna PHY AWGN smoke demo with explicit fallback behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_phy_helpers import add_awgn_torch, try_import_sionna_phy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _qpsk_map(bits: torch.Tensor) -> torch.Tensor:
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    return (real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=bits.device))


def _qpsk_hard_demod(symbols: torch.Tensor) -> torch.Tensor:
    real_bits = (symbols.real < 0).to(torch.int64)
    imag_bits = (symbols.imag < 0).to(torch.int64)
    return torch.stack([real_bits, imag_bits], dim=-1)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    num_symbols = 256
    snr_db = 10.0

    summary: dict[str, object] = {
        "sionna_import_ok": env_info["sionna_import_ok"],
        "used_sionna_phy": False,
        "fallback_used": False,
        "torch_version": env_info["torch_version"],
        "sionna_version": env_info["sionna_version"],
        "device": str(device),
        "batch_size": batch_size,
        "modulation_order_or_note": "QPSK",
        "snr_db": snr_db,
        "mse": None,
        "ber_if_available": None,
        "demo_status": "skipped",
        "notes": [],
    }

    if not env_info["sionna_import_ok"]:
        summary["notes"] = [
            "Sionna is not installed.",
            env_info["install_hint"],
        ]
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved Sionna PHY AWGN summary to {out_path}")
        return

    phy = try_import_sionna_phy()
    bits = torch.randint(0, 2, (batch_size, num_symbols, 2), device=device)
    tx_symbols = _qpsk_map(bits)

    rx_symbols: torch.Tensor
    noise_var: float
    if phy["import_ok"]:
        try:
            awgn = phy["AWGN"]()
            noise_var = float(10.0 ** (-snr_db / 10.0))
            no = torch.full((batch_size, 1, 1), noise_var, dtype=torch.float32, device=device)
            rx_symbols = awgn(tx_symbols, no=no)
            summary["used_sionna_phy"] = True
            summary["notes"].append("Used sionna.phy.channel.AWGN for the channel step.")
        except Exception as exc:  # pragma: no cover - optional runtime path
            rx_symbols, noise_var = add_awgn_torch(tx_symbols, snr_db)
            summary["fallback_used"] = True
            summary["notes"].append(f"Sionna PHY AWGN call failed; used torch fallback: {type(exc).__name__}: {exc}")
    else:
        rx_symbols, noise_var = add_awgn_torch(tx_symbols, snr_db)
        summary["fallback_used"] = True
        summary["notes"].append(f"Sionna PHY helpers unavailable; used torch fallback: {phy['error']}")

    mse = torch.mean(torch.abs(rx_symbols - tx_symbols) ** 2).item()
    rx_bits = _qpsk_hard_demod(rx_symbols)
    ber = (rx_bits != bits).float().mean().item()

    summary["mse"] = float(mse)
    summary["ber_if_available"] = float(ber)
    summary["noise_var"] = float(noise_var)
    summary["demo_status"] = "ok"
    summary["notes"].append("This is a minimal PHY smoke demo, not a full Sionna end-to-end link.")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Sionna PHY AWGN summary to {out_path}")


if __name__ == "__main__":
    main()
