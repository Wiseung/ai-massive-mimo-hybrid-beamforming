#!/usr/bin/env python
"""Minimal OFDM resource-grid smoke demo with Sionna-first behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_phy_helpers import add_awgn_torch, try_import_sionna_ofdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _qpsk_map(bits: torch.Tensor) -> torch.Tensor:
    real = 1.0 - 2.0 * bits[..., 0].float()
    imag = 1.0 - 2.0 * bits[..., 1].float()
    return (real + 1j * imag) / torch.sqrt(torch.tensor(2.0, device=bits.device))


def _hard_demod(symbols: torch.Tensor) -> torch.Tensor:
    return torch.stack([(symbols.real < 0).to(torch.int64), (symbols.imag < 0).to(torch.int64)], dim=-1)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snr_db = 10.0
    batch_size = 64
    num_ofdm_symbols = 4
    fft_size = 16
    summary: dict[str, object] = {
        "sionna_import_ok": env_info["sionna_import_ok"],
        "used_sionna_ofdm": False,
        "fallback_used": False,
        "sionna_version": env_info["sionna_version"],
        "torch_version": env_info["torch_version"],
        "device": str(device),
        "num_ofdm_symbols": num_ofdm_symbols,
        "fft_size": fft_size,
        "num_data_symbols": None,
        "modulation": "QPSK",
        "snr_db": snr_db,
        "mse": None,
        "ber_if_available": None,
        "demo_status": "skipped",
        "notes": [],
    }

    if not env_info["sionna_import_ok"]:
        summary["notes"] = ["Sionna is not installed.", env_info["install_hint"]]
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved Sionna OFDM resource-grid summary to {out_path}")
        return

    ofdm = try_import_sionna_ofdm()
    if ofdm["import_ok"]:
        try:
            rg = ofdm["ResourceGrid"](
                num_ofdm_symbols=num_ofdm_symbols,
                fft_size=fft_size,
                subcarrier_spacing=15_000.0,
                num_tx=1,
                num_streams_per_tx=1,
                num_guard_carriers=(1, 1),
                dc_null=True,
            )
            mapper = ofdm["ResourceGridMapper"](rg)
            sm = ofdm["StreamManagement"](np.array([[1]]), num_streams_per_tx=1)
            demapper = ofdm["ResourceGridDemapper"](rg, sm)
            remove_nulled = ofdm["RemoveNulledSubcarriers"](rg)
            awgn = ofdm["AWGN"]()

            num_data_symbols = int(rg.num_data_symbols)
            bits = torch.randint(0, 2, (batch_size, 1, 1, num_data_symbols, 2), device=device)
            tx_symbols = _qpsk_map(bits)
            tx_grid = mapper(tx_symbols)
            rx_grid = awgn(tx_grid, no=torch.full((batch_size, 1, 1, 1, 1), 10.0 ** (-snr_db / 10.0), dtype=torch.float32, device=device))
            rx_symbols = demapper(rx_grid)
            _ = remove_nulled(rx_grid)

            mse = float(torch.mean(torch.abs(rx_symbols - tx_symbols) ** 2).item())
            ber = float((_hard_demod(rx_symbols) != bits).float().mean().item())
            empirical_snr = float(10.0 * torch.log10(torch.mean(torch.abs(tx_symbols) ** 2) / torch.mean(torch.abs(rx_symbols - tx_symbols) ** 2)).item())

            summary.update(
                {
                    "used_sionna_ofdm": True,
                    "fallback_used": False,
                    "num_data_symbols": num_data_symbols,
                    "mse": mse,
                    "ber_if_available": ber,
                    "empirical_snr_db": empirical_snr,
                    "demo_status": "ok",
                    "notes": [
                        "Used real Sionna OFDM ResourceGrid, ResourceGridMapper, ResourceGridDemapper, RemoveNulledSubcarriers, and AWGN.",
                        "This is still a smoke demo, not a full Sionna end-to-end training pipeline.",
                    ],
                }
            )
        except Exception as exc:  # pragma: no cover
            summary["fallback_used"] = True
            summary["notes"].append(f"Sionna OFDM path failed; used torch fallback: {type(exc).__name__}: {exc}")
    else:
        summary["fallback_used"] = True
        summary["notes"].append(f"Sionna OFDM components unavailable; used torch fallback: {ofdm['error']}")

    if summary["demo_status"] != "ok":
        num_data_symbols = num_ofdm_symbols * fft_size
        bits = torch.randint(0, 2, (batch_size, num_data_symbols, 2), device=device)
        tx_symbols = _qpsk_map(bits)
        tx_grid = tx_symbols.view(batch_size, num_ofdm_symbols, fft_size)
        rx_grid, _ = add_awgn_torch(tx_grid, snr_db)
        rx_symbols = rx_grid.reshape(batch_size, num_data_symbols)
        mse = float(torch.mean(torch.abs(rx_symbols - tx_symbols.reshape(batch_size, num_data_symbols)) ** 2).item())
        ber = float((_hard_demod(rx_symbols) != bits).float().mean().item())
        empirical_snr = float(10.0 * torch.log10(torch.mean(torch.abs(tx_grid) ** 2) / torch.mean(torch.abs(rx_grid - tx_grid) ** 2)).item())
        summary.update(
            {
                "num_data_symbols": num_data_symbols,
                "mse": mse,
                "ber_if_available": ber,
                "empirical_snr_db": empirical_snr,
                "demo_status": "ok",
            }
        )

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Sionna OFDM resource-grid summary to {out_path}")


if __name__ == "__main__":
    main()
