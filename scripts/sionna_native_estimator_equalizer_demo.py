#!/usr/bin/env python
"""Minimal Sionna-native estimator/equalizer/demapper success chain."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sionna_device = resolve_sionna_device(device)

    summary: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "used_sionna_resource_grid": False,
        "used_sionna_channel": False,
        "used_sionna_estimator": False,
        "used_sionna_equalizer": False,
        "used_sionna_demapper": False,
        "fallback_used": False,
        "ber_if_available": None,
        "symbol_mse": None,
        "demo_status": "skipped",
        "notes": [],
    }

    if not env["sionna_import_ok"]:
        summary["notes"] = ["Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`."]
        write_json(out_path, summary)
        write_markdown(md_path, ["# Estimator Equalizer Demo", "", "- Sionna not installed."])
        print(f"Saved estimator/equalizer demo summary to {out_path}")
        return

    ResourceGrid, _, _ = load_component("ResourceGrid")
    ResourceGridMapper, _, _ = load_component("ResourceGridMapper")
    OFDMChannel, _, _ = load_component("OFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    Demapper, _, _ = load_component("Demapper")
    BinarySource, _, _ = load_component("BinarySource")
    Mapper, _, _ = load_component("Mapper")
    StreamManagement, _, _ = load_component("StreamManagement")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")

    try:
        rg = ResourceGrid(
            num_ofdm_symbols=4,
            fft_size=16,
            subcarrier_spacing=15_000.0,
            num_tx=1,
            num_streams_per_tx=1,
            num_guard_carriers=(1, 1),
            dc_null=True,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[0],
            device=sionna_device,
        )
        sm = StreamManagement(np.array([[1]]), num_streams_per_tx=1)
        src = BinarySource(device=sionna_device)
        mapper = Mapper("qam", 2, device=sionna_device)
        rg_mapper = ResourceGridMapper(rg, device=sionna_device)
        bits = src([32, 1, 1, int(rg.num_data_symbols * 2)])
        tx_symbols = mapper(bits)
        tx_grid = rg_mapper(tx_symbols)
        channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1, device=sionna_device)
        channel = OFDMChannel(channel_model, rg, return_channel=True, device=sionna_device)
        rx_grid, _ = channel(tx_grid, no=torch.full((32, 1, 1), 0.1, dtype=torch.float32, device=device))
        estimator = LSChannelEstimator(rg, device=sionna_device)
        equalizer = LMMSEEqualizer(rg, sm, device=sionna_device)
        demapper = Demapper("app", "qam", 2, hard_out=True, device=sionna_device)
        h_hat, err_var = estimator(rx_grid, torch.full((32, 1, 1), 0.1, dtype=torch.float32, device=device))
        x_hat, _ = equalizer(rx_grid, h_hat, err_var, torch.full((32, 1, 1), 0.1, dtype=torch.float32, device=device))
        hard_bits = demapper(x_hat, torch.full((32, 1, 1, 1), 0.1, dtype=torch.float32, device=device))
        ber = float((hard_bits.to(torch.int64) != bits.to(torch.int64)).float().mean().item())
        mse = float(torch.mean(torch.abs(x_hat - tx_symbols) ** 2).item())
        summary.update(
            {
                "used_sionna_resource_grid": True,
                "used_sionna_channel": True,
                "used_sionna_estimator": True,
                "used_sionna_equalizer": True,
                "used_sionna_demapper": True,
                "ber_if_available": ber,
                "symbol_mse": mse,
                "demo_status": "ok",
                "notes": ["Minimal pilot-based Sionna receiver chain succeeded without beamforming."],
            }
        )
    except Exception as exc:  # pragma: no cover
        summary.update(
            {
                "fallback_used": True,
                "demo_status": "failed",
                "notes": [f"Receiver chain failed: {type(exc).__name__}: {exc}"],
            }
        )

    write_json(out_path, summary)
    write_markdown(
        md_path,
        [
            "# Sionna Native Estimator Equalizer Demo",
            "",
            f"- Demo status: `{summary['demo_status']}`",
            f"- used_sionna_estimator: `{summary['used_sionna_estimator']}`",
            f"- used_sionna_equalizer: `{summary['used_sionna_equalizer']}`",
            f"- used_sionna_demapper: `{summary['used_sionna_demapper']}`",
            f"- ber_if_available: `{summary['ber_if_available']}`",
            f"- symbol_mse: `{summary['symbol_mse']}`",
        ],
    )
    print(f"Saved estimator/equalizer demo summary to {out_path}")


if __name__ == "__main__":
    main()
