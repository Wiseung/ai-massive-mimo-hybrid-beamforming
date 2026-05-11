#!/usr/bin/env python
"""Run an optional Sionna-native OFDM baseline chain without training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import (
    format_baseline_markdown,
    load_component,
    resolve_sionna_device,
    write_json,
    write_markdown,
)
from beamforming.utils.sionna_phy_helpers import add_awgn_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _hard_qpsk(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int64)


def _ber_from_hard_bits(hard_bits: torch.Tensor, ref_bits: torch.Tensor) -> float:
    return float((hard_bits.to(torch.int64) != ref_bits.to(torch.int64)).float().mean().item())


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env_info = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sionna_device = resolve_sionna_device(device)

    summary: dict[str, Any] = {
        "demo_scope": "experimental_sionna_native_ofdm_link_chain",
        "sionna_import_ok": env_info["sionna_import_ok"],
        "sionna_version": env_info["sionna_version"],
        "torch_version": env_info["torch_version"],
        "device": str(device),
        "demo_status": "skipped",
        "used_sionna_native_components": False,
        "used_components": [],
        "fallback_used": False,
        "fallback_flags": {
            "channel_fallback": False,
            "estimator_fallback": False,
            "equalizer_fallback": False,
            "demapper_fallback": False,
        },
        "ber_if_available": None,
        "symbol_mse": None,
        "empirical_snr_db": None,
        "notes": [],
    }

    if not env_info["sionna_import_ok"]:
        summary["notes"] = [
            "Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`.",
        ]
        write_json(out_path, summary)
        write_markdown(md_path, format_baseline_markdown(summary))
        print(f"Saved Sionna native OFDM baseline summary to {out_path}")
        return

    BinarySource, _, _ = load_component("BinarySource")
    Mapper, _, _ = load_component("Mapper")
    Demapper, _, _ = load_component("Demapper")
    ResourceGrid, _, _ = load_component("ResourceGrid")
    ResourceGridMapper, _, _ = load_component("ResourceGridMapper")
    ResourceGridDemapper, _, _ = load_component("ResourceGridDemapper")
    RemoveNulledSubcarriers, _, _ = load_component("RemoveNulledSubcarriers")
    OFDMChannel, _, _ = load_component("OFDMChannel")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    StreamManagement, _, _ = load_component("StreamManagement")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")

    batch_size = 32
    snr_db = 12.0
    noise_var = float(10.0 ** (-snr_db / 10.0))
    used_components: list[str] = []
    notes: list[str] = []

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
    )
    sm = StreamManagement(np.array([[1]]), num_streams_per_tx=1)
    source = BinarySource(device=sionna_device)
    mapper = Mapper("qam", 2, device=sionna_device)
    rg_mapper = ResourceGridMapper(rg, device=sionna_device)
    rg_demapper = ResourceGridDemapper(rg, sm, device=sionna_device)
    remove_nulled = RemoveNulledSubcarriers(rg, device=sionna_device)
    used_components.extend(
        [
            "BinarySource",
            "Mapper",
            "ResourceGrid",
            "ResourceGridMapper",
            "ResourceGridDemapper",
            "RemoveNulledSubcarriers",
        ]
    )

    bits = source([batch_size, 1, 1, int(rg.num_data_symbols * 2)]).to(device)
    tx_symbols = mapper(bits)
    tx_grid = rg_mapper(tx_symbols)
    _ = remove_nulled(tx_grid)

    rx_grid: torch.Tensor
    h_freq: torch.Tensor | None = None
    used_channel = False
    if OFDMChannel is not None and RayleighBlockFading is not None:
        try:
            channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1, device=sionna_device)
            channel = OFDMChannel(channel_model, rg, return_channel=True, device=sionna_device)
            rx_grid, h_freq = channel(tx_grid, no=torch.full((batch_size, 1, 1), noise_var, dtype=torch.float32, device=device))
            used_components.extend(["RayleighBlockFading", "OFDMChannel"])
            used_channel = True
            notes.append("Used real Sionna OFDMChannel with RayleighBlockFading.")
        except Exception as exc:  # pragma: no cover - optional runtime path
            summary["fallback_flags"]["channel_fallback"] = True
            summary["fallback_used"] = True
            notes.append(f"OFDMChannel path failed; used AWGN fallback: {type(exc).__name__}: {exc}")

    if not used_channel:
        rx_grid, _ = add_awgn_torch(tx_grid, snr_db)
        used_components.append("TorchAWGNFallback")

    rx_symbols = rg_demapper(rx_grid)
    symbol_reference = tx_symbols
    symbol_for_metrics = rx_symbols

    used_est_eq = False
    if used_channel and h_freq is not None and LSChannelEstimator is not None and LMMSEEqualizer is not None:
        try:
            estimator = LSChannelEstimator(rg, device=sionna_device)
            equalizer = LMMSEEqualizer(rg, sm, device=sionna_device)
            h_hat, err_var = estimator(rx_grid, torch.full((batch_size, 1, 1), noise_var, dtype=torch.float32, device=device))
            x_hat, no_eff = equalizer(
                rx_grid,
                h_hat,
                err_var,
                torch.full((batch_size, 1, 1), noise_var, dtype=torch.float32, device=device),
            )
            used_components.extend(["LSChannelEstimator", "LMMSEEqualizer"])
            symbol_for_metrics = x_hat
            used_est_eq = True
            notes.append(
                "Used real Sionna LSChannelEstimator and LMMSEEqualizer. "
                f"Effective noise variance mean={float(no_eff.real.mean().item()):.6f}."
            )
        except Exception as exc:  # pragma: no cover - optional runtime path
            summary["fallback_flags"]["estimator_fallback"] = True
            summary["fallback_flags"]["equalizer_fallback"] = True
            summary["fallback_used"] = True
            notes.append(f"Estimator/equalizer path failed; used demapped grid symbols only: {type(exc).__name__}: {exc}")
    else:
        if not used_channel:
            summary["fallback_flags"]["estimator_fallback"] = True
            summary["fallback_flags"]["equalizer_fallback"] = True
            notes.append("Skipped LS/LMMSE because the fallback channel path does not provide Sionna channel estimates.")

    ber = None
    if Demapper is not None:
        try:
            demapper = Demapper("app", "qam", 2, hard_out=True, device=sionna_device)
            hard_bits = demapper(symbol_for_metrics, torch.full((batch_size, 1, 1, 1), noise_var, dtype=torch.float32, device=device))
            ber = _ber_from_hard_bits(_hard_qpsk(hard_bits), bits)
            used_components.append("Demapper")
            notes.append("Used real Sionna Demapper for hard-bit BER.")
        except Exception as exc:  # pragma: no cover - optional runtime path
            summary["fallback_flags"]["demapper_fallback"] = True
            summary["fallback_used"] = True
            notes.append(f"Demapper path failed; BER omitted: {type(exc).__name__}: {exc}")
    else:
        summary["fallback_flags"]["demapper_fallback"] = True
        notes.append("Demapper is unavailable; BER omitted.")

    mse = float(torch.mean(torch.abs(symbol_for_metrics - symbol_reference) ** 2).item())
    empirical_snr_db = float(
        10.0
        * torch.log10(
            torch.mean(torch.abs(symbol_reference) ** 2)
            / torch.mean(torch.abs(symbol_for_metrics - symbol_reference) ** 2).clamp_min(1e-12)
        ).item()
    )

    summary.update(
        {
            "demo_status": "ok",
            "used_sionna_native_components": True,
            "used_components": used_components,
            "ber_if_available": ber,
            "symbol_mse": mse,
            "empirical_snr_db": empirical_snr_db,
            "notes": notes
            + [
                "Recommended beamforming insertion point for the next phase is frequency-domain per-subcarrier precoding before OFDMChannel.",
                "This remains a synthetic channel-level smoke chain, not a full 5G NR or production e2e stack.",
            ],
            "used_ls_lmmse": used_est_eq,
        }
    )

    write_json(out_path, summary)
    write_markdown(md_path, format_baseline_markdown(summary))
    print(f"Saved Sionna native OFDM baseline summary to {out_path}")


if __name__ == "__main__":
    main()
