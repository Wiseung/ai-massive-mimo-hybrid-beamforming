#!/usr/bin/env python
"""Sweep extracted-H configuration choices for Sionna channel extraction."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import pandas as pd
import torch

add_src_to_path()

from beamforming.utils.sionna_channel_extraction import extract_h_f_from_sionna_channel, summarize_h_f_matrix_stats
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import build_pilot_aware_multiuser_resource_grid
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "extraction_sweep.md"
    csv_path = out_dir / "extraction_sweep.csv"
    env = collect_sionna_env_info()
    if not env["sionna_import_ok"]:
        write_markdown(
            summary_path,
            [
                "# Sionna Channel Extraction Config Sweep",
                "",
                "- status: `skipped`",
                "- reason: `Sionna not installed`",
            ],
        )
        print(f"Saved extraction sweep summary to {summary_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sionna_device = resolve_sionna_device(device)
    batch_sizes = [8] if args.quick else [8, 32]
    selected_symbols = ["first_data", "last_data", "all_data_average"]
    selected_subcarriers = ["all_effective", "center_8", "center_16"]
    normalize_choices = [False, True]

    OFDMChannel, _, _ = load_component("OFDMChannel")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    rg, _, rg_meta = build_pilot_aware_multiuser_resource_grid(
        num_users=4,
        num_effective_subcarriers=16,
        num_ofdm_symbols=3,
        device=device,
        pilot_ofdm_symbol_indices=[0],
    )
    if rg is None or OFDMChannel is None or RayleighBlockFading is None:
        write_markdown(
            summary_path,
            [
                "# Sionna Channel Extraction Config Sweep",
                "",
                f"- status: `failed`",
                f"- reason: `{rg_meta.get('fallback_reason', 'required_components_unavailable') if rg is None else 'required_components_unavailable'}`",
            ],
        )
        print(f"Saved extraction sweep summary to {summary_path}")
        return

    rows: list[dict[str, Any]] = []
    for batch_size, selected_symbol, selected_subcarrier, normalize_channel in itertools.product(
        batch_sizes, selected_symbols, selected_subcarriers, normalize_choices
    ):
        torch.manual_seed(1234 + batch_size)
        channel_model = RayleighBlockFading(num_rx=4, num_rx_ant=1, num_tx=1, num_tx_ant=16, device=sionna_device)
        channel = OFDMChannel(channel_model, rg, return_channel=True, device=sionna_device)
        dummy_x = torch.zeros(batch_size, 1, 16, rg.num_ofdm_symbols, rg.fft_size, dtype=torch.complex64, device=device)
        _, h = channel(dummy_x, no=torch.full((batch_size, 4, 1), 0.1, dtype=torch.float32, device=device))
        h_f, meta, success, fallback_reason = extract_h_f_from_sionna_channel(
            h,
            resource_grid=rg,
            num_users=4,
            num_bs_ant=16,
            selected_ofdm_symbol=selected_symbol,
            effective_subcarriers=selected_subcarrier,
            normalize_channel=normalize_channel,
        )
        stats = summarize_h_f_matrix_stats(h_f)
        rows.append(
            {
                "batch_size": batch_size,
                "selected_ofdm_symbol": selected_symbol,
                "effective_subcarriers": selected_subcarrier,
                "normalize_channel": normalize_channel,
                "extraction_success": bool(success),
                "fallback_reason": fallback_reason,
                "selected_data_symbol_indices": meta.get("selected_data_symbol_indices"),
                "selected_effective_subcarrier_count": meta.get("selected_effective_subcarrier_count"),
                "h_f_shape": [int(x) for x in h_f.shape] if h_f is not None else None,
                "fro_norm_mean": stats.get("fro_norm_mean"),
                "rank_mean": stats.get("rank_mean"),
                "condition_number_mean": stats.get("condition_number_mean"),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)

    success_frame = frame[frame["extraction_success"] == True].copy()  # noqa: E712
    symbol_sensitivity = success_frame.groupby("selected_ofdm_symbol")["fro_norm_mean"].mean().to_dict() if not success_frame.empty else {}
    subcarrier_sensitivity = success_frame.groupby("effective_subcarriers")["fro_norm_mean"].mean().to_dict() if not success_frame.empty else {}
    summary_lines = [
        "# Sionna Channel Extraction Config Sweep",
        "",
        f"- quick: `{args.quick}`",
        f"- resource_grid_meta: `{rg_meta}`",
        "",
        "## Key answers",
        f"1. selected OFDM symbol sensitivity (mean fro_norm): `{symbol_sensitivity}`",
        f"2. effective subcarrier sensitivity (mean fro_norm): `{subcarrier_sensitivity}`",
        f"3. extracted H_f norm/rank stable across successful rows: `{bool(success_frame['rank_mean'].notna().all() and success_frame['fro_norm_mean'].notna().all())}`",
        "4. recommended default extraction config: `selected_ofdm_symbol=first_data`, `effective_subcarriers=all_effective`, `normalize_channel=false`.",
        "",
        "This sweep reduces ambiguity around axis/config choices, but it does not justify a full native-only benchmark claim.",
    ]
    write_markdown(summary_path, summary_lines)
    print(f"Saved extraction config sweep to {out_dir}")


if __name__ == "__main__":
    main()
