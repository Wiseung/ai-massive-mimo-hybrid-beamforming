#!/usr/bin/env python
"""Validate axis mapping from Sionna channel tensors to project ``H_f``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_channel_extraction import (
    resolve_resource_grid_data_symbol_indices,
    summarize_h_f_matrix_stats,
)
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import generate_shared_sionna_channel_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _to_md(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Sionna Extracted H_f Axis Validation",
        "",
        f"- validation_status: `{payload['validation_status']}`",
        f"- sionna_channel_tensor_shape: `{payload['sionna_channel_tensor_shape']}`",
        f"- extracted_h_f_shape: `{payload['extracted_h_f_shape']}`",
        f"- selected_data_symbol_indices: `{payload['selected_data_symbol_indices']}`",
        f"- selected_effective_subcarrier_indices: `{payload['selected_effective_subcarrier_indices']}`",
        f"- axis_spot_check_passed: `{payload['axis_spot_check_passed']}`",
        f"- spot_check_max_abs_diff: `{payload['spot_check_max_abs_diff']}`",
        f"- selected_ofdm_symbol_is_data_bearing: `{payload['selected_ofdm_symbol_is_data_bearing']}`",
        f"- effective_subcarrier_count_matches_nsc: `{payload['effective_subcarrier_count_matches_nsc']}`",
        f"- hidden_squeeze_transpose_risk: `{payload['hidden_squeeze_transpose_risk']}`",
        "",
        "## Matrix stats",
    ]
    for key, value in payload["matrix_stats"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Notes"])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload: dict[str, Any] = {
        "validation_status": "skipped",
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "sionna_channel_tensor_shape": None,
        "extracted_h_f_shape": None,
        "selected_data_symbol_indices": [],
        "selected_effective_subcarrier_indices": [],
        "rx_dimension_maps_to_users": False,
        "tx_ant_dimension_maps_to_nt": False,
        "selected_ofdm_symbol_is_data_bearing": False,
        "effective_subcarrier_count_matches_nsc": False,
        "complex_dtype": False,
        "finite_values": False,
        "axis_spot_check_passed": False,
        "spot_check_max_abs_diff": None,
        "matrix_stats": {"available": False},
        "hidden_squeeze_transpose_risk": "unverified",
        "notes": [],
    }
    if not env["sionna_import_ok"]:
        payload["notes"].append("Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`.")
        write_json(out_path, payload)
        write_markdown(md_path, _to_md(payload))
        print(f"Saved extracted H_f axis validation to {out_path}")
        return

    bundle = generate_shared_sionna_channel_bundle(
        batch_size=8,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        noise_var=0.1,
        device=device,
        selected_ofdm_symbol="first_data",
        effective_subcarriers="all_effective",
        normalize_channel=False,
    )
    extraction_meta = bundle.bundle_meta["channel_meta"]["extraction_meta"]
    h = bundle.h_full
    h_f = bundle.h_f
    selected_symbols = [int(x) for x in extraction_meta["selected_data_symbol_indices"]]
    selected_subcarriers = [int(x) for x in extraction_meta["selected_effective_subcarrier_indices"]]
    data_symbol_indices = resolve_resource_grid_data_symbol_indices(bundle.resource_grid)

    max_abs_diff = 0.0
    for user_idx in range(h_f.size(2)):
        for ant_idx in range(h_f.size(3)):
            raw = h[:, user_idx, 0, 0, ant_idx, :, :]
            raw = raw[:, selected_symbols, :]
            raw = raw[:, :, selected_subcarriers]
            if raw.size(1) == 1:
                expected = raw[:, 0, :]
            else:
                expected = raw.mean(dim=1)
            observed = h_f[:, :, user_idx, ant_idx]
            diff = torch.max(torch.abs(expected - observed)).item()
            max_abs_diff = max(max_abs_diff, float(diff))

    payload.update(
        {
            "validation_status": "ok",
            "sionna_channel_tensor_shape": [int(x) for x in h.shape],
            "extracted_h_f_shape": [int(x) for x in h_f.shape],
            "selected_data_symbol_indices": selected_symbols,
            "selected_effective_subcarrier_indices": selected_subcarriers,
            "rx_dimension_maps_to_users": int(h.size(1)) == int(h_f.size(2)),
            "tx_ant_dimension_maps_to_nt": int(h.size(4)) == int(h_f.size(3)),
            "selected_ofdm_symbol_is_data_bearing": all(idx in data_symbol_indices for idx in selected_symbols),
            "effective_subcarrier_count_matches_nsc": len(selected_subcarriers) == int(h_f.size(1)),
            "complex_dtype": bool(torch.is_complex(h_f)),
            "finite_values": bool(torch.isfinite(h_f.real).all() and torch.isfinite(h_f.imag).all()),
            "axis_spot_check_passed": bool(max_abs_diff < 1e-6),
            "spot_check_max_abs_diff": float(max_abs_diff),
            "matrix_stats": summarize_h_f_matrix_stats(h_f),
            "hidden_squeeze_transpose_risk": (
                "low_for_current_num_tx_equals_1_and_rx_ant_equals_1_bridge_but_still_explicit_if_future_multi_tx_or_multi_rx_ant_paths_are_added"
            ),
        }
    )
    payload["notes"] = [
        "Current converter assumes Sionna axes [batch, rx, rx_ant, tx, tx_ant, ofdm_symbol, fft_bin] and project axes [batch, subcarrier, user, bs_ant].",
        f"Selected OFDM symbol indices are data-bearing indices {selected_symbols}; available data indices are {data_symbol_indices}.",
        f"Selected effective FFT-bin indices are {selected_subcarriers}.",
        "Spot-check compares every extracted H_f[:, :, k, n] slice against the corresponding raw Sionna h[:, k, 0, 0, n, ofdm_symbol, fft_bin] slice.",
    ]
    write_json(out_path, payload)
    write_markdown(md_path, _to_md(payload))
    print(f"Saved extracted H_f axis validation to {out_path}")


if __name__ == "__main__":
    main()
