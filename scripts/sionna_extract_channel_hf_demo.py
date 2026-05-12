#!/usr/bin/env python
"""Demonstrate project-side H_f extraction from a Sionna channel tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_channel_extraction import (
    compare_extracted_h_f_with_synthetic_reference,
    extract_h_f_from_sionna_channel,
    validate_extracted_h_f,
)
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import build_pilot_aware_multiuser_resource_grid, compute_project_precoder_per_subcarrier
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _md(payload: dict) -> list[str]:
    return [
        "# Sionna Extract H_f Demo",
        "",
        f"- extraction_success: `{payload['extraction_success']}`",
        f"- sionna_channel_component_used: `{payload['sionna_channel_component_used']}`",
        f"- sionna_channel_tensor_shape: `{payload['sionna_channel_tensor_shape']}`",
        f"- extracted_h_f_shape: `{payload['extracted_h_f_shape']}`",
        f"- project_h_f_shape_compatible: `{payload['project_h_f_shape_compatible']}`",
        f"- used_for_project_rzf: `{payload['used_for_project_rzf']}`",
        f"- fallback_used: `{payload['fallback_used']}`",
        f"- fallback_reason: `{payload['fallback_reason']}`",
        f"- h_f_norm_mean: `{payload['h_f_norm_mean']}`",
        f"- h_f_rank_stats: `{payload['h_f_rank_stats']}`",
        f"- approximate_sum_rate_if_available: `{payload['approximate_sum_rate_if_available']}`",
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sionna_device = resolve_sionna_device(device)

    summary = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "extraction_success": False,
        "sionna_channel_component_used": None,
        "sionna_channel_tensor_shape": None,
        "extracted_h_f_shape": None,
        "project_h_f_shape_compatible": False,
        "used_for_project_rzf": False,
        "fallback_used": False,
        "fallback_reason": "",
        "h_f_norm_mean": None,
        "h_f_rank_stats": None,
        "approximate_sum_rate_if_available": None,
        "notes": [],
    }
    if not env["sionna_import_ok"]:
        summary["fallback_used"] = True
        summary["fallback_reason"] = "Sionna not installed"
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved extracted H_f demo summary to {out_path}")
        return

    OFDMChannel, _, _ = load_component("OFDMChannel")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    rg, _, rg_meta = build_pilot_aware_multiuser_resource_grid(
        num_users=4,
        num_effective_subcarriers=16,
        num_ofdm_symbols=2,
        device=device,
    )
    if rg is None or OFDMChannel is None or RayleighBlockFading is None:
        summary["fallback_used"] = True
        summary["fallback_reason"] = "required_sionna_channel_components_unavailable"
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved extracted H_f demo summary to {out_path}")
        return

    channel_model = RayleighBlockFading(num_rx=4, num_rx_ant=1, num_tx=1, num_tx_ant=16, device=sionna_device)
    channel = OFDMChannel(channel_model, rg, return_channel=True, device=sionna_device)
    dummy_x = torch.zeros(8, 1, 16, rg.num_ofdm_symbols, rg.fft_size, dtype=torch.complex64, device=device)
    _, h = channel(dummy_x, no=torch.full((8, 4, 1), 0.1, dtype=torch.float32, device=device))
    summary["sionna_channel_component_used"] = "OFDMChannel(return_channel=True)"
    summary["sionna_channel_tensor_shape"] = [int(v) for v in h.shape]

    h_f, meta, success, fallback_reason = extract_h_f_from_sionna_channel(
        h,
        resource_grid=rg,
        num_users=4,
        num_bs_ant=16,
    )
    summary["extraction_success"] = bool(success)
    summary["project_h_f_shape_compatible"] = bool(success and h_f is not None and list(h_f.shape) == [8, 16, 4, 16])
    summary["extracted_h_f_shape"] = [int(v) for v in h_f.shape] if h_f is not None else None
    summary["fallback_used"] = not success
    summary["fallback_reason"] = fallback_reason
    summary["extraction_meta"] = meta
    summary["notes"].append(f"ResourceGrid metadata: {rg_meta}")

    if h_f is not None:
        validation = validate_extracted_h_f(h_f)
        summary["validation"] = validation
        summary["h_f_norm_mean"] = validation["norm_mean"]
        summary["h_f_rank_stats"] = {
            "rank_mean": float(torch.linalg.matrix_rank(h_f).float().mean().item()),
            "rank_min": int(torch.linalg.matrix_rank(h_f).min().item()),
            "rank_max": int(torch.linalg.matrix_rank(h_f).max().item()),
        }
        synthetic_ref = (torch.randn_like(h_f.real) + 1j * torch.randn_like(h_f.real)).to(torch.complex64) / torch.sqrt(torch.tensor(2.0, device=device))
        summary["synthetic_reference_comparison"] = compare_extracted_h_f_with_synthetic_reference(h_f, synthetic_ref)
        precoder = compute_project_precoder_per_subcarrier("rzf", h_f, 0.1)
        signal = torch.matmul(precoder, torch.ones(8, 16, 4, 1, dtype=torch.complex64, device=device)).squeeze(-1)
        rx = torch.sum(torch.abs(torch.einsum("bskn,bsn->bsk", h_f, signal)) ** 2, dim=-1)
        summary["approximate_sum_rate_if_available"] = float(torch.mean(torch.log2(1.0 + rx / 0.1)).item())
        summary["used_for_project_rzf"] = True

    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved extracted H_f demo summary to {out_path}")


if __name__ == "__main__":
    main()
