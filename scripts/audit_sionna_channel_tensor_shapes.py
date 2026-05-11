#!/usr/bin/env python
"""Audit Sionna channel tensor shapes relevant to native H_f extraction."""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_channel_extraction import extract_h_f_from_sionna_channel
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import build_pilot_aware_multiuser_resource_grid
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_json, write_markdown


COMPONENTS = [
    ("OFDMChannel", "sionna.phy.channel"),
    ("ApplyOFDMChannel", "sionna.phy.channel"),
    ("RayleighBlockFading", "sionna.phy.channel"),
    ("GenerateOFDMChannel", "sionna.phy.channel"),
    ("cir_to_ofdm_channel", "sionna.phy.channel"),
    ("subcarrier_frequencies", "sionna.phy.channel"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _sig(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {type(exc).__name__}: {exc}"


def _load(module_path: str, name: str) -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, name), None
    except Exception as exc:  # pragma: no cover
        return None, f"{type(exc).__name__}: {exc}"


def _md(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Sionna Channel Tensor Audit",
        "",
        f"- Sionna import ok: `{payload['sionna_import_ok']}`",
        f"- Sionna version: `{payload['sionna_version']}`",
        f"- Recommended extraction path: `{payload['recommended_extraction_path']}`",
        "",
        "| Component | Import OK | Constructor | Call | Channel Returned | Output Shape | Probe OK |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["components"]:
        lines.append(
            f"| {row['name']} | {row['import_ok']} | `{row['constructor_signature'] or ''}` | `{row['call_signature'] or ''}` | "
            f"{row['returns_channel_tensor']} | {row['observed_output_shape']} | {row['minimal_probe_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            f"- OFDMChannel returns channel tensor: `{payload['summary']['ofdmchannel_returns_channel_tensor']}`",
            f"- observed channel tensor shape: `{payload['summary']['ofdmchannel_channel_tensor_shape']}`",
            f"- project H_f conversion possible: `{payload['summary']['can_convert_to_project_h_f']}`",
            f"- project H_f shape: `{payload['summary']['project_h_f_shape']}`",
        ]
    )
    for note in payload["notes"]:
        lines.append(f"- {note}")
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sionna_device = resolve_sionna_device(device)

    payload: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "components": [],
        "summary": {
            "ofdmchannel_returns_channel_tensor": False,
            "ofdmchannel_channel_tensor_shape": None,
            "channel_tensor_axes_interpretation": ["batch", "rx", "rx_ant", "tx", "tx_ant", "ofdm_symbol", "fft_bin"],
            "can_convert_to_project_h_f": False,
            "project_h_f_shape": None,
        },
        "recommended_extraction_path": "unavailable",
        "notes": [],
    }
    if not env["sionna_import_ok"]:
        payload["notes"] = ["Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`."]
        write_json(out_path, payload)
        write_markdown(md_path, _md(payload))
        print(f"Saved Sionna channel tensor audit to {out_path}")
        return

    ResourceGrid, _, _ = load_component("ResourceGrid")
    OFDMChannel, _, _ = load_component("OFDMChannel")
    RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
    rg, _, rg_meta = build_pilot_aware_multiuser_resource_grid(
        num_users=4,
        num_effective_subcarriers=16,
        num_ofdm_symbols=2,
        device=device,
    )
    observed_channel = None
    for name, module_path in COMPONENTS:
        symbol, error = _load(module_path, name)
        row = {
            "name": name,
            "module_path": module_path,
            "import_ok": symbol is not None,
            "constructor_signature": _sig(symbol) if symbol is not None else None,
            "call_signature": _sig(getattr(symbol, "call")) if symbol is not None and hasattr(symbol, "call") else None,
            "returns_channel_tensor": False,
            "observed_output_shape": None,
            "channel_tensor_shape": None,
            "minimal_probe_passed": False,
            "error": error,
            "notes": [],
        }
        if symbol is not None and name == "OFDMChannel" and rg is not None and RayleighBlockFading is not None:
            try:
                channel_model = RayleighBlockFading(num_rx=4, num_rx_ant=1, num_tx=1, num_tx_ant=16, device=sionna_device)
                block = symbol(channel_model, rg, return_channel=True, device=sionna_device)
                x = torch.zeros(2, 1, 16, rg.num_ofdm_symbols, rg.fft_size, dtype=torch.complex64, device=device)
                y, h = block(x, no=torch.full((2, 4, 1), 0.1, dtype=torch.float32, device=device))
                observed_channel = h
                row["minimal_probe_passed"] = True
                row["returns_channel_tensor"] = True
                row["observed_output_shape"] = [int(v) for v in y.shape]
                row["channel_tensor_shape"] = [int(v) for v in h.shape]
                row["notes"].append("return_channel=True returns (y, h).")
            except Exception as exc:  # pragma: no cover
                row["notes"].append(f"Probe failed: {type(exc).__name__}: {exc}")
        elif symbol is not None and name == "ApplyOFDMChannel":
            row["notes"].append("ApplyOFDMChannel applies a provided channel tensor and does not generate one.")
        elif symbol is not None and name == "GenerateOFDMChannel":
            row["notes"].append("GenerateOFDMChannel is available only if present in the installed Sionna build.")
        payload["components"].append(row)

    if observed_channel is not None and rg is not None:
        h_f, meta, success, _ = extract_h_f_from_sionna_channel(
            observed_channel,
            resource_grid=rg,
            num_users=4,
            num_bs_ant=16,
        )
        payload["summary"]["ofdmchannel_returns_channel_tensor"] = True
        payload["summary"]["ofdmchannel_channel_tensor_shape"] = [int(v) for v in observed_channel.shape]
        payload["summary"]["can_convert_to_project_h_f"] = bool(success)
        payload["summary"]["project_h_f_shape"] = [int(v) for v in h_f.shape] if h_f is not None else None
        payload["summary"]["conversion_meta"] = meta
        payload["recommended_extraction_path"] = "OFDMChannel(return_channel=True) -> extract_h_f_from_sionna_channel"
    else:
        payload["recommended_extraction_path"] = "fallback to synthetic/project-assisted H_f"

    payload["notes"] = [
        "Current observed OFDMChannel channel tensor axes are interpreted as batch/rx/rx_ant/tx/tx_ant/ofdm_symbol/fft_bin.",
        "The current bridge assumes num_tx=1, rx=user, rx_ant=1, and tx_ant=Nt for MU downlink extraction.",
        f"ResourceGrid metadata for the probe: {rg_meta}",
    ]
    write_json(out_path, payload)
    write_markdown(md_path, _md(payload))
    print(f"Saved Sionna channel tensor audit to {out_path}")


if __name__ == "__main__":
    main()
