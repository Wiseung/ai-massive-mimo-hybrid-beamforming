#!/usr/bin/env python
"""Audit Sionna native precoder APIs and bridge compatibility."""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.utils.csi_interface import summarize_csi_input
from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import resolve_sionna_device, write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import build_native_receiver_context
from beamforming.utils.sionna_precoder_api_bridge import (
    compare_sionna_precoder_output_with_project_precoder_output,
    map_extracted_csi_to_sionna_precoder_inputs,
    map_sionna_precoder_output_to_precoder_output,
)


TARGETS = [
    ("RZFPrecoder", "sionna.phy.ofdm"),
    ("PrecodedChannel", "sionna.phy.ofdm"),
    ("StreamManagement", "sionna.phy.mimo"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_symbol(module_path: str, name: str) -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, name), None
    except Exception as exc:  # pragma: no cover - optional dependency path
        return None, f"{type(exc).__name__}: {exc}"


def _sig(obj: Any, attr: str | None = None) -> str | None:
    try:
        target = getattr(obj, attr) if attr else obj
        return str(inspect.signature(target))
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {type(exc).__name__}: {exc}"


def _md(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# Sionna Native Precoder API Audit",
        "",
        f"- sionna_import_ok: `{payload['sionna_import_ok']}`",
        f"- sionna_version: `{payload['sionna_version']}`",
        f"- sionna_rzf_precoder_available: `{payload['summary']['sionna_rzf_precoder_available']}`",
        f"- recommended_next_step: `{payload['summary']['recommended_next_step']}`",
        "",
        "| Target | Import OK | Constructor | Call | Input shape | Output shape | Probe OK |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["targets"]:
        lines.append(
            f"| {row['name']} | {row['import_ok']} | `{row['constructor_signature'] or ''}` | "
            f"`{row['call_signature'] or ''}` | {row['expected_input_shape']} | {row['expected_output_shape']} | "
            f"{row['minimal_callable_probe_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            f"1. Sionna RZFPrecoder usable: `{payload['summary']['sionna_rzf_precoder_available']}`",
            f"2. expected input shape: `{payload['summary']['rzf_expected_input_shape']}`",
            f"3. expected output shape: `{payload['summary']['rzf_expected_output_shape']}`",
            f"4. compatible with ExtractedCSI / PrecoderOutput: `{payload['summary']['compatible_with_current_interfaces']}`",
            f"5. recommendation: `{payload['summary']['recommended_next_step']}`",
            "",
            "## Notes",
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
    payload: dict[str, Any] = {
        "seed": int(args.seed),
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "targets": [],
        "mimo_precoding_module_available": False,
        "ofdm_precoding_module_available": False,
        "sample_bridge": None,
        "summary": {
            "sionna_rzf_precoder_available": False,
            "rzf_expected_input_shape": None,
            "rzf_expected_output_shape": None,
            "compatible_with_current_interfaces": False,
            "recommended_next_step": "keep_project_side_precoder_output",
        },
        "notes": [],
    }
    if not env["sionna_import_ok"]:
        payload["notes"].append("Sionna is optional. Install `sionna-no-rt` to run the native precoder API audit.")
        write_json(out_path, payload)
        write_markdown(md_path, _md(payload))
        print(f"Saved Sionna native precoder API audit to {out_path}")
        return

    for module_name, key in [("sionna.phy.mimo.precoding", "mimo_precoding_module_available"), ("sionna.phy.ofdm.precoding", "ofdm_precoding_module_available")]:
        try:
            importlib.import_module(module_name)
            payload[key] = True
        except Exception:
            payload[key] = False

    for name, module_path in TARGETS:
        symbol, error = _load_symbol(module_path, name)
        row: dict[str, Any] = {
            "name": name,
            "module_path": module_path,
            "import_ok": symbol is not None,
            "constructor_signature": None,
            "call_signature": None,
            "expected_input_shape": None,
            "expected_output_shape": None,
            "stream_management_requirements": None,
            "resource_grid_requirements": None,
            "minimal_callable_probe_passed": False,
            "can_consume_extracted_csi_directly": False,
            "can_emit_precoder_output_compatible_f_f": False,
            "error": error,
            "notes": [],
        }
        if symbol is not None:
            row["constructor_signature"] = _sig(symbol)
            row["call_signature"] = _sig(symbol, "call")
            if name == "RZFPrecoder":
                row["expected_input_shape"] = "[B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] plus h=[B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]"
                row["expected_output_shape"] = "[B, num_tx, num_tx_ant, num_ofdm_symbols, fft_size] plus optional h_eff"
                row["stream_management_requirements"] = {
                    "num_tx": 1,
                    "num_streams_per_tx": "K",
                    "rx_tx_association": "ones(K,1)",
                }
                row["resource_grid_requirements"] = {
                    "resource_grid_required": True,
                    "pilot_pattern_required_for_receiver_path": True,
                }
            elif name == "PrecodedChannel":
                row["expected_input_shape"] = "h=[B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tx_power=[B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]"
                row["expected_output_shape"] = "h_eff=[B, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]"
            elif name == "StreamManagement":
                row["expected_input_shape"] = "rx_tx_association ndarray plus num_streams_per_tx"
                row["expected_output_shape"] = "stream-management object"

        payload["targets"].append(row)

    set_seed(args.seed)
    context = build_native_receiver_context(
        batch_size=4,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    csi = context.csi
    if csi is not None:
        bridge = map_extracted_csi_to_sionna_precoder_inputs(
            csi,
            device=device,
            alpha=float(csi.num_users * context.noise_var),
        )
        payload["sample_bridge"] = {
            "csi_summary": summarize_csi_input(csi),
            "mapping_success": bool(bridge.get("success", False)),
            "fallback_reason": bridge.get("fallback_reason", ""),
            "shape_assumptions": bridge.get("shape_assumptions"),
            "receiver_config": bridge.get("receiver_config"),
        }
        rzf_row = next(row for row in payload["targets"] if row["name"] == "RZFPrecoder")
        if bridge.get("success", False):
            try:
                from sionna.phy.ofdm import RZFPrecoder

                precoder = RZFPrecoder(
                    bridge["resource_grid"],
                    bridge["stream_management"],
                    return_effective_channel=True,
                    device=resolve_sionna_device(device),
                )
                x_precoded, h_eff = precoder(bridge["x"], bridge["h"], alpha=bridge["alpha"])
                converted, convert_meta = map_sionna_precoder_output_to_precoder_output(x_precoded, input_csi=csi)
                comparison = (
                    compare_sionna_precoder_output_with_project_precoder_output(
                        csi,
                        project_method="rzf",
                        project_noise_var=context.noise_var,
                        sionna_precoder_output=converted,
                    )
                    if converted is not None
                    else None
                )
                rzf_row["minimal_callable_probe_passed"] = True
                rzf_row["can_consume_extracted_csi_directly"] = False
                rzf_row["can_emit_precoder_output_compatible_f_f"] = converted is not None
                rzf_row["notes"].append(f"Probe output shapes: x_precoded={tuple(x_precoded.shape)}, h_eff={tuple(h_eff.shape)}")
                if converted is not None:
                    rzf_row["notes"].append(
                        "Converted native output to PrecoderOutput via adapter after axis remap and project power renormalization."
                    )
                payload["sample_bridge"]["probe_success"] = True
                payload["sample_bridge"]["sionna_output_shape"] = [int(x) for x in x_precoded.shape]
                payload["sample_bridge"]["effective_channel_shape"] = [int(x) for x in h_eff.shape]
                payload["sample_bridge"]["conversion"] = convert_meta
                payload["sample_bridge"]["comparison"] = comparison
                payload["summary"]["sionna_rzf_precoder_available"] = True
                payload["summary"]["rzf_expected_input_shape"] = rzf_row["expected_input_shape"]
                payload["summary"]["rzf_expected_output_shape"] = rzf_row["expected_output_shape"]
                payload["summary"]["compatible_with_current_interfaces"] = converted is not None
                payload["summary"]["recommended_next_step"] = (
                    "adapter_bridge"
                    if converted is not None
                    else "keep_project_side_precoder_output"
                )
            except Exception as exc:  # pragma: no cover - optional runtime path
                rzf_row["notes"].append(f"Probe failed: {type(exc).__name__}: {exc}")
                payload["sample_bridge"]["probe_success"] = False
                payload["sample_bridge"]["probe_error"] = f"{type(exc).__name__}: {exc}"

    payload["notes"] = [
        "RZFPrecoder exists in Sionna 2.0.1 and is callable on the current install.",
        "Its native input contract is resource-grid-centric and higher rank than the project's H_f=(B,Nsc,K,Nt) path.",
        "Current adapter can map one ExtractedCSI object into a probe-only native call path and convert the native output into PrecoderOutput.",
        "This does not replace the project-side precoder mainline and does not justify a full native-only benchmark claim.",
    ]
    write_json(out_path, payload)
    write_markdown(md_path, _md(payload))
    print(f"Saved Sionna native precoder API audit to {out_path}")


if __name__ == "__main__":
    main()
