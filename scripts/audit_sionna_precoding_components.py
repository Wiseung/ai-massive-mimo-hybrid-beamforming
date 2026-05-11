#!/usr/bin/env python
"""Audit Sionna precoding-related OFDM/MIMO components."""

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

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import resolve_sionna_device, write_json, write_markdown


COMPONENTS = [
    ("RZFPrecoder", "sionna.phy.ofdm"),
    ("PrecodedChannel", "sionna.phy.ofdm"),
    ("StreamManagement", "sionna.phy.mimo"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
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
        "# Sionna Precoding Component Audit",
        "",
        f"- Sionna import ok: `{payload['sionna_import_ok']}`",
        f"- Sionna version: `{payload['sionna_version']}`",
        f"- Recommendation: `{payload['recommendation']}`",
        "",
        "| Component | Import OK | Constructor | Call | Expected Input | Expected Output | Probe OK |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["components"]:
        lines.append(
            f"| {row['name']} | {row['import_ok']} | `{row['constructor_signature'] or ''}` | "
            f"`{row['call_signature'] or ''}` | {row['expected_input_shape']} | {row['expected_output_shape']} | {row['minimal_callable_probe_passed']} |"
        )
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
    sionna_device = resolve_sionna_device(device)

    payload: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "components": [],
        "mimo_precoding_module_available": False,
        "mimo_utils_module_available": False,
        "resource_grid_precoder_shape_note": "Expected Sionna RZFPrecoder input x is [B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size].",
        "recommendation": "fallback to project precoder if Sionna API shape is incompatible",
        "notes": [],
    }

    if not env["sionna_import_ok"]:
        payload["notes"] = ["Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`."]
        write_json(out_path, payload)
        write_markdown(md_path, _md(payload))
        print(f"Saved Sionna precoding component audit to {out_path}")
        return

    try:
        importlib.import_module("sionna.phy.mimo.precoding")
        payload["mimo_precoding_module_available"] = True
    except Exception:
        payload["mimo_precoding_module_available"] = False
    try:
        importlib.import_module("sionna.phy.mimo.utils")
        payload["mimo_utils_module_available"] = True
    except Exception:
        payload["mimo_utils_module_available"] = False

    for name, module_path in COMPONENTS:
        symbol, error = _load_symbol(module_path, name)
        row: dict[str, Any] = {
            "name": name,
            "module_path": module_path,
            "import_ok": symbol is not None,
            "constructor_signature": None,
            "call_signature": None,
            "expected_input_shape": None,
            "expected_output_shape": None,
            "minimal_callable_probe_passed": False,
            "error": error,
            "notes": [],
        }
        if symbol is not None:
            row["constructor_signature"] = _sig(symbol)
            row["call_signature"] = _sig(symbol, "call")
            if name == "RZFPrecoder":
                row["expected_input_shape"] = "[B, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] plus h with Sionna channel tensor layout"
                row["expected_output_shape"] = "precoded resource grid, optionally effective channel"
                try:
                    from sionna.phy.ofdm import ResourceGrid, RZFPrecoder
                    from sionna.phy.mimo import StreamManagement

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
                    precoder = RZFPrecoder(rg, sm, return_effective_channel=True, device=sionna_device)
                    x = torch.randn(2, 1, 1, 4, 16, dtype=torch.complex64, device=device)
                    h = torch.randn(2, 1, 1, 1, 1, 4, 16, dtype=torch.complex64, device=device)
                    y, h_eff = precoder(x, h)
                    row["minimal_callable_probe_passed"] = True
                    row["notes"].append(f"Probe output shapes: y={tuple(y.shape)}, h_eff={tuple(h_eff.shape)}.")
                except Exception as exc:  # pragma: no cover
                    row["notes"].append(f"Probe failed: {type(exc).__name__}: {exc}")
            elif name == "PrecodedChannel":
                row["expected_input_shape"] = "effective channel tensor h plus tx_power, optional h_hat"
                row["expected_output_shape"] = "effective post-precoding channel"
                row["notes"].append("Abstract effective-channel helper; not the primary insertion point for project precoders.")
            elif name == "StreamManagement":
                row["expected_input_shape"] = "rx_tx_association ndarray, num_streams_per_tx"
                row["expected_output_shape"] = "stream-management object"
                try:
                    symbol(np.array([[1]]), num_streams_per_tx=1)
                    row["minimal_callable_probe_passed"] = True
                except Exception as exc:  # pragma: no cover
                    row["notes"].append(f"Probe failed: {type(exc).__name__}: {exc}")
        payload["components"].append(row)

    rzf_probe = next(row for row in payload["components"] if row["name"] == "RZFPrecoder")
    if rzf_probe["minimal_callable_probe_passed"]:
        payload["recommendation"] = "use project frequency-domain precoder insertion first; keep Sionna RZFPrecoder as an optional shape-checked reference path"
    else:
        payload["recommendation"] = "fallback to project precoder if Sionna API shape is incompatible"
    payload["notes"] = [
        "Current project channel/precoder tensors use H_f=(B, Nsc, K, Nt) and F_f=(B, Nsc, Nt, K).",
        "Current Sionna RZFPrecoder expects a resource-grid tensor plus a higher-rank channel tensor layout, so direct substitution is not the clean mainline path.",
    ]
    write_json(out_path, payload)
    write_markdown(md_path, _md(payload))
    print(f"Saved Sionna precoding component audit to {out_path}")


if __name__ == "__main__":
    main()
