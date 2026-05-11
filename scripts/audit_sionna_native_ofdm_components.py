#!/usr/bin/env python
"""Audit Sionna-native OFDM components against the currently installed API."""

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
    COMPONENT_SPECS,
    ComponentProbeResult,
    describe_signature,
    format_audit_markdown,
    load_component,
    write_json,
    write_markdown,
)

AUDIT_COMPONENT_ORDER = [
    "ResourceGrid",
    "ResourceGridMapper",
    "ResourceGridDemapper",
    "OFDMChannel",
    "ApplyOFDMChannel",
    "LSChannelEstimator",
    "LMMSEEqualizer",
    "RemoveNulledSubcarriers",
    "Mapper",
    "Demapper",
    "BinarySource",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _probe_component(name: str) -> ComponentProbeResult:
    obj, module_path, error = load_component(name)
    if obj is None:
        return ComponentProbeResult(
            name=name,
            import_ok=False,
            module_path=None,
            constructor_signature=None,
            callable_probe_ok=False,
            notes=["Component could not be imported from the expected modules."],
            error=error,
        )

    signature = describe_signature(obj)
    notes: list[str] = []
    probe_ok = False
    try:
        if name == "ResourceGrid":
            rg = obj(
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
            notes.append(f"Created test grid with num_data_symbols={int(rg.num_data_symbols)}.")
            probe_ok = True
        elif name == "ResourceGridMapper":
            ResourceGrid, _, _ = load_component("ResourceGrid")
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
            mapper = obj(rg)
            x = torch.randn(2, 1, 1, int(rg.num_data_symbols), dtype=torch.complex64)
            y = mapper(x)
            notes.append(f"Mapping probe shape: {tuple(y.shape)}.")
            probe_ok = True
        elif name == "ResourceGridDemapper":
            ResourceGrid, _, _ = load_component("ResourceGrid")
            StreamManagement, _, _ = load_component("StreamManagement")
            ResourceGridMapper, _, _ = load_component("ResourceGridMapper")
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
            mapper = ResourceGridMapper(rg)
            demapper = obj(rg, sm)
            x = torch.randn(2, 1, 1, int(rg.num_data_symbols), dtype=torch.complex64)
            y = demapper(mapper(x))
            notes.append(f"Demapping probe shape: {tuple(y.shape)}.")
            probe_ok = True
        elif name == "RemoveNulledSubcarriers":
            ResourceGrid, _, _ = load_component("ResourceGrid")
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
            block = obj(rg)
            y = block(torch.randn(2, 1, 1, 4, 16, dtype=torch.complex64))
            notes.append(f"Effective subcarrier probe shape: {tuple(y.shape)}.")
            probe_ok = True
        elif name == "BinarySource":
            source = obj()
            bits = source([2, 1, 1, 10])
            notes.append(f"Generated bit tensor shape: {tuple(bits.shape)}.")
            probe_ok = True
        elif name == "Mapper":
            mapper = obj("qam", 2)
            bits = torch.randint(0, 2, (2, 1, 1, 10), dtype=torch.float32)
            symbols = mapper(bits)
            notes.append(f"Mapped symbol tensor shape: {tuple(symbols.shape)}.")
            probe_ok = True
        elif name == "Demapper":
            demapper = obj("app", "qam", 2, hard_out=False)
            rx = torch.randn(2, 1, 1, 5, dtype=torch.complex64)
            llr = demapper(rx, torch.full((2, 1, 1, 1), 0.01, dtype=torch.float32))
            notes.append(f"Demapper output shape: {tuple(llr.shape)}.")
            probe_ok = True
        elif name == "OFDMChannel":
            ResourceGrid, _, _ = load_component("ResourceGrid")
            RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
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
            channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1)
            block = obj(channel_model, rg, return_channel=True)
            x = torch.randn(2, 1, 1, 4, 16, dtype=torch.complex64)
            y, h = block(x, no=torch.full((2, 1, 1), 0.01, dtype=torch.float32))
            notes.append(f"OFDMChannel output shape: {tuple(y.shape)}, channel shape: {tuple(h.shape)}.")
            probe_ok = True
        elif name == "ApplyOFDMChannel":
            block = obj()
            x = torch.randn(2, 1, 1, 4, 16, dtype=torch.complex64)
            h = torch.ones(2, 1, 1, 1, 1, 4, 16, dtype=torch.complex64)
            y = block(x, h, no=torch.full((2, 1, 1), 0.01, dtype=torch.float32))
            notes.append(f"ApplyOFDMChannel output shape: {tuple(y.shape)}.")
            probe_ok = True
        elif name == "LSChannelEstimator":
            ResourceGrid, _, _ = load_component("ResourceGrid")
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
            est = obj(rg)
            y = torch.randn(2, 1, 1, 4, 16, dtype=torch.complex64)
            h_hat, err_var = est(y, torch.full((2, 1, 1), 0.01, dtype=torch.float32))
            notes.append(f"LS estimate shape: {tuple(h_hat.shape)}, error shape: {tuple(err_var.shape)}.")
            probe_ok = True
        elif name == "LMMSEEqualizer":
            ResourceGrid, _, _ = load_component("ResourceGrid")
            StreamManagement, _, _ = load_component("StreamManagement")
            ResourceGridMapper, _, _ = load_component("ResourceGridMapper")
            Mapper, _, _ = load_component("Mapper")
            BinarySource, _, _ = load_component("BinarySource")
            OFDMChannel, _, _ = load_component("OFDMChannel")
            RayleighBlockFading, _, _ = load_component("RayleighBlockFading")
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
            eq = obj(rg, sm)
            source = BinarySource()
            mapper = Mapper("qam", 2)
            rg_mapper = ResourceGridMapper(rg)
            bits = source([2, 1, 1, int(rg.num_data_symbols * 2)])
            x = rg_mapper(mapper(bits))
            channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1)
            channel = OFDMChannel(channel_model, rg, return_channel=True)
            y, _ = channel(x, no=torch.full((2, 1, 1), 0.01, dtype=torch.float32))
            estimator = load_component("LSChannelEstimator")[0](rg)
            h_hat, err_var = estimator(y, torch.full((2, 1, 1), 0.01, dtype=torch.float32))
            x_hat, no_eff = eq(y, h_hat, err_var, torch.full((2, 1, 1), 0.01, dtype=torch.float32))
            notes.append(f"LMMSE equalizer output shape: {tuple(x_hat.shape)}, no_eff shape: {tuple(no_eff.shape)}.")
            probe_ok = True
        else:
            notes.append("Import/signature inspection succeeded, but no callable probe is implemented for this symbol.")
            probe_ok = True
    except Exception as exc:  # pragma: no cover - depends on optional runtime API
        notes.append(f"Callable probe failed: {type(exc).__name__}: {exc}")
        probe_ok = False

    return ComponentProbeResult(
        name=name,
        import_ok=True,
        module_path=module_path,
        constructor_signature=signature,
        callable_probe_ok=probe_ok,
        notes=notes,
        error=error,
    )


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env_info = collect_sionna_env_info()

    payload: dict[str, Any] = {
        "sionna_import_ok": env_info["sionna_import_ok"],
        "sionna_version": env_info["sionna_version"],
        "components": [],
        "recommended_next_component": "ResourceGrid",
        "notes": [],
    }

    if env_info["sionna_import_ok"]:
        probes = [_probe_component(name) for name in AUDIT_COMPONENT_ORDER]
        payload["components"] = [probe.to_dict() for probe in probes]
        component_map = {probe.name: probe for probe in probes}
        chain_priority = [
            "BinarySource",
            "Mapper",
            "ResourceGrid",
            "ResourceGridMapper",
            "OFDMChannel",
            "LSChannelEstimator",
            "LMMSEEqualizer",
            "Demapper",
        ]
        for name in chain_priority:
            probe = component_map[name]
            if not probe.callable_probe_ok:
                payload["recommended_next_component"] = name
                break
        else:
            payload["recommended_next_component"] = "frequency-domain per-subcarrier beamforming insertion"
        payload["notes"] = [
            "This audit uses the currently installed Sionna 2.x API instead of assuming older module paths.",
            "In the current install, OFDMChannel and ApplyOFDMChannel resolve from sionna.phy.channel.",
        ]
    else:
        payload["notes"] = [
            "Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt` and rerun.",
        ]

    write_json(out_path, payload)
    write_markdown(md_path, format_audit_markdown(payload))
    print(f"Saved Sionna native OFDM component audit to {out_path}")


if __name__ == "__main__":
    main()
