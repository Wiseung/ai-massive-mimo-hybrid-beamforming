#!/usr/bin/env python
"""Audit ResourceGrid pilot-pattern requirements for the Sionna receiver chain."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import numpy as np

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import load_component, resolve_sionna_device, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _describe_rg(rg: Any) -> dict[str, Any]:
    return {
        "num_ofdm_symbols": int(rg.num_ofdm_symbols),
        "fft_size": int(rg.fft_size),
        "num_tx": int(rg.num_tx),
        "num_streams_per_tx": int(rg.num_streams_per_tx),
        "pilot_pattern_type": type(rg.pilot_pattern).__name__,
        "num_data_symbols": int(rg.num_data_symbols),
        "num_pilot_symbols": int(rg.num_pilot_symbols),
        "effective_subcarrier_ind": [int(x) for x in rg.effective_subcarrier_ind],
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    payload: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "resource_grid_configs": [],
        "summary": {},
        "notes": [],
    }

    if not env["sionna_import_ok"]:
        payload["notes"] = ["Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`."]
        write_json(out_path, payload)
        write_markdown(md_path, ["# Pilot Pattern Audit", "", "- Sionna not installed."])
        print(f"Saved pilot pattern audit to {out_path}")
        return

    ResourceGrid, _, _ = load_component("ResourceGrid")
    LSChannelEstimator, _, _ = load_component("LSChannelEstimator")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    StreamManagement, _, _ = load_component("StreamManagement")

    configs = [
        {"name": "empty_default", "pilot_pattern": None, "pilot_ofdm_symbol_indices": None},
        {"name": "kronecker_one_pilot_symbol", "pilot_pattern": "kronecker", "pilot_ofdm_symbol_indices": [0]},
        {"name": "kronecker_two_pilot_symbols", "pilot_pattern": "kronecker", "pilot_ofdm_symbol_indices": [0, 1]},
    ]
    for cfg in configs:
        kwargs = dict(
            num_ofdm_symbols=4,
            fft_size=16,
            subcarrier_spacing=15_000.0,
            num_tx=1,
            num_streams_per_tx=1,
            num_guard_carriers=(1, 1),
            dc_null=True,
        )
        if cfg["pilot_pattern"] is not None:
            kwargs["pilot_pattern"] = cfg["pilot_pattern"]
            kwargs["pilot_ofdm_symbol_indices"] = cfg["pilot_ofdm_symbol_indices"]
        rg = ResourceGrid(**kwargs)
        row = {"name": cfg["name"], **_describe_rg(rg), "ls_estimator_ok": False, "lmmse_ctor_ok": False, "notes": []}
        try:
            LSChannelEstimator(rg)
            row["ls_estimator_ok"] = True
        except Exception as exc:  # pragma: no cover
            row["notes"].append(f"LSChannelEstimator failed: {type(exc).__name__}: {exc}")
        try:
            sm = StreamManagement(np.array([[1]]), num_streams_per_tx=1)
            LMMSEEqualizer(rg, sm)
            row["lmmse_ctor_ok"] = True
        except Exception as exc:  # pragma: no cover
            row["notes"].append(f"LMMSEEqualizer ctor failed: {type(exc).__name__}: {exc}")
        payload["resource_grid_configs"].append(row)

    payload["summary"] = {
        "why_pilot_pattern_empty": "beamforming_chain used pilot_pattern=None, which creates EmptyPilotPattern and causes LSChannelEstimator to raise AssertionError.",
        "ls_estimator_working_config": "pilot_pattern='kronecker' with pilot_ofdm_symbol_indices=[0]",
        "pilot_indices_required": True,
        "recommended_resource_grid_config": {
            "num_ofdm_symbols": 4,
            "fft_size": 16,
            "num_tx": 1,
            "num_streams_per_tx": 1,
            "pilot_pattern": "kronecker",
            "pilot_ofdm_symbol_indices": [0],
            "num_guard_carriers": [1, 1],
            "dc_null": True,
        },
    }
    payload["notes"] = [
        "EmptyPilotPattern is valid for data-only chains but not for LSChannelEstimator.",
        "The current minimal receiver-success path uses a non-empty Kronecker pilot pattern.",
    ]
    md_lines = [
        "# Sionna ResourceGrid Pilot Audit",
        "",
        f"- Current pilot-pattern failure cause: {payload['summary']['why_pilot_pattern_empty']}",
        f"- Working LS estimator config: `{payload['summary']['ls_estimator_working_config']}`",
        f"- Need pilot_ofdm_symbol_indices: `{payload['summary']['pilot_indices_required']}`",
        "",
        "| Config | Pilot Pattern | Pilot Symbols | Data Symbols | LS OK | LMMSE ctor OK |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["resource_grid_configs"]:
        md_lines.append(
            f"| {row['name']} | {row['pilot_pattern_type']} | {row['num_pilot_symbols']} | {row['num_data_symbols']} | {row['ls_estimator_ok']} | {row['lmmse_ctor_ok']} |"
        )
    write_json(out_path, payload)
    write_markdown(md_path, md_lines)
    print(f"Saved pilot pattern audit to {out_path}")


if __name__ == "__main__":
    main()
