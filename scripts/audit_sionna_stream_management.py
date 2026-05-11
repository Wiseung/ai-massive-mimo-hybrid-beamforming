#!/usr/bin/env python
"""Audit StreamManagement for minimal and beamformed Sionna receiver chains."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import numpy as np
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import load_component, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _describe(sm: Any, association: np.ndarray, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "rx_tx_association": association.tolist(),
        "num_rx": int(sm.num_rx),
        "num_tx": int(sm.num_tx),
        "num_streams_per_tx": int(sm.num_streams_per_tx),
        "num_streams_per_rx": int(sm.num_streams_per_rx),
        "num_interfering_streams_per_rx": int(sm.num_interfering_streams_per_rx),
        "stream_ind": sm.stream_ind.tolist(),
        "detection_desired_ind": sm.detection_desired_ind.tolist(),
        "detection_undesired_ind": sm.detection_undesired_ind.tolist(),
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    payload: dict[str, Any] = {
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "cases": [],
        "summary": {},
    }
    if not env["sionna_import_ok"]:
        write_json(out_path, payload)
        write_markdown(md_path, ["# StreamManagement Audit", "", "- Sionna not installed."])
        print(f"Saved StreamManagement audit to {out_path}")
        return

    StreamManagement, _, _ = load_component("StreamManagement")
    ResourceGrid, _, _ = load_component("ResourceGrid")
    LMMSEEqualizer, _, _ = load_component("LMMSEEqualizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    minimal_assoc = np.array([[1]], dtype=int)
    minimal_sm = StreamManagement(minimal_assoc, num_streams_per_tx=1)
    payload["cases"].append(_describe(minimal_sm, minimal_assoc, "minimal_success"))

    failing_assoc = np.ones((4, 1), dtype=int)
    failing_sm = StreamManagement(failing_assoc, num_streams_per_tx=1)
    payload["cases"].append(_describe(failing_sm, failing_assoc, "previous_beamformed_failing_sm"))

    recommended_assoc = np.ones((4, 1), dtype=int)
    recommended_sm = StreamManagement(recommended_assoc, num_streams_per_tx=4)
    payload["cases"].append(_describe(recommended_sm, recommended_assoc, "recommended_beamformed_native_sm"))

    rg_ok = ResourceGrid(
        num_ofdm_symbols=2,
        fft_size=19,
        subcarrier_spacing=15_000.0,
        num_tx=1,
        num_streams_per_tx=4,
        num_guard_carriers=(1, 1),
        dc_null=True,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[0],
    )
    LMMSEEqualizer(rg_ok, recommended_sm)

    payload["summary"] = {
        "why_minimal_succeeds": "The minimal chain uses num_tx=1, num_streams_per_tx=1, rx_tx_association=[[1]], and a pilot-aware grid with positive num_data_symbols.",
        "beamformed_mapping_recommendation": "Model the downlink beamformed chain as num_tx=1 and num_streams_per_tx=K, not as K receivers each fed by a single-stream transmitter.",
        "recommended_rx_tx_association": recommended_assoc.tolist(),
        "zero_dimension_from_stream_management": False,
        "stream_management_failure_mode": "The old beamformed setup with rx_tx_association shape (K,1) and num_streams_per_tx=1 does not create zero desired streams, but it mismatches the intended K-stream downlink semantics and contributes to an incompatible equalizer interpretation.",
        "recommended_configuration": {
            "num_tx": 1,
            "num_streams_per_tx": 4,
            "rx_tx_association": recommended_assoc.tolist(),
            "resource_grid_num_ofdm_symbols": 2,
            "pilot_ofdm_symbol_indices": [0],
        },
    }
    lines = [
        "# StreamManagement Audit",
        "",
        "## Summary",
        "",
        f"1. minimal receiver demo succeeds because: {payload['summary']['why_minimal_succeeds']}",
        f"2. beamformed chain should map K users as: `{payload['summary']['beamformed_mapping_recommendation']}`",
        f"3. recommended rx_tx_association: `{payload['summary']['recommended_rx_tx_association']}`",
        f"4. zero dimension came directly from StreamManagement: `{payload['summary']['zero_dimension_from_stream_management']}`",
        "",
        "| Case | num_rx | num_tx | num_streams_per_tx | desired_ind | undesired_ind |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| {row['name']} | {row['num_rx']} | {row['num_tx']} | {row['num_streams_per_tx']} | "
            f"`{row['detection_desired_ind']}` | `{row['detection_undesired_ind']}` |"
        )
    write_json(out_path, payload)
    write_markdown(md_path, lines)
    print(f"Saved StreamManagement audit to {out_path}")


if __name__ == "__main__":
    main()
