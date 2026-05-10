#!/usr/bin/env python
"""Inspect Sionna OFDM-related API symbols for the current install."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info


EXPECTED_SYMBOLS = [
    "ResourceGrid",
    "ResourceGridMapper",
    "ResourceGridDemapper",
    "LSChannelEstimator",
    "LMMSEEqualizer",
    "OFDMChannel",
    "ApplyOFDMChannel",
    "RemoveNulledSubcarriers",
    "ZFPrecoder",
    "RZFPrecoder",
    "LinearDetector",
    "MaximumLikelihoodDetector",
    "StreamManagement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_info = collect_sionna_env_info()
    payload: dict[str, object] = {
        "sionna_version": env_info["sionna_version"],
        "module_import_ok": False,
        "available_ofdm_symbols": [],
        "missing_expected_symbols": EXPECTED_SYMBOLS,
        "public_name_count": 0,
        "public_names_sample": [],
        "recommendation": "Install sionna-no-rt or sionna and rerun the OFDM API inspection.",
    }

    if not env_info["sionna_import_ok"]:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved Sionna OFDM API summary to {out_path}")
        return

    ofdm = importlib.import_module("sionna.phy.ofdm")
    channel = importlib.import_module("sionna.phy.channel")
    mimo = importlib.import_module("sionna.phy.mimo")
    public_names = sorted(name for name in dir(ofdm) if not name.startswith("_"))
    available: list[str] = []
    missing: list[str] = []

    for name in EXPECTED_SYMBOLS:
        if hasattr(ofdm, name) or hasattr(channel, name) or hasattr(mimo, name):
            available.append(name)
        else:
            missing.append(name)

    if {"ResourceGrid", "ResourceGridMapper", "ResourceGridDemapper", "RemoveNulledSubcarriers"}.issubset(set(available)):
        recommendation = "ResourceGrid-based OFDM smoke demo should use real Sionna OFDM components."
    else:
        recommendation = "Use torch fallback for OFDM grid handling and record missing Sionna OFDM symbols explicitly."

    payload.update(
        {
            "module_import_ok": True,
            "available_ofdm_symbols": available,
            "missing_expected_symbols": missing,
            "public_name_count": len(public_names),
            "public_names_sample": public_names[:40],
            "recommendation": recommendation,
        }
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Sionna OFDM API summary to {out_path}")


if __name__ == "__main__":
    main()
