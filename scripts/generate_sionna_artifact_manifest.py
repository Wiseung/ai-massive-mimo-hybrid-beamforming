#!/usr/bin/env python
"""Generate a manifest for optional Sionna smoke artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "path": "outputs/sionna_smoke/sionna_api_summary.json",
        "description": "Base Sionna PHY/API environment introspection summary.",
        "command": "python scripts/inspect_sionna_api.py --out outputs/sionna_smoke/sionna_api_summary.json",
    },
    {
        "path": "outputs/sionna_smoke/sionna_phy_awgn_summary.json",
        "description": "Minimal Sionna PHY AWGN smoke demo summary.",
        "command": "python scripts/sionna_phy_awgn_demo.py --out outputs/sionna_smoke/sionna_phy_awgn_summary.json",
    },
    {
        "path": "outputs/sionna_smoke/sionna_phy_beamforming_link_summary.json",
        "description": "Sionna PHY AWGN plus PyTorch beamforming link smoke summary.",
        "command": "python scripts/sionna_phy_beamforming_link_demo.py --out outputs/sionna_smoke/sionna_phy_beamforming_link_summary.json",
    },
    {
        "path": "outputs/sionna_smoke/sionna_ofdm_api_summary.json",
        "description": "Sionna OFDM symbol availability introspection summary.",
        "command": "python scripts/inspect_sionna_ofdm_api.py --out outputs/sionna_smoke/sionna_ofdm_api_summary.json",
    },
    {
        "path": "outputs/sionna_smoke/sionna_ofdm_resource_grid_summary.json",
        "description": "Minimal Sionna OFDM ResourceGrid smoke demo summary.",
        "command": "python scripts/sionna_ofdm_resource_grid_demo.py --out outputs/sionna_smoke/sionna_ofdm_resource_grid_summary.json",
    },
    {
        "path": "outputs/sionna_smoke/sionna_ofdm_beamforming_bridge_summary.json",
        "description": "Sionna OFDM grid plus PyTorch beamforming bridge summary.",
        "command": "python scripts/sionna_ofdm_beamforming_bridge_demo.py --out outputs/sionna_smoke/sionna_ofdm_beamforming_bridge_summary.json",
    },
    {
        "path": "outputs/sionna_smoke/differentiable_beamformer_gradcheck.json",
        "description": "Gradient check output for the tiny differentiable beamformer.",
        "command": "python scripts/check_differentiable_beamformer_gradients.py --out outputs/sionna_smoke/differentiable_beamformer_gradcheck.json",
    },
    {
        "path": "outputs/sionna_smoke/sionna_ofdm_differentiable_beamforming_summary.json",
        "description": "Few-step differentiable OFDM beamforming smoke demo summary.",
        "command": "python scripts/sionna_ofdm_differentiable_beamforming_demo.py --out outputs/sionna_smoke/sionna_ofdm_differentiable_beamforming_summary.json",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _bool_or_none(payload: dict[str, Any], key: str) -> bool | None:
    if key not in payload:
        return None
    return bool(payload[key])


def _artifact_row(item: dict[str, str], commit: str) -> dict[str, Any]:
    path = Path(item["path"])
    row: dict[str, Any] = {
        "path": item["path"],
        "description": item["description"],
        "generating_command": item["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        "uses_real_sionna_phy_or_ofdm": None,
        "fallback_used": None,
        "is_full_e2e": False,
    }
    if not path.exists():
        return row

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return row

    used_phy = _bool_or_none(payload, "used_sionna_phy")
    used_ofdm = _bool_or_none(payload, "used_sionna_ofdm")
    used_channel = _bool_or_none(payload, "used_sionna_channel")
    uses_real = any(value is True for value in (used_phy, used_ofdm, used_channel))
    row["uses_real_sionna_phy_or_ofdm"] = uses_real
    row["fallback_used"] = _bool_or_none(payload, "fallback_used")
    return row


def main() -> None:
    args = parse_args()
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_json.with_suffix(".md")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    rows = [_artifact_row(item, commit) for item in ARTIFACTS]
    payload = {
        "generated_from_commit": commit,
        "note": "Sionna artifact manifest lists optional PHY/OFDM smoke outputs. None of these artifacts represent a full Sionna end-to-end, RT, or 5G NR full-stack benchmark.",
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: optional Sionna smoke artifacts only; full end-to-end, RT, and 5G NR full-stack claims remain out of scope.",
        "",
        "| path | exists | real_sionna | fallback_used | full_e2e | command |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['path']} | {row['exists']} | {row['uses_real_sionna_phy_or_ofdm']} | {row['fallback_used']} | {row['is_full_e2e']} | `{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved Sionna artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
