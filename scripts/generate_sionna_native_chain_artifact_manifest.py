#!/usr/bin/env python
"""Generate a manifest for Sionna-native OFDM chain artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "path": "outputs/sionna_native_chain/ofdm_component_audit.json",
        "description": "Sionna-native OFDM component audit.",
        "command": "python scripts/audit_sionna_native_ofdm_components.py --out outputs/sionna_native_chain/ofdm_component_audit.json",
        "project_h_f_assisted": False,
    },
    {
        "path": "outputs/sionna_native_chain/baseline_chain_summary.json",
        "description": "Sionna-native OFDM baseline chain summary.",
        "command": "python scripts/sionna_native_ofdm_baseline_chain.py --out outputs/sionna_native_chain/baseline_chain_summary.json",
        "project_h_f_assisted": False,
    },
    {
        "path": "outputs/sionna_native_chain/precoding_component_audit.json",
        "description": "Sionna-native precoding API audit.",
        "command": "python scripts/audit_sionna_precoding_components.py --out outputs/sionna_native_chain/precoding_component_audit.json",
        "project_h_f_assisted": False,
    },
    {
        "path": "outputs/sionna_native_chain/beamforming_chain_summary.json",
        "description": "Project-side frequency-domain beamforming insertion summary.",
        "command": "python scripts/sionna_native_ofdm_beamforming_chain.py --out outputs/sionna_native_chain/beamforming_chain_summary.json",
        "project_h_f_assisted": True,
    },
    {
        "path": "outputs/sionna_native_chain/pilot_pattern_audit.json",
        "description": "Pilot pattern audit for the Sionna-native receiver chain.",
        "command": "python scripts/audit_sionna_resource_grid_pilots.py --out outputs/sionna_native_chain/pilot_pattern_audit.json",
        "project_h_f_assisted": False,
    },
    {
        "path": "outputs/sionna_native_chain/estimator_equalizer_demo_summary.json",
        "description": "Minimal estimator/equalizer success demo summary.",
        "command": "python scripts/sionna_native_estimator_equalizer_demo.py --out outputs/sionna_native_chain/estimator_equalizer_demo_summary.json",
        "project_h_f_assisted": False,
    },
    {
        "path": "outputs/sionna_native_chain/beamforming_receiver_chain_v2_summary.json",
        "description": "Beamformed native receiver chain v2 summary.",
        "command": "python scripts/sionna_native_ofdm_beamforming_chain.py --out outputs/sionna_native_chain/beamforming_receiver_chain_v2_summary.json --enable-receiver-chain --receiver-mode auto --trace-shapes",
        "project_h_f_assisted": True,
    },
    {
        "path": "outputs/sionna_native_chain/beamformed_receiver_shape_trace.json",
        "description": "Beamformed receiver shape trace summary.",
        "command": "python scripts/trace_sionna_beamformed_receiver_shapes.py --out outputs/sionna_native_chain/beamformed_receiver_shape_trace.json",
        "project_h_f_assisted": True,
    },
    {
        "path": "outputs/sionna_native_chain/stream_management_audit.json",
        "description": "StreamManagement audit for the beamformed receiver chain.",
        "command": "python scripts/audit_sionna_stream_management.py --out outputs/sionna_native_chain/stream_management_audit.json",
        "project_h_f_assisted": False,
    },
    {
        "path": "outputs/sionna_native_chain/learned_beamforming_receiver_summary.json",
        "description": "Learned beamformer insertion summary on the native receiver chain.",
        "command": "python scripts/sionna_native_ofdm_learned_beamforming_chain.py --out outputs/sionna_native_chain/learned_beamforming_receiver_summary.json --receiver-mode auto --trace-shapes",
        "project_h_f_assisted": True,
    },
    {
        "path": "outputs/sionna_native_chain/native_learned_comparison.md",
        "description": "Learned vs analytic native-chain comparison summary.",
        "command": "python scripts/compare_sionna_native_learned_beamforming.py --analytic-summary outputs/sionna_native_chain/beamforming_receiver_chain_v2_summary.json --analytic-metrics outputs/sionna_native_chain/beamforming_receiver_chain_v2_metrics.csv --learned-summary outputs/sionna_native_chain/learned_beamforming_receiver_summary.json --learned-metrics outputs/sionna_native_chain/learned_beamforming_receiver_metrics.csv --out outputs/sionna_native_chain",
        "project_h_f_assisted": True,
    },
    {
        "path": "outputs/sionna_native_chain/native_learned_minibench/summary.md",
        "description": "Checkpoint-only SNR mini benchmark for native learned beamforming.",
        "command": "python scripts/run_sionna_native_learned_chain_minibench.py --out outputs/sionna_native_chain/native_learned_minibench",
        "project_h_f_assisted": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_json_if_possible(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary_flags(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "uses_real_sionna_resource_grid": False,
            "uses_real_sionna_channel": False,
            "uses_real_sionna_estimator": False,
            "uses_real_sionna_equalizer": False,
            "uses_real_sionna_demapper": False,
        }
    return {
        "uses_real_sionna_resource_grid": bool(
            payload.get("used_sionna_resource_grid", payload.get("used_sionna_native_components", False))
        ),
        "uses_real_sionna_channel": bool(payload.get("used_sionna_channel", False)),
        "uses_real_sionna_estimator": bool(payload.get("used_sionna_estimator", payload.get("used_ls_lmmse", False))),
        "uses_real_sionna_equalizer": bool(payload.get("used_sionna_equalizer", payload.get("used_ls_lmmse", False))),
        "uses_real_sionna_demapper": bool(payload.get("used_sionna_demapper", False)),
    }


def _artifact_row(item: dict[str, Any], commit: str) -> dict[str, Any]:
    path = Path(item["path"])
    payload = _load_json_if_possible(path) if path.exists() else None
    flags = _summary_flags(payload)
    row: dict[str, Any] = {
        "path": item["path"],
        "description": item["description"],
        "generating_command": item["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        **flags,
        "project_h_f_assisted": bool(item["project_h_f_assisted"]),
        "full_native_only": bool(
            flags["uses_real_sionna_resource_grid"]
            and flags["uses_real_sionna_channel"]
            and flags["uses_real_sionna_estimator"]
            and flags["uses_real_sionna_equalizer"]
            and flags["uses_real_sionna_demapper"]
            and not item["project_h_f_assisted"]
        ),
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }
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
        "note": (
            "Sionna-native chain artifacts remain optional and synthetic/project-H_f-assisted where stated. "
            "No Sionna RT, no ray tracing, and no 5G NR full stack are used."
        ),
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna Native Chain Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: optional synthetic Sionna-native chain artifacts only; project-H_f-assisted entries are not full native-only benchmarks.",
        "",
        "| path | exists | resource grid | channel | estimator | equalizer | demapper | project H_f assisted | full native only | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['path']} | {row['exists']} | {row['uses_real_sionna_resource_grid']} | {row['uses_real_sionna_channel']} | "
            f"{row['uses_real_sionna_estimator']} | {row['uses_real_sionna_equalizer']} | {row['uses_real_sionna_demapper']} | "
            f"{row['project_h_f_assisted']} | {row['full_native_only']} | `{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved Sionna native-chain artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
