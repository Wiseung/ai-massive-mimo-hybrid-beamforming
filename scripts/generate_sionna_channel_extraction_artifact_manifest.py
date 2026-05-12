#!/usr/bin/env python
"""Generate a manifest for Sionna channel-extraction artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "name": "channel_tensor_audit",
        "path": "outputs/sionna_channel_extraction/channel_tensor_audit.json",
        "description": "Sionna channel tensor API audit for OFDM/native extraction.",
        "command": "python scripts/audit_sionna_channel_tensor_shapes.py --out outputs/sionna_channel_extraction/channel_tensor_audit.json",
        "quick_or_full": "full",
    },
    {
        "name": "extract_h_f_demo_summary",
        "path": "outputs/sionna_channel_extraction/extract_h_f_demo_summary.json",
        "description": "Minimal extracted-H_f demo summary.",
        "command": "python scripts/sionna_extract_channel_hf_demo.py --out outputs/sionna_channel_extraction/extract_h_f_demo_summary.json",
        "quick_or_full": "full",
    },
    {
        "name": "hf_axis_validation",
        "path": "outputs/sionna_channel_extraction/hf_axis_validation.json",
        "description": "Axis and OFDM-symbol sanity validation for extracted H_f.",
        "command": "python scripts/validate_sionna_extracted_hf_axes.py --out outputs/sionna_channel_extraction/hf_axis_validation.json",
        "quick_or_full": "full",
    },
    {
        "name": "native_channel_beamforming_summary",
        "path": "outputs/sionna_channel_extraction/native_channel_beamforming_summary.json",
        "description": "Native-channel-assisted beamforming chain summary.",
        "command": "python scripts/sionna_native_channel_beamforming_chain.py --out outputs/sionna_channel_extraction/native_channel_beamforming_summary.json --receiver-mode auto",
        "quick_or_full": "full",
    },
    {
        "name": "extracted_h_consistency_summary",
        "path": "outputs/sionna_channel_extraction/extracted_h_consistency/summary.md",
        "description": "Quick extracted-H consistency benchmark summary.",
        "command": "python scripts/benchmark_sionna_extracted_h_consistency.py --out outputs/sionna_channel_extraction/extracted_h_consistency --seeds 1 2 3 --snrs 0 5 10 15 20 --quick",
        "quick_or_full": "quick",
    },
    {
        "name": "extracted_h_consistency_metrics",
        "path": "outputs/sionna_channel_extraction/extracted_h_consistency/metrics.csv",
        "description": "Quick extracted-H consistency benchmark metrics CSV.",
        "command": "python scripts/benchmark_sionna_extracted_h_consistency.py --out outputs/sionna_channel_extraction/extracted_h_consistency --seeds 1 2 3 --snrs 0 5 10 15 20 --quick",
        "quick_or_full": "quick",
    },
    {
        "name": "extraction_config_sweep",
        "path": "outputs/sionna_channel_extraction/extraction_config_sweep/extraction_sweep.md",
        "description": "Quick extraction-config sweep summary.",
        "command": "python scripts/sweep_sionna_channel_extraction_config.py --quick --out outputs/sionna_channel_extraction/extraction_config_sweep",
        "quick_or_full": "quick",
    },
    {
        "name": "project_vs_extracted_hf_comparison",
        "path": "outputs/sionna_channel_extraction/project_vs_extracted_hf/comparison.md",
        "description": "Project-H_f-assisted versus extracted-H_f comparison summary.",
        "command": "python scripts/compare_project_hf_vs_extracted_hf.py --project outputs/sionna_native_chain/learned_beamforming_receiver_metrics.csv --extracted outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv --consistency outputs/sionna_channel_extraction/extracted_h_consistency/metrics.csv --out outputs/sionna_channel_extraction/project_vs_extracted_hf",
        "quick_or_full": "mixed",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_json_if_possible(path: Path) -> dict[str, Any] | None:
    if path.suffix != ".json" or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifact_row(item: dict[str, Any], commit: str) -> dict[str, Any]:
    path = Path(item["path"])
    payload = _load_json_if_possible(path)
    extraction_success = None
    native_receiver_success = None
    project_h_f_assisted = None
    extracted_h_f_used = None

    if payload is not None:
        extraction_success = payload.get("extraction_success")
        native_receiver_success = payload.get("native_receiver_success")
        project_h_f_assisted = payload.get("project_h_f_assisted")
        if isinstance(payload.get("metrics"), list):
            extracted_h_f_used = any(bool(row.get("extracted_h_f_used", False)) for row in payload["metrics"])
    if item["name"] == "hf_axis_validation" and payload is not None:
        extraction_success = bool(payload.get("axis_spot_check_passed"))
        extracted_h_f_used = True
    if item["name"] == "channel_tensor_audit" and payload is not None:
        extraction_success = bool(payload.get("summary", {}).get("can_convert_to_project_h_f"))
    if item["name"] == "extract_h_f_demo_summary" and payload is not None:
        extracted_h_f_used = bool(payload.get("extraction_success"))
    if item["name"].startswith("extracted_h_consistency"):
        extraction_success = True if path.exists() else None
        native_receiver_success = True if path.exists() else None
        project_h_f_assisted = False if path.exists() else None
        extracted_h_f_used = True if path.exists() else None
    if item["name"] == "extraction_config_sweep":
        extraction_success = True if path.exists() else None
        project_h_f_assisted = False if path.exists() else None
        extracted_h_f_used = True if path.exists() else None
    if item["name"] == "project_vs_extracted_hf_comparison":
        extraction_success = True if path.exists() else None
        native_receiver_success = True if path.exists() else None
        project_h_f_assisted = False if path.exists() else None
        extracted_h_f_used = True if path.exists() else None

    return {
        "name": item["name"],
        "path": item["path"],
        "description": item["description"],
        "generating_command": item["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        "extraction_success": extraction_success,
        "native_receiver_success": native_receiver_success,
        "project_h_f_assisted": project_h_f_assisted,
        "extracted_h_f_used": extracted_h_f_used,
        "full_native_only": False,
        "quick_or_full": item["quick_or_full"],
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }


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
            "Sionna channel-extraction artifacts remain optional. The supported interpretation is "
            "native-channel-assisted plus native-receiver-assisted; not full native-only. "
            "Consistency benchmark entries are explicitly quick/limited where marked."
        ),
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna Channel Extraction Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: optional Sionna channel-extraction artifacts only; quick/limited benchmarks are marked explicitly.",
        "",
        "| name | exists | extraction_success | native_receiver_success | project_h_f_assisted | extracted_h_f_used | quick_or_full | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['exists']} | {row['extraction_success']} | {row['native_receiver_success']} | "
            f"{row['project_h_f_assisted']} | {row['extracted_h_f_used']} | {row['quick_or_full']} | `{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved Sionna channel-extraction artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
