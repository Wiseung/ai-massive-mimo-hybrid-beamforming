#!/usr/bin/env python
"""Generate a manifest for CSI-interface artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "name": "channel_tensor_audit",
        "path": "outputs/sionna_channel_extraction/channel_tensor_audit.json",
        "description": "Sionna channel tensor API audit for the channel-extraction bridge.",
        "command": "python scripts/audit_sionna_channel_tensor_shapes.py --out outputs/sionna_channel_extraction/channel_tensor_audit.json",
    },
    {
        "name": "extract_h_f_demo_summary",
        "path": "outputs/sionna_channel_extraction/extract_h_f_demo_summary.json",
        "description": "Minimal extracted-H_f demo summary.",
        "command": "python scripts/sionna_extract_channel_hf_demo.py --out outputs/sionna_channel_extraction/extract_h_f_demo_summary.json",
    },
    {
        "name": "hf_axis_validation",
        "path": "outputs/sionna_channel_extraction/hf_axis_validation.json",
        "description": "Axis and OFDM-symbol sanity validation for extracted H_f.",
        "command": "python scripts/validate_sionna_extracted_hf_axes.py --out outputs/sionna_channel_extraction/hf_axis_validation.json",
    },
    {
        "name": "native_channel_beamforming_summary",
        "path": "outputs/sionna_channel_extraction/native_channel_beamforming_summary.json",
        "description": "Native-channel-assisted beamforming summary before CSI interface standardization.",
        "command": "python scripts/sionna_native_channel_beamforming_chain.py --out outputs/sionna_channel_extraction/native_channel_beamforming_summary.json --receiver-mode auto",
    },
    {
        "name": "csi_interface_audit",
        "path": "outputs/sionna_channel_extraction/csi_interface_audit.json",
        "description": "Provenance audit for the standardized ExtractedCSI object.",
        "command": "python scripts/audit_sionna_csi_interface.py --out outputs/sionna_channel_extraction/csi_interface_audit.json",
    },
    {
        "name": "csi_backed_beamforming_summary",
        "path": "outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json",
        "description": "CSI-backed beamforming summary on the native receiver path.",
        "command": "python scripts/sionna_csi_backed_beamforming_chain.py --out outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json --receiver-mode auto --seed 0",
    },
    {
        "name": "csi_interface_comparison",
        "path": "outputs/sionna_channel_extraction/csi_interface_comparison.md",
        "description": "Cross-run raw-vs-CSI comparison summary with corrected semantics.",
        "command": "python scripts/compare_csi_backed_vs_raw_extracted_h.py --raw outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv --csi outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --out outputs/sionna_channel_extraction",
    },
    {
        "name": "csi_same_batch_equivalence",
        "path": "outputs/sionna_channel_extraction/csi_same_batch_equivalence.json",
        "description": "Same-batch raw-vs-CSI equivalence validation summary.",
        "command": "python scripts/validate_csi_same_batch_equivalence.py --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json",
    },
    {
        "name": "csi_raw_mismatch_audit",
        "path": "outputs/sionna_channel_extraction/csi_raw_mismatch_audit.json",
        "description": "Root-cause audit for the earlier raw-vs-CSI mismatch.",
        "command": "python scripts/audit_csi_raw_comparison_mismatch.py --raw outputs/sionna_channel_extraction/native_channel_beamforming_metrics.csv --csi outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --out outputs/sionna_channel_extraction",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.suffix != ".json":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _extract_md_value(text: str | None, label: str) -> str | None:
    if not text:
        return None
    match = re.search(rf"{re.escape(label)}: `([^`]+)`", text)
    if match is None:
        return None
    return match.group(1)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _artifact_row(item: dict[str, Any], commit: str) -> dict[str, Any]:
    path = Path(item["path"])
    payload = _load_json(path)
    text = _load_text(path) if payload is None else None

    row: dict[str, Any] = {
        "name": item["name"],
        "path": item["path"],
        "description": item["description"],
        "generating_command": item["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        "csi_interface_used": None,
        "same_batch_equivalence": None,
        "numeric_consistency_within_tolerance": None,
        "ranking_consistent": None,
        "comparison_type": None,
        "project_h_f_assisted": None,
        "extracted_h_f_used": None,
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }

    if payload is not None:
        row["csi_interface_used"] = _coerce_bool(payload.get("csi_interface_used"))
        row["numeric_consistency_within_tolerance"] = _coerce_bool(payload.get("numeric_consistency_within_tolerance"))
        row["ranking_consistent"] = _coerce_bool(payload.get("ranking_consistent"))
        row["project_h_f_assisted"] = _coerce_bool(payload.get("project_h_f_assisted"))
        row["extracted_h_f_used"] = _coerce_bool(payload.get("extracted_h_f_used"))

    if item["name"] == "channel_tensor_audit" and payload is not None:
        row["csi_interface_used"] = False
        row["project_h_f_assisted"] = True
        row["extracted_h_f_used"] = False
        row["comparison_type"] = "channel_tensor_audit"
    elif item["name"] == "extract_h_f_demo_summary" and payload is not None:
        row["csi_interface_used"] = True
        row["project_h_f_assisted"] = True if payload.get("fallback_used") else False
        row["extracted_h_f_used"] = _coerce_bool(payload.get("extraction_success"))
        row["comparison_type"] = "extraction_demo"
    elif item["name"] == "hf_axis_validation" and payload is not None:
        row["csi_interface_used"] = True
        row["comparison_type"] = "axis_validation"
        row["extracted_h_f_used"] = True
        row["project_h_f_assisted"] = False
    elif item["name"] == "native_channel_beamforming_summary" and payload is not None:
        row["csi_interface_used"] = _coerce_bool(payload.get("csi_interface_used"))
        row["comparison_type"] = "native_channel_assisted_summary"
    elif item["name"] == "csi_interface_audit" and payload is not None:
        row["comparison_type"] = "csi_provenance_audit"
    elif item["name"] == "csi_backed_beamforming_summary" and payload is not None:
        row["comparison_type"] = "single_run_summary"
    elif item["name"] == "csi_interface_comparison":
        row["csi_interface_used"] = True
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = _coerce_bool(
            _extract_md_value(text, "1. numeric consistency within tolerance")
        )
        row["ranking_consistent"] = _coerce_bool(_extract_md_value(text, "2. method ranking consistent"))
        row["comparison_type"] = _extract_md_value(text, "- comparison_type")
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True
    elif item["name"] == "csi_same_batch_equivalence" and payload is not None:
        row["csi_interface_used"] = True
        row["same_batch_equivalence"] = True
        row["numeric_consistency_within_tolerance"] = _coerce_bool(payload.get("numeric_consistency_within_tolerance"))
        row["ranking_consistent"] = _coerce_bool(payload.get("ranking_consistent"))
        row["comparison_type"] = "same_batch_equivalence"
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True
    elif item["name"] == "csi_raw_mismatch_audit" and payload is not None:
        row["csi_interface_used"] = True
        row["same_batch_equivalence"] = False
        row["comparison_type"] = "cross_run_comparison_audit"
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True

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
            "CSI-interface artifacts remain optional and provenance-focused. "
            "Same-batch equivalence is the valid strict comparison; cross-run comparison is not. "
            "No Sionna RT, no ray tracing, and no 5G NR full stack are used."
        ),
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna CSI Interface Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: optional Sionna CSI-interface artifacts only; same-batch equivalence and cross-run comparison are tracked separately.",
        "",
        "| name | exists | csi_interface_used | same_batch_equivalence | numeric_consistency | ranking_consistent | comparison_type | project_h_f_assisted | extracted_h_f_used | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['exists']} | {row['csi_interface_used']} | {row['same_batch_equivalence']} | "
            f"{row['numeric_consistency_within_tolerance']} | {row['ranking_consistent']} | {row['comparison_type']} | "
            f"{row['project_h_f_assisted']} | {row['extracted_h_f_used']} | `{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved Sionna CSI-interface artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
