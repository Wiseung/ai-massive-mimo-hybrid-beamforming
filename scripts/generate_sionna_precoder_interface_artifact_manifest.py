#!/usr/bin/env python
"""Generate a manifest for PrecoderOutput bridge artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "name": "precoder_interface_audit",
        "path": "outputs/sionna_channel_extraction/precoder_interface_audit.json",
        "description": "Audit of raw-F_f vs PrecoderOutput producer/consumer coverage across analytic, learned, native-chain, comparison, and docs paths.",
        "command": "python scripts/audit_precoder_interface_consumers.py --out outputs/sionna_channel_extraction/precoder_interface_audit.json",
    },
    {
        "name": "unified_csi_precoder_summary",
        "path": "outputs/sionna_channel_extraction/unified_csi_precoder_summary.json",
        "description": "Unified ExtractedCSI + PrecoderOutput demo summary for one shared CSI object and native receiver path.",
        "command": "python scripts/demo_unified_csi_and_precoder_interfaces.py --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json",
    },
    {
        "name": "unified_csi_precoder_metrics",
        "path": "outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv",
        "description": "Per-method metrics produced by the unified ExtractedCSI + PrecoderOutput demo.",
        "command": "python scripts/demo_unified_csi_and_precoder_interfaces.py --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json",
    },
    {
        "name": "precoder_output_comparison",
        "path": "outputs/sionna_channel_extraction/precoder_output_comparison.md",
        "description": "Cross-run raw-F_f vs PrecoderOutput comparison with conservative semantics.",
        "command": "python scripts/compare_raw_ff_vs_precoder_output.py --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv --out outputs/sionna_channel_extraction",
    },
    {
        "name": "precoder_output_same_batch_equivalence",
        "path": "outputs/sionna_channel_extraction/precoder_output_same_batch_equivalence.json",
        "description": "Same-batch raw-F_f vs PrecoderOutput equivalence validation under one shared CSI/F_f realization.",
        "command": "python scripts/validate_precoder_output_same_batch_equivalence.py --out outputs/sionna_channel_extraction/precoder_output_same_batch_equivalence.json",
    },
    {
        "name": "precoder_output_mismatch_audit",
        "path": "outputs/sionna_channel_extraction/precoder_output_mismatch_audit.json",
        "description": "Root-cause audit for the earlier raw-F_f vs PrecoderOutput ranking mismatch.",
        "command": "python scripts/audit_precoder_output_comparison_mismatch.py --raw outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --precoder-output outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv --out outputs/sionna_channel_extraction",
    },
    {
        "name": "csi_interface_audit",
        "path": "outputs/sionna_channel_extraction/csi_interface_audit.json",
        "description": "Relevant v0.6.0 ExtractedCSI provenance audit for the input side of the bridge.",
        "command": "python scripts/audit_sionna_csi_interface.py --out outputs/sionna_channel_extraction/csi_interface_audit.json",
    },
    {
        "name": "csi_same_batch_equivalence",
        "path": "outputs/sionna_channel_extraction/csi_same_batch_equivalence.json",
        "description": "Relevant v0.6.0 same-batch ExtractedCSI equivalence validation for shared-realization reference.",
        "command": "python scripts/validate_csi_same_batch_equivalence.py --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json",
    },
    {
        "name": "csi_consumer_audit",
        "path": "outputs/sionna_channel_extraction/csi_consumer_audit.json",
        "description": "Relevant v0.7.0 CSI consumer audit showing high-priority input-side gaps are closed.",
        "command": "python scripts/audit_csi_consumers.py --out outputs/sionna_channel_extraction/csi_consumer_audit.json",
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


def _summary_from_csv(path: Path) -> dict[str, Any]:
    if not path.exists() or path.suffix != ".csv":
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}
    return {"row_count": len(rows)}


def _artifact_row(spec: dict[str, str], commit: str) -> dict[str, Any]:
    path = Path(spec["path"])
    payload = _load_json(path)
    text = _load_text(path) if payload is None else None
    unified_summary = _load_json(Path("outputs/sionna_channel_extraction/unified_csi_precoder_summary.json")) or {}
    same_batch_summary = _load_json(Path("outputs/sionna_channel_extraction/precoder_output_same_batch_equivalence.json")) or {}
    mismatch_summary = _load_json(Path("outputs/sionna_channel_extraction/precoder_output_mismatch_audit.json")) or {}

    row: dict[str, Any] = {
        "name": spec["name"],
        "path": spec["path"],
        "description": spec["description"],
        "generating_command": spec["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        "csi_interface_used": None,
        "precoder_interface_used": None,
        "same_csi_object_used_for_all_methods": None,
        "all_precoders_emit_precoder_output": None,
        "all_receiver_consumers_accept_precoder_output": None,
        "same_batch_equivalence": None,
        "numeric_consistency_within_tolerance": None,
        "strict_equivalence_claim_allowed": None,
        "comparison_type": None,
        "project_side_precoder": True,
        "sionna_native_precoder": False,
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }

    if spec["name"] == "precoder_interface_audit" and payload is not None:
        row["csi_interface_used"] = True
        row["precoder_interface_used"] = True
        row["same_csi_object_used_for_all_methods"] = None
        row["all_precoders_emit_precoder_output"] = payload.get("raw_only_high_priority_paths", 1) == 0
        row["all_receiver_consumers_accept_precoder_output"] = payload.get("raw_only_high_priority_paths", 1) == 0
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = None
        row["strict_equivalence_claim_allowed"] = False
        row["comparison_type"] = "precoder_audit"
    elif spec["name"] == "unified_csi_precoder_summary" and payload is not None:
        row["csi_interface_used"] = _coerce_bool(payload.get("csi_interface_used"))
        row["precoder_interface_used"] = _coerce_bool(payload.get("precoder_interface_used"))
        row["same_csi_object_used_for_all_methods"] = _coerce_bool(payload.get("same_csi_object_used_for_all_methods"))
        row["all_precoders_emit_precoder_output"] = _coerce_bool(payload.get("all_precoders_emit_precoder_output"))
        row["all_receiver_consumers_accept_precoder_output"] = _coerce_bool(payload.get("all_receiver_consumers_accept_precoder_output"))
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = None
        row["strict_equivalence_claim_allowed"] = False
        row["comparison_type"] = "single_run_unified_demo"
    elif spec["name"] == "unified_csi_precoder_metrics":
        row["csi_interface_used"] = _coerce_bool(unified_summary.get("csi_interface_used"))
        row["precoder_interface_used"] = _coerce_bool(unified_summary.get("precoder_interface_used"))
        row["same_csi_object_used_for_all_methods"] = _coerce_bool(unified_summary.get("same_csi_object_used_for_all_methods"))
        row["all_precoders_emit_precoder_output"] = _coerce_bool(unified_summary.get("all_precoders_emit_precoder_output"))
        row["all_receiver_consumers_accept_precoder_output"] = _coerce_bool(unified_summary.get("all_receiver_consumers_accept_precoder_output"))
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = None
        row["strict_equivalence_claim_allowed"] = False
        row["comparison_type"] = "single_run_unified_demo_metrics"
        row.update(_summary_from_csv(path))
    elif spec["name"] == "precoder_output_comparison":
        row["csi_interface_used"] = True
        row["precoder_interface_used"] = True
        row["same_csi_object_used_for_all_methods"] = False
        row["all_precoders_emit_precoder_output"] = True
        row["all_receiver_consumers_accept_precoder_output"] = True
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = False
        row["strict_equivalence_claim_allowed"] = _coerce_bool(_extract_md_value(text, "- strict_equivalence_claim_allowed"))
        row["comparison_type"] = _extract_md_value(text, "- comparison_type")
    elif spec["name"] == "precoder_output_same_batch_equivalence" and payload is not None:
        row["csi_interface_used"] = True
        row["precoder_interface_used"] = True
        row["same_csi_object_used_for_all_methods"] = True
        row["all_precoders_emit_precoder_output"] = True
        row["all_receiver_consumers_accept_precoder_output"] = True
        row["same_batch_equivalence"] = True
        row["numeric_consistency_within_tolerance"] = _coerce_bool(payload.get("numeric_consistency_within_tolerance"))
        row["strict_equivalence_claim_allowed"] = _coerce_bool(payload.get("strict_equivalence_claim_allowed"))
        row["comparison_type"] = "same_batch_equivalence"
    elif spec["name"] == "precoder_output_mismatch_audit" and payload is not None:
        row["csi_interface_used"] = True
        row["precoder_interface_used"] = True
        row["same_csi_object_used_for_all_methods"] = False
        row["all_precoders_emit_precoder_output"] = True
        row["all_receiver_consumers_accept_precoder_output"] = True
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = False
        row["strict_equivalence_claim_allowed"] = False
        row["comparison_type"] = payload.get("comparison_type")
    elif spec["name"] == "csi_interface_audit" and payload is not None:
        row["csi_interface_used"] = True
        row["precoder_interface_used"] = False
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = None
        row["strict_equivalence_claim_allowed"] = False
        row["comparison_type"] = "v0_6_csi_audit"
    elif spec["name"] == "csi_same_batch_equivalence" and payload is not None:
        row["csi_interface_used"] = True
        row["precoder_interface_used"] = False
        row["same_batch_equivalence"] = True
        row["numeric_consistency_within_tolerance"] = _coerce_bool(payload.get("numeric_consistency_within_tolerance"))
        row["strict_equivalence_claim_allowed"] = True
        row["comparison_type"] = "v0_6_same_batch_equivalence"
    elif spec["name"] == "csi_consumer_audit" and payload is not None:
        row["csi_interface_used"] = True
        row["precoder_interface_used"] = False
        row["same_batch_equivalence"] = False
        row["numeric_consistency_within_tolerance"] = None
        row["strict_equivalence_claim_allowed"] = False
        row["comparison_type"] = "v0_7_csi_consumer_audit"

    return row


def main() -> None:
    args = parse_args()
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_json.with_suffix(".md")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = [_artifact_row(spec, commit) for spec in ARTIFACTS]

    payload = {
        "generated_from_commit": commit,
        "note": (
            "PrecoderOutput bridge artifacts remain optional and interface-focused. "
            "Same-batch equivalence is the valid strict comparison; cross-run comparison is not. "
            "No Sionna RT, no ray tracing, and no 5G NR full stack are used."
        ),
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna Precoder Interface Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: optional Sionna PrecoderOutput artifacts only; same-batch equivalence and cross-run comparison are tracked separately.",
        "",
        "| name | exists | csi_interface_used | precoder_interface_used | same_csi_object_used_for_all_methods | all_precoders_emit_precoder_output | all_receiver_consumers_accept_precoder_output | same_batch_equivalence | numeric_consistency_within_tolerance | strict_equivalence_claim_allowed | comparison_type | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['exists']} | {row['csi_interface_used']} | {row['precoder_interface_used']} | "
            f"{row['same_csi_object_used_for_all_methods']} | {row['all_precoders_emit_precoder_output']} | "
            f"{row['all_receiver_consumers_accept_precoder_output']} | {row['same_batch_equivalence']} | "
            f"{row['numeric_consistency_within_tolerance']} | {row['strict_equivalence_claim_allowed']} | "
            f"{row['comparison_type']} | `{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved Sionna PrecoderOutput artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
