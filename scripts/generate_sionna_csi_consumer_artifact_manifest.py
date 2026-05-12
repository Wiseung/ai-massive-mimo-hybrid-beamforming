#!/usr/bin/env python
"""Generate a manifest for CSI consumer-unification artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "name": "csi_consumer_audit",
        "path": "outputs/sionna_channel_extraction/csi_consumer_audit.json",
        "description": "Audit of raw-H vs ExtractedCSI consumer coverage across analytic, learned, native-chain, comparison, and docs paths.",
        "command": "python scripts/audit_csi_consumers.py --out outputs/sionna_channel_extraction/csi_consumer_audit.json",
    },
    {
        "name": "unified_csi_consumers_summary",
        "path": "outputs/sionna_channel_extraction/unified_csi_consumers_summary.json",
        "description": "Unified CSI consumer demo summary for one shared ExtractedCSI object.",
        "command": "python scripts/demo_unified_csi_consumers.py --out outputs/sionna_channel_extraction/unified_csi_consumers_summary.json",
    },
    {
        "name": "unified_csi_consumers_metrics",
        "path": "outputs/sionna_channel_extraction/unified_csi_consumers_metrics.csv",
        "description": "Per-method metrics produced by the unified CSI consumer demo.",
        "command": "python scripts/demo_unified_csi_consumers.py --out outputs/sionna_channel_extraction/unified_csi_consumers_summary.json",
    },
    {
        "name": "unified_csi_consumer_comparison",
        "path": "outputs/sionna_channel_extraction/unified_csi_consumer_comparison.md",
        "description": "Cross-run comparison between the existing CSI-backed path and the unified CSI consumer demo.",
        "command": "python scripts/compare_unified_csi_consumers.py --baseline outputs/sionna_channel_extraction/csi_backed_beamforming_metrics.csv --unified outputs/sionna_channel_extraction/unified_csi_consumers_metrics.csv --out outputs/sionna_channel_extraction",
    },
    {
        "name": "csi_backed_beamforming_summary",
        "path": "outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json",
        "description": "Updated CSI-backed native receiver summary used as the baseline for consumer unification.",
        "command": "python scripts/sionna_csi_backed_beamforming_chain.py --out outputs/sionna_channel_extraction/csi_backed_beamforming_summary.json --receiver-mode auto --seed 0",
    },
    {
        "name": "csi_interface_audit",
        "path": "outputs/sionna_channel_extraction/csi_interface_audit.json",
        "description": "Relevant v0.6.0 provenance audit for the ExtractedCSI schema.",
        "command": "python scripts/audit_sionna_csi_interface.py --out outputs/sionna_channel_extraction/csi_interface_audit.json",
    },
    {
        "name": "csi_same_batch_equivalence",
        "path": "outputs/sionna_channel_extraction/csi_same_batch_equivalence.json",
        "description": "Relevant v0.6.0 same-batch equivalence result for strict shared-realization validation.",
        "command": "python scripts/validate_csi_same_batch_equivalence.py --out outputs/sionna_channel_extraction/csi_same_batch_equivalence.json",
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


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _extract_line_value(text: str | None, prefix: str) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        if prefix in line:
            marker = line.split("`")
            if len(marker) >= 2:
                return marker[1]
    return None


def _artifact_row(spec: dict[str, str], commit: str) -> dict[str, Any]:
    path = Path(spec["path"])
    payload = _load_json(path)
    text = _load_text(path) if payload is None else None

    row: dict[str, Any] = {
        "name": spec["name"],
        "path": spec["path"],
        "description": spec["description"],
        "generating_command": spec["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        "csi_interface_used": None,
        "same_csi_object_used_for_all_methods": None,
        "all_consumers_accept_csi": None,
        "no_new_fallback_introduced": None,
        "comparison_type": None,
        "strict_equivalence_claim_allowed": None,
        "project_h_f_assisted": None,
        "extracted_h_f_used": None,
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }

    if spec["name"] == "csi_consumer_audit" and payload is not None:
        row["csi_interface_used"] = True
        row["same_csi_object_used_for_all_methods"] = None
        row["all_consumers_accept_csi"] = payload.get("raw_only_high_priority_paths", 1) == 0
        row["no_new_fallback_introduced"] = True
        row["comparison_type"] = "consumer_audit"
        row["strict_equivalence_claim_allowed"] = False
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True
    elif spec["name"] == "unified_csi_consumers_summary" and payload is not None:
        row["csi_interface_used"] = _bool_or_none(payload.get("csi_interface_used"))
        row["same_csi_object_used_for_all_methods"] = _bool_or_none(payload.get("same_csi_object_used_for_all_methods"))
        row["all_consumers_accept_csi"] = _bool_or_none(payload.get("all_consumers_accept_csi"))
        row["no_new_fallback_introduced"] = _bool_or_none(payload.get("no_new_fallback_introduced"))
        row["comparison_type"] = "single_run_unified_demo"
        row["strict_equivalence_claim_allowed"] = False
        row["project_h_f_assisted"] = _bool_or_none(payload.get("project_h_f_assisted"))
        row["extracted_h_f_used"] = _bool_or_none(payload.get("extracted_h_f_used"))
    elif spec["name"] == "unified_csi_consumers_metrics":
        summary = _load_json(path.with_name("unified_csi_consumers_summary.json"))
        row["csi_interface_used"] = _bool_or_none(summary.get("csi_interface_used")) if summary else None
        row["same_csi_object_used_for_all_methods"] = _bool_or_none(summary.get("same_csi_object_used_for_all_methods")) if summary else None
        row["all_consumers_accept_csi"] = _bool_or_none(summary.get("all_consumers_accept_csi")) if summary else None
        row["no_new_fallback_introduced"] = _bool_or_none(summary.get("no_new_fallback_introduced")) if summary else None
        row["comparison_type"] = "single_run_unified_demo_metrics"
        row["strict_equivalence_claim_allowed"] = False
        row["project_h_f_assisted"] = _bool_or_none(summary.get("project_h_f_assisted")) if summary else None
        row["extracted_h_f_used"] = _bool_or_none(summary.get("extracted_h_f_used")) if summary else None
    elif spec["name"] == "unified_csi_consumer_comparison":
        row["csi_interface_used"] = True
        row["same_csi_object_used_for_all_methods"] = False
        row["all_consumers_accept_csi"] = _bool_or_none(_extract_line_value(text, "2. all key consumers accept ExtractedCSI") == "True")
        row["no_new_fallback_introduced"] = _bool_or_none(_extract_line_value(text, "3. additional fallback introduced") == "False")
        row["comparison_type"] = _extract_line_value(text, "- comparison_type")
        row["strict_equivalence_claim_allowed"] = _bool_or_none(_extract_line_value(text, "- strict_equivalence_claim_allowed") == "True")
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True
    elif spec["name"] == "csi_backed_beamforming_summary" and payload is not None:
        row["csi_interface_used"] = _bool_or_none(payload.get("csi_interface_used"))
        row["same_csi_object_used_for_all_methods"] = False
        row["all_consumers_accept_csi"] = True
        row["no_new_fallback_introduced"] = True
        row["comparison_type"] = "single_run_baseline"
        row["strict_equivalence_claim_allowed"] = False
        row["project_h_f_assisted"] = _bool_or_none(payload.get("project_h_f_assisted"))
        row["extracted_h_f_used"] = _bool_or_none(payload.get("extracted_h_f_used"))
    elif spec["name"] == "csi_interface_audit" and payload is not None:
        row["csi_interface_used"] = _bool_or_none(payload.get("csi_interface_used"))
        row["same_csi_object_used_for_all_methods"] = False
        row["all_consumers_accept_csi"] = None
        row["no_new_fallback_introduced"] = True
        row["comparison_type"] = "v0_6_csi_audit"
        row["strict_equivalence_claim_allowed"] = False
        row["project_h_f_assisted"] = _bool_or_none(payload.get("project_h_f_assisted"))
        row["extracted_h_f_used"] = _bool_or_none(payload.get("extracted_h_f_used"))
    elif spec["name"] == "csi_same_batch_equivalence" and payload is not None:
        row["csi_interface_used"] = True
        row["same_csi_object_used_for_all_methods"] = None
        row["all_consumers_accept_csi"] = None
        row["no_new_fallback_introduced"] = True
        row["comparison_type"] = "same_batch_equivalence"
        row["strict_equivalence_claim_allowed"] = True
        row["project_h_f_assisted"] = False
        row["extracted_h_f_used"] = True

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
            "CSI consumer-unification artifacts keep Sionna optional and preserve the boundary that "
            "unified-vs-baseline is a cross-run comparison, not a strict equivalence claim. "
            "No Sionna RT, no ray tracing, and no 5G NR full stack are used."
        ),
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna CSI Consumer Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: consumer unification keeps ExtractedCSI preferred, raw H_f as fallback, and cross-run comparison semantics conservative.",
        "",
        "| name | exists | csi_interface_used | same_csi_object_used_for_all_methods | all_consumers_accept_csi | no_new_fallback_introduced | comparison_type | strict_equivalence_claim_allowed | project_h_f_assisted | extracted_h_f_used | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['exists']} | {row['csi_interface_used']} | {row['same_csi_object_used_for_all_methods']} | "
            f"{row['all_consumers_accept_csi']} | {row['no_new_fallback_introduced']} | {row['comparison_type']} | "
            f"{row['strict_equivalence_claim_allowed']} | {row['project_h_f_assisted']} | {row['extracted_h_f_used']} | "
            f"`{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved Sionna CSI consumer artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
