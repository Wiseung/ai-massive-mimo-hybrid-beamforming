#!/usr/bin/env python
"""Audit raw-H and CSI consumer coverage across the project."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _consumer_rows() -> list[dict[str, object]]:
    return [
        {
            "file": "src/beamforming/utils/sionna_native_beamforming_chain.py",
            "function_or_script": "compute_project_precoder_per_subcarrier",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "high",
            "recommended_change": "Use as_project_h_f() internally so analytic precoders accept ExtractedCSI and raw H_f.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "src/beamforming/utils/sionna_native_learned_beamforming.py",
            "function_or_script": "infer_learned_precoder",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "high",
            "recommended_change": "Normalize inputs through as_project_h_f() and record CSI provenance in inference metadata.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/sionna_csi_backed_beamforming_chain.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "high",
            "recommended_change": "Pass ExtractedCSI directly into analytic and learned consumers; keep summary fields for provenance.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/sionna_native_channel_beamforming_chain.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "high",
            "recommended_change": "Prefer context.csi when available and only keep raw H_f as fallback for legacy path.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/sionna_native_ofdm_learned_beamforming_chain.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "CSI-backed path is wired in; keep summary provenance fields and raw fallback for backward compatibility.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/validate_csi_same_batch_equivalence.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "Use one shared ExtractedCSI object and compare raw fallback vs CSI-backed consumers under one realization.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/audit_sionna_csi_interface.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "Audit consumer calls using ExtractedCSI directly and summarize provenance completeness.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/benchmark_sionna_extracted_h_consistency.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "Benchmark now prefers context.csi when available; preserve quick/limited wording and raw fallback.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/sionna_extract_channel_hf_demo.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "Prefer CSI summary in artifacts instead of direct tensor-only reporting.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/run_sionna_native_learned_chain_minibench.py",
            "function_or_script": "main",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "low",
            "recommended_change": "Keep optional minibench aligned with ExtractedCSI-first input summary while preserving raw fallback.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/sionna_native_ofdm_beamforming_chain.py",
            "function_or_script": "main",
            "current_input_type": "raw_h_f",
            "should_support_csi": False,
            "migration_priority": "low",
            "recommended_change": "Legacy v0.4-era project-assisted script; optional future cleanup if this path is revived.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "tests/test_csi_interface.py",
            "function_or_script": "CSI base tests",
            "current_input_type": "ExtractedCSI",
            "should_support_csi": True,
            "migration_priority": "low",
            "recommended_change": "Keep schema tests and extend adapters in dedicated adapter test file.",
            "backwards_compatibility_required": False,
        },
        {
            "file": "tests/test_sionna_channel_extraction_optional.py",
            "function_or_script": "optional Sionna script tests",
            "current_input_type": "unknown",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "Cover audit/demo/comparison scripts that should now surface CSI-backed behavior.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "README.md",
            "function_or_script": "command examples and branch status",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "Document ExtractedCSI-first path and raw H_f fallback boundaries.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "docs/sionna_native_channel_extraction.md",
            "function_or_script": "CSI interface documentation",
            "current_input_type": "both",
            "should_support_csi": True,
            "migration_priority": "medium",
            "recommended_change": "Add consumer-unification status table and clarify remaining raw fallback consumers.",
            "backwards_compatibility_required": True,
        },
    ]


def _md(payload: dict[str, object]) -> list[str]:
    lines = [
        "# CSI Consumer Audit",
        "",
        f"- total_consumers_audited: `{payload['total_consumers_audited']}`",
        f"- raw_only_high_priority_paths: `{payload['raw_only_high_priority_paths']}`",
        f"- already_support_both: `{payload['already_support_both']}`",
        f"- csi_value_weakened_by_unified_gaps: `{payload['csi_value_weakened_by_unified_gaps']}`",
        "",
        "## Summary",
        f"1. raw-only key paths: `{', '.join(payload['raw_only_key_paths']) if payload['raw_only_key_paths'] else 'none'}`",
        f"2. already support ExtractedCSI: `{', '.join(payload['already_support_csi_paths']) if payload['already_support_csi_paths'] else 'none'}`",
        f"3. priority migration targets: `{', '.join(payload['priority_migration_targets']) if payload['priority_migration_targets'] else 'none'}`",
        f"4. ununified consumers still weaken v0.6.0 CSI value: `{payload['csi_value_weakened_by_unified_gaps']}`",
        "",
        "| File | Function/Script | Input | Should support CSI | Priority | Recommended change |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["audit"]:
        lines.append(
            f"| {row['file']} | {row['function_or_script']} | {row['current_input_type']} | "
            f"{row['should_support_csi']} | {row['migration_priority']} | {row['recommended_change']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    audit = _consumer_rows()

    raw_only_key_paths = [
        f"{row['file']}::{row['function_or_script']}"
        for row in audit
        if row["current_input_type"] == "raw_h_f" and row["migration_priority"] in {"high", "medium"}
    ]
    already_support_csi_paths = [
        f"{row['file']}::{row['function_or_script']}"
        for row in audit
        if row["current_input_type"] in {"ExtractedCSI", "both"}
    ]
    priority_targets = [
        f"{row['file']}::{row['function_or_script']}"
        for row in audit
        if row["migration_priority"] == "high"
    ]
    payload = {
        "audit": audit,
        "total_consumers_audited": len(audit),
        "raw_only_high_priority_paths": len(raw_only_key_paths),
        "already_support_both": sum(1 for row in audit if row["current_input_type"] == "both"),
        "raw_only_key_paths": raw_only_key_paths,
        "already_support_csi_paths": already_support_csi_paths,
        "priority_migration_targets": priority_targets,
        "csi_value_weakened_by_unified_gaps": bool(raw_only_key_paths),
    }
    write_json(out_path, payload)
    write_markdown(md_path, _md(payload))
    print(f"Saved CSI consumer audit to {out_path}")


if __name__ == "__main__":
    main()
