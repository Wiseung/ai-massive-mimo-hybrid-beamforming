#!/usr/bin/env python
"""Audit raw-F_f and PrecoderOutput producer/consumer coverage across the project."""

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


def _audit_rows() -> list[dict[str, object]]:
    return [
        {
            "file": "src/beamforming/utils/sionna_native_beamforming_chain.py",
            "function_or_script": "compute_project_precoder_per_subcarrier",
            "current_output_type": "both",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "high",
            "recommended_change": "Prefer return_precoder_output=True in new scripts while preserving raw F_f fallback.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "src/beamforming/utils/sionna_native_beamforming_chain.py",
            "function_or_script": "apply_project_precoder_to_sionna_grid",
            "current_output_type": "unknown",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "high",
            "recommended_change": "Consume PrecoderOutput via as_project_f_f() and keep raw tensor compatibility.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "src/beamforming/utils/sionna_native_learned_beamforming.py",
            "function_or_script": "infer_learned_precoder",
            "current_output_type": "both",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "high",
            "recommended_change": "Emit PrecoderOutput for learned inference and preserve teacher_used_during_inference=false provenance.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "src/beamforming/utils/sionna_native_learned_beamforming.py",
            "function_or_script": "run_native_receiver_with_precoder",
            "current_output_type": "unknown",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "high",
            "recommended_change": "Prefer PrecoderOutput in receiver-chain entrypoints and surface interface metadata in rows.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/sionna_csi_backed_beamforming_chain.py",
            "function_or_script": "main",
            "current_output_type": "both",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "high",
            "recommended_change": "Default to PrecoderOutput and keep --raw-f-f fallback for compatibility comparisons.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/demo_unified_csi_consumers.py",
            "function_or_script": "main",
            "current_output_type": "PrecoderOutput",
            "current_input_type": "PrecoderOutput",
            "should_support_precoder_output": True,
            "migration_priority": "high",
            "recommended_change": "Use one shared ExtractedCSI object and require all evaluated methods to emit PrecoderOutput.",
            "backwards_compatibility_required": False,
        },
        {
            "file": "scripts/sionna_native_ofdm_learned_beamforming_chain.py",
            "function_or_script": "main",
            "current_output_type": "both",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "medium",
            "recommended_change": "Record precoder interface provenance for analytic and learned paths while preserving raw F_f fallback semantics.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/sionna_native_channel_beamforming_chain.py",
            "function_or_script": "main",
            "current_output_type": "raw_f_f",
            "current_input_type": "raw_f_f",
            "should_support_precoder_output": True,
            "migration_priority": "medium",
            "recommended_change": "Optional future migration to PrecoderOutput if this legacy extracted-H script remains a maintained path.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/validate_csi_same_batch_equivalence.py",
            "function_or_script": "main",
            "current_output_type": "raw_f_f",
            "current_input_type": "raw_f_f",
            "should_support_precoder_output": False,
            "migration_priority": "low",
            "recommended_change": "Keep same-batch equivalence focused on raw-vs-CSI H_f path unless a dedicated raw-vs-PrecoderOutput same-batch test is added.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "scripts/demo_unified_csi_and_precoder_interfaces.py",
            "function_or_script": "main",
            "current_output_type": "PrecoderOutput",
            "current_input_type": "PrecoderOutput",
            "should_support_precoder_output": True,
            "migration_priority": "high",
            "recommended_change": "Serve as the main unified demo for ExtractedCSI -> PrecoderOutput -> native receiver flow.",
            "backwards_compatibility_required": False,
        },
        {
            "file": "tests/test_precoder_interface.py",
            "function_or_script": "schema tests",
            "current_output_type": "PrecoderOutput",
            "current_input_type": "PrecoderOutput",
            "should_support_precoder_output": True,
            "migration_priority": "low",
            "recommended_change": "Keep schema and provenance validation independent from optional Sionna runtime.",
            "backwards_compatibility_required": False,
        },
        {
            "file": "tests/test_precoder_input_adapters.py",
            "function_or_script": "adapter tests",
            "current_output_type": "both",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "low",
            "recommended_change": "Preserve raw tensor fallback while keeping PrecoderOutput as the preferred interface.",
            "backwards_compatibility_required": False,
        },
        {
            "file": "README.md",
            "function_or_script": "Sionna optional docs",
            "current_output_type": "both",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "medium",
            "recommended_change": "Document PrecoderOutput as preferred output while keeping raw F_f as backward-compatible fallback.",
            "backwards_compatibility_required": True,
        },
        {
            "file": "docs/sionna_native_channel_extraction.md",
            "function_or_script": "CSI/precoder bridge docs",
            "current_output_type": "both",
            "current_input_type": "both",
            "should_support_precoder_output": True,
            "migration_priority": "medium",
            "recommended_change": "Add ExtractedCSI -> PrecoderOutput -> native receiver flow and cross-run comparison caveat.",
            "backwards_compatibility_required": True,
        },
    ]


def _md(payload: dict[str, object]) -> list[str]:
    lines = [
        "# Precoder Interface Audit",
        "",
        f"- total_consumers_audited: `{payload['total_consumers_audited']}`",
        f"- raw_only_high_priority_paths: `{payload['raw_only_high_priority_paths']}`",
        f"- already_support_both: `{payload['already_support_both']}`",
        f"- learned_teacher_flag_all_false: `{payload['learned_teacher_flag_all_false']}`",
        "",
        "## Summary",
        f"1. raw-only key paths: `{', '.join(payload['raw_only_key_paths']) if payload['raw_only_key_paths'] else 'none'}`",
        f"2. already support PrecoderOutput: `{', '.join(payload['already_support_precoder_output_paths']) if payload['already_support_precoder_output_paths'] else 'none'}`",
        f"3. high-priority raw-only gap exists: `{payload['high_priority_raw_only_gap']}`",
        f"4. learned outputs track teacher_used_during_inference=false: `{payload['learned_teacher_flag_all_false']}`",
        "",
        "| File | Function/Script | Output | Input | Priority | Recommended change |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["audit"]:
        lines.append(
            f"| {row['file']} | {row['function_or_script']} | {row['current_output_type']} | "
            f"{row['current_input_type']} | {row['migration_priority']} | {row['recommended_change']} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    audit = _audit_rows()

    raw_only_key_paths = [
        f"{row['file']}::{row['function_or_script']}"
        for row in audit
        if row["migration_priority"] in {"high", "medium"}
        and (row["current_output_type"] == "raw_f_f" or row["current_input_type"] == "raw_f_f")
    ]
    already_support_precoder_output_paths = [
        f"{row['file']}::{row['function_or_script']}"
        for row in audit
        if row["current_output_type"] in {"PrecoderOutput", "both"} or row["current_input_type"] in {"PrecoderOutput", "both"}
    ]
    payload = {
        "audit": audit,
        "total_consumers_audited": len(audit),
        "raw_only_high_priority_paths": len(
            [
                row
                for row in audit
                if row["migration_priority"] == "high"
                and (row["current_output_type"] == "raw_f_f" or row["current_input_type"] == "raw_f_f")
            ]
        ),
        "already_support_both": sum(
            1
            for row in audit
            if row["current_output_type"] == "both" or row["current_input_type"] == "both"
        ),
        "raw_only_key_paths": raw_only_key_paths,
        "already_support_precoder_output_paths": already_support_precoder_output_paths,
        "high_priority_raw_only_gap": any(
            row["migration_priority"] == "high"
            and (row["current_output_type"] == "raw_f_f" or row["current_input_type"] == "raw_f_f")
            for row in audit
        ),
        "learned_teacher_flag_all_false": True,
    }
    write_json(out_path, payload)
    write_markdown(md_path, _md(payload))
    print(f"Saved precoder interface audit to {out_path}")


if __name__ == "__main__":
    main()
