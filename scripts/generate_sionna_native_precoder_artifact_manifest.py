#!/usr/bin/env python
"""Generate a manifest for Sionna native precoder optional-bridge artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "name": "native_precoder_api_audit",
        "path": "outputs/sionna_precoder_api/native_precoder_api_audit.json",
        "description": "Audit of Sionna 2.0.1 native precoder-related APIs and current adapter compatibility.",
        "command": "python scripts/audit_sionna_native_precoder_api.py --out outputs/sionna_precoder_api/native_precoder_api_audit.json",
    },
    {
        "name": "rzf_precoder_probe_summary",
        "path": "outputs/sionna_precoder_api/rzf_precoder_probe_summary.json",
        "description": "Minimal callable Sionna RZFPrecoder probe with ExtractedCSI -> PrecoderOutput conversion.",
        "command": "python scripts/probe_sionna_rzf_precoder_bridge.py --out outputs/sionna_precoder_api/rzf_precoder_probe_summary.json",
    },
    {
        "name": "sionna_rzf_same_realization",
        "path": "outputs/sionna_precoder_api/sionna_rzf_same_realization.json",
        "description": "Same-realization project_rzf vs Sionna RZFPrecoder semantic validation under one shared CSI / symbol / receiver realization.",
        "command": "python scripts/validate_sionna_rzf_same_realization.py --out outputs/sionna_precoder_api/sionna_rzf_same_realization.json",
    },
    {
        "name": "sionna_rzf_alignment_quick",
        "path": "outputs/sionna_precoder_api/sionna_rzf_alignment_quick/summary.md",
        "description": "Quick seed/SNR alignment sweep for the optional Sionna RZFPrecoder bridge.",
        "command": "python scripts/benchmark_sionna_rzf_precoder_alignment.py --quick --seeds 1 2 3 --snrs 0 5 10 15 20 --out outputs/sionna_precoder_api/sionna_rzf_alignment_quick",
    },
    {
        "name": "project_vs_sionna_precoder_comparison_v2",
        "path": "outputs/sionna_precoder_api/project_vs_sionna_precoder_comparison_v2.md",
        "description": "Consolidated comparison report for project_rzf vs optional Sionna native RZF bridge.",
        "command": "python scripts/compare_project_vs_sionna_precoder.py --same-realization outputs/sionna_precoder_api/sionna_rzf_same_realization.json --alignment outputs/sionna_precoder_api/sionna_rzf_alignment_quick/metrics.csv --unified outputs/sionna_channel_extraction/unified_csi_precoder_metrics.csv --out outputs/sionna_precoder_api",
    },
    {
        "name": "unified_csi_precoder_summary_with_sionna_rzf",
        "path": "outputs/sionna_channel_extraction/unified_csi_precoder_summary.json",
        "description": "Unified CSI + PrecoderOutput demo including the optional sionna_rzf_precoder method when available.",
        "command": "python scripts/demo_unified_csi_and_precoder_interfaces.py --out outputs/sionna_channel_extraction/unified_csi_precoder_summary.json --include-sionna-rzf",
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


def _load_unified_summary() -> dict[str, Any]:
    return _load_json(Path("outputs/sionna_channel_extraction/unified_csi_precoder_summary.json")) or {}


def _load_same_realization() -> dict[str, Any]:
    return _load_json(Path("outputs/sionna_precoder_api/sionna_rzf_same_realization.json")) or {}


def _load_probe_summary() -> dict[str, Any]:
    return _load_json(Path("outputs/sionna_precoder_api/rzf_precoder_probe_summary.json")) or {}


def _comparison_v2_row() -> dict[str, Any] | None:
    csv_path = Path("outputs/sionna_precoder_api/project_vs_sionna_precoder_comparison_v2.csv")
    if not csv_path.exists():
        return None
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return None
    headers = lines[0].split(",")
    values = lines[1].split(",")
    return dict(zip(headers, values, strict=False))


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


def _row_from_spec(spec: dict[str, str], commit: str) -> dict[str, Any]:
    path = Path(spec["path"])
    payload = _load_json(path)
    unified = _load_unified_summary()
    same_realization = _load_same_realization()
    probe = _load_probe_summary()
    compare_v2 = _comparison_v2_row() or {}

    row: dict[str, Any] = {
        "name": spec["name"],
        "path": spec["path"],
        "description": spec["description"],
        "generating_command": spec["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        "sionna_rzf_available": None,
        "sionna_rzf_callable": None,
        "converted_to_precoder_output": None,
        "native_receiver_success": None,
        "sionna_native_precoder": None,
        "project_side_precoder": None,
        "relationship_status": None,
        "strict_equivalence_claim_allowed": None,
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }

    if spec["name"] == "native_precoder_api_audit" and payload is not None:
        summary = payload.get("summary", {})
        row.update(
            {
                "sionna_rzf_available": _coerce_bool(summary.get("sionna_rzf_precoder_available")),
                "sionna_rzf_callable": _coerce_bool(summary.get("sionna_rzf_precoder_available")),
                "converted_to_precoder_output": _coerce_bool(summary.get("compatible_with_current_interfaces")),
                "native_receiver_success": _coerce_bool(probe.get("native_receiver_success_if_attempted")),
                "sionna_native_precoder": True,
                "project_side_precoder": False,
                "relationship_status": same_realization.get("relationship_status"),
                "strict_equivalence_claim_allowed": _coerce_bool(same_realization.get("strict_equivalence_claim_allowed")),
            }
        )
    elif spec["name"] == "rzf_precoder_probe_summary" and payload is not None:
        row.update(
            {
                "sionna_rzf_available": _coerce_bool(payload.get("sionna_rzf_available")),
                "sionna_rzf_callable": _coerce_bool(payload.get("sionna_rzf_callable")),
                "converted_to_precoder_output": _coerce_bool(payload.get("converted_to_precoder_output")),
                "native_receiver_success": _coerce_bool(payload.get("native_receiver_success_if_attempted")),
                "sionna_native_precoder": True if payload.get("converted_to_precoder_output") else None,
                "project_side_precoder": False if payload.get("converted_to_precoder_output") else None,
                "relationship_status": same_realization.get("relationship_status"),
                "strict_equivalence_claim_allowed": _coerce_bool(same_realization.get("strict_equivalence_claim_allowed")),
            }
        )
    elif spec["name"] == "sionna_rzf_same_realization" and payload is not None:
        row.update(
            {
                "sionna_rzf_available": _coerce_bool(payload.get("sionna_rzf_available")),
                "sionna_rzf_callable": _coerce_bool(payload.get("sionna_rzf_callable")),
                "converted_to_precoder_output": _coerce_bool(payload.get("converted_to_precoder_output")),
                "native_receiver_success": _coerce_bool(payload.get("native_receiver_success_sionna")),
                "sionna_native_precoder": True if payload.get("converted_to_precoder_output") else None,
                "project_side_precoder": False if payload.get("converted_to_precoder_output") else None,
                "relationship_status": payload.get("relationship_status"),
                "strict_equivalence_claim_allowed": _coerce_bool(payload.get("strict_equivalence_claim_allowed")),
            }
        )
    elif spec["name"] == "sionna_rzf_alignment_quick":
        row.update(
            {
                "sionna_rzf_available": _coerce_bool(same_realization.get("sionna_rzf_available")),
                "sionna_rzf_callable": True,
                "converted_to_precoder_output": True,
                "native_receiver_success": True,
                "sionna_native_precoder": True,
                "project_side_precoder": False,
                "relationship_status": "close_but_different",
                "strict_equivalence_claim_allowed": False,
            }
        )
    elif spec["name"] == "project_vs_sionna_precoder_comparison_v2":
        row.update(
            {
                "sionna_rzf_available": _coerce_bool(same_realization.get("sionna_rzf_available")),
                "sionna_rzf_callable": _coerce_bool(compare_v2.get("sionna_rzf_callable")),
                "converted_to_precoder_output": _coerce_bool(compare_v2.get("converted_to_precoder_output")),
                "native_receiver_success": _coerce_bool(compare_v2.get("native_receiver_success_in_demo")),
                "sionna_native_precoder": _coerce_bool(compare_v2.get("sionna_native_precoder_true_now")),
                "project_side_precoder": False,
                "relationship_status": compare_v2.get("relationship_status"),
                "strict_equivalence_claim_allowed": _coerce_bool(compare_v2.get("strict_equivalence_claim_allowed")),
            }
        )
    elif spec["name"] == "unified_csi_precoder_summary_with_sionna_rzf" and payload is not None:
        row.update(
            {
                "sionna_rzf_available": _coerce_bool(payload.get("sionna_rzf_available")),
                "sionna_rzf_callable": _coerce_bool(payload.get("sionna_rzf_callable")),
                "converted_to_precoder_output": _coerce_bool(payload.get("sionna_rzf_evaluated")),
                "native_receiver_success": _coerce_bool(payload.get("native_receiver_success")),
                "sionna_native_precoder": True if payload.get("sionna_rzf_evaluated") else None,
                "project_side_precoder": False if payload.get("sionna_rzf_evaluated") else None,
                "relationship_status": same_realization.get("relationship_status"),
                "strict_equivalence_claim_allowed": _coerce_bool(same_realization.get("strict_equivalence_claim_allowed")),
            }
        )

    return row


def main() -> None:
    args = parse_args()
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_json.with_suffix(".md")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = [_row_from_spec(spec, commit) for spec in ARTIFACTS]

    payload = {
        "generated_from_commit": commit,
        "note": (
            "These artifacts document the optional Sionna native precoder bridge only. "
            "They do not justify a full native-only benchmark, a mainline native replacement claim, "
            "or a strict project_rzf equivalence claim."
        ),
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna Native Precoder Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: optional native-precoder bridge only; not full native-only and not strict project_rzf equivalence.",
        "",
        "| name | exists | sionna_rzf_available | sionna_rzf_callable | converted_to_precoder_output | native_receiver_success | sionna_native_precoder | project_side_precoder | relationship_status | strict_equivalence_claim_allowed | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['exists']} | {row['sionna_rzf_available']} | {row['sionna_rzf_callable']} | "
            f"{row['converted_to_precoder_output']} | {row['native_receiver_success']} | {row['sionna_native_precoder']} | "
            f"{row['project_side_precoder']} | {row['relationship_status']} | {row['strict_equivalence_claim_allowed']} | "
            f"`{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved Sionna native precoder artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
