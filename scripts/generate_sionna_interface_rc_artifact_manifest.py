#!/usr/bin/env python
"""Generate a release-candidate manifest across the interface-first Sionna bridge lineage."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "name": "channel_extraction_artifact_manifest",
        "path": "outputs/sionna_channel_extraction/channel_extraction_artifact_manifest.json",
        "description": "v0.5 channel-extraction bridge artifact manifest.",
        "generating_command": "python scripts/generate_sionna_channel_extraction_artifact_manifest.py --out outputs/sionna_channel_extraction/channel_extraction_artifact_manifest.json",
        "interface_layer": "channel_extraction",
    },
    {
        "name": "csi_interface_artifact_manifest",
        "path": "outputs/sionna_channel_extraction/csi_interface_artifact_manifest.json",
        "description": "v0.6 ExtractedCSI interface artifact manifest.",
        "generating_command": "python scripts/generate_sionna_csi_interface_artifact_manifest.py --out outputs/sionna_channel_extraction/csi_interface_artifact_manifest.json",
        "interface_layer": "csi_interface",
    },
    {
        "name": "csi_consumer_artifact_manifest",
        "path": "outputs/sionna_channel_extraction/csi_consumer_artifact_manifest.json",
        "description": "v0.7 CSI consumer unification artifact manifest.",
        "generating_command": "python scripts/generate_sionna_csi_consumer_artifact_manifest.py --out outputs/sionna_channel_extraction/csi_consumer_artifact_manifest.json",
        "interface_layer": "csi_consumer",
    },
    {
        "name": "precoder_output_artifact_manifest",
        "path": "outputs/sionna_channel_extraction/precoder_interface_artifact_manifest.json",
        "description": "v0.8 PrecoderOutput bridge artifact manifest.",
        "generating_command": "python scripts/generate_sionna_precoder_interface_artifact_manifest.py --out outputs/sionna_channel_extraction/precoder_interface_artifact_manifest.json",
        "interface_layer": "precoder_output",
    },
    {
        "name": "native_precoder_artifact_manifest",
        "path": "outputs/sionna_precoder_api/native_precoder_artifact_manifest.json",
        "description": "v0.9 optional Sionna native precoder bridge artifact manifest.",
        "generating_command": "python scripts/generate_sionna_native_precoder_artifact_manifest.py --out outputs/sionna_precoder_api/native_precoder_artifact_manifest.json",
        "interface_layer": "sionna_native_precoder",
    },
    {
        "name": "native_precoder_contract_validation",
        "path": "outputs/sionna_precoder_api/native_precoder_contract_validation.json",
        "description": "Contract validation for the optional Sionna native precoder bridge.",
        "generating_command": "python scripts/validate_sionna_native_precoder_contract.py --out outputs/sionna_precoder_api/native_precoder_contract_validation.json",
        "interface_layer": "contract_hardening",
    },
    {
        "name": "native_precoder_contract_demo",
        "path": "outputs/sionna_precoder_api/native_precoder_contract_demo.json",
        "description": "Contract-aware demo for ExtractedCSI -> Sionna native precoder -> PrecoderOutput -> native receiver.",
        "generating_command": "python scripts/demo_sionna_native_precoder_contract.py --out outputs/sionna_precoder_api/native_precoder_contract_demo.json",
        "interface_layer": "contract_hardening",
    },
    {
        "name": "native_precoder_contract_matrix",
        "path": "outputs/sionna_precoder_api/native_precoder_contract_matrix.json",
        "description": "Regression matrix for contract / skip / fallback semantics.",
        "generating_command": "python scripts/test_sionna_native_precoder_contract_matrix.py --out outputs/sionna_precoder_api/native_precoder_contract_matrix.json",
        "interface_layer": "contract_hardening",
    },
    {
        "name": "native_precoder_minimal_reproduction",
        "path": "outputs/repro/sionna_native_precoder_minimal_summary.json",
        "description": "Minimal reproduction for the optional native precoder bridge.",
        "generating_command": "python scripts/reproduce_sionna_native_precoder_minimal.py --out outputs/repro/sionna_native_precoder_minimal_summary.json",
        "interface_layer": "sionna_native_precoder",
    },
    {
        "name": "interface_rc_minimal_reproduction",
        "path": "outputs/repro/sionna_interface_rc_minimal_summary.json",
        "description": "Minimal end-to-end reproduction for the interface-first Sionna bridge RC.",
        "generating_command": "python scripts/reproduce_sionna_interface_rc_minimal.py --out outputs/repro/sionna_interface_rc_minimal_summary.json",
        "interface_layer": "contract_hardening",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def _artifact_row(spec: dict[str, str], commit: str) -> dict[str, Any]:
    path = Path(spec["path"])
    payload = _load_json(path) or {}
    row: dict[str, Any] = {
        "name": spec["name"],
        "path": spec["path"],
        "description": spec["description"],
        "generating_command": spec["generating_command"],
        "generated_from_commit": commit,
        "interface_layer": spec["interface_layer"],
        "optional_sionna_dependency": True,
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
        "strict_equivalence_claim_allowed": payload.get("strict_equivalence_claim_allowed"),
        "relationship_status": payload.get("relationship_status"),
        "contract_valid": payload.get("contract_valid"),
        "regression_matrix_passed": payload.get("all_scenarios_contract_compliant"),
        "exists": path.exists(),
    }
    if spec["name"] == "native_precoder_artifact_manifest":
        row["relationship_status"] = "close_but_different"
        row["strict_equivalence_claim_allowed"] = False
    if spec["name"] == "native_precoder_minimal_reproduction":
        row["relationship_status"] = payload.get("relationship_status")
        row["strict_equivalence_claim_allowed"] = payload.get("strict_equivalence_claim_allowed")
    if spec["name"] == "interface_rc_minimal_reproduction":
        row["contract_valid"] = payload.get("contract_valid")
        row["regression_matrix_passed"] = payload.get("contract_matrix_passed")
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
            "This v1.0.0-rc1 manifest summarizes the interface-first Sionna bridge lineage. "
            "It is release-candidate material only and does not imply production e2e, full native-only, "
            "or strict project_rzf equivalence."
        ),
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Interface-first Sionna Bridge RC Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: interface-first release candidate only; optional Sionna dependency remains in place.",
        "",
        "| name | layer | exists | strict_equivalence_claim_allowed | relationship_status | contract_valid | regression_matrix_passed | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['interface_layer']} | {row['exists']} | {row['strict_equivalence_claim_allowed']} | "
            f"{row['relationship_status']} | {row['contract_valid']} | {row['regression_matrix_passed']} | "
            f"`{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved interface RC artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
