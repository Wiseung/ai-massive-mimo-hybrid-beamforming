#!/usr/bin/env python
"""Generate a stable-stage artifact manifest for the interface-first Sionna bridge."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ARTIFACTS = [
    ("interface_rc_artifact_manifest", "outputs/sionna_interface_rc/interface_rc_artifact_manifest.json", "rc1"),
    ("interface_rc_minimal_summary", "outputs/repro/sionna_interface_rc_minimal_summary.json", "rc1"),
    ("interface_rc_release_consistency", "outputs/sionna_interface_rc/interface_rc_release_consistency.json", "stable_candidate"),
    ("interface_rc_artifact_provenance", "outputs/sionna_interface_rc/interface_rc_artifact_provenance.json", "stable_candidate"),
    ("interface_rc_smoke_matrix", "outputs/sionna_interface_rc/interface_rc_smoke_matrix.json", "stable_candidate"),
    ("stable_readiness_report", "outputs/sionna_interface_rc/stable_readiness_report.json", "stable_candidate"),
    ("native_precoder_contract_validation", "outputs/sionna_precoder_api/native_precoder_contract_validation.json", "stable"),
    ("native_precoder_contract_demo", "outputs/sionna_precoder_api/native_precoder_contract_demo.json", "stable"),
    ("native_precoder_contract_matrix", "outputs/sionna_precoder_api/native_precoder_contract_matrix.json", "stable"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    args = parse_args()
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_json.with_suffix(".md")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = []
    for name, path_str, stage in ARTIFACTS:
        path = Path(path_str)
        payload = _load(path)
        rows.append(
            {
                "name": name,
                "path": path_str,
                "description": name.replace("_", " "),
                "generating_command": "",
                "generated_from_commit": commit,
                "release_stage": stage,
                "optional_sionna_dependency": True,
                "full_native_only": False,
                "sionna_rt_used": False,
                "ray_tracing_used": False,
                "fiveg_full_stack_used": False,
                "strict_equivalence_claim_allowed": payload.get("strict_equivalence_claim_allowed"),
                "relationship_status": payload.get("relationship_status"),
                "contract_valid": payload.get("contract_valid"),
                "regression_matrix_passed": payload.get("all_scenarios_contract_compliant") or payload.get("contract_matrix_passed"),
                "ready_for_v1_0_0_final": payload.get("ready_for_v1_0_0_final"),
                "exists": path.exists(),
            }
        )
    payload = {"generated_from_commit": commit, "artifacts": rows}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Interface-first Sionna Bridge Stable Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "",
        "| name | stage | exists | relationship_status | contract_valid | regression_matrix_passed | ready_for_v1_0_0_final |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['release_stage']} | {row['exists']} | {row['relationship_status']} | "
            f"{row['contract_valid']} | {row['regression_matrix_passed']} | {row['ready_for_v1_0_0_final']} |"
        )
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved stable artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
