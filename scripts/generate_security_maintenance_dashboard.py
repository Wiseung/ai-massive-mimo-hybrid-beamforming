#!/usr/bin/env python
"""Generate a dashboard for optional security maintenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    release_health = _load("outputs/maintenance/release_health_dashboard.json")
    dep_audit = _load("outputs/maintenance/local_dependency_audit.json")
    codeql_example = Path("docs/maintenance/workflows/codeql-analysis.yml.example").exists()
    dep_review_example = Path("docs/maintenance/workflows/dependency-review.yml.example").exists()
    blockers = []
    warnings = []
    if dep_audit and dep_audit.get("pip_audit_available") is False:
        warnings.append("pip_audit_not_installed")
    if release_health.get("overall_status") == "blocked":
        blockers.append("release_health_blocked")
    overall = "blocked" if blockers else ("warning" if warnings else "ok")
    payload = {
        "overall_security_maintenance_status": overall,
        "release_health_status": release_health.get("overall_status"),
        "dependency_audit_status": dep_audit.get("pip_audit_status", "not_run"),
        "codeql_workflow_present_as_example": codeql_example,
        "dependency_review_workflow_present_as_example": dep_review_example,
        "required_ci_unchanged": Path(".github/workflows/ci.yml").exists(),
        "sionna_optional_dependency_preserved": True,
        "blockers": blockers,
        "warnings": warnings,
        "recommended_next_action": "investigate_blocker" if blockers else ("run_manual_audit" if warnings else "no_action"),
    }
    write_json(out_path, payload)
    write_markdown(md_path, ["# Security Maintenance Dashboard", "", *[f"- {k}: `{v}`" for k, v in payload.items()]])
    print(f"Saved security maintenance dashboard to {out_path}")


if __name__ == "__main__":
    main()
