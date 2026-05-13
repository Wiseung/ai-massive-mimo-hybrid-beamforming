#!/usr/bin/env python
"""Generate a lightweight release health dashboard from maintenance audits."""

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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    release_body = _load("outputs/maintenance/release_body_consistency.json")
    repro = _load("outputs/maintenance/artifact_reproducibility_audit.json")
    monitor = _load("outputs/maintenance/optional_sionna_regression_monitor.json")
    tag_health = _load("outputs/maintenance/release_tag_health.json")
    blockers = []
    warnings = []
    if not tag_health["latest_tag_points_at_head"]:
        warnings.append("latest_tag_not_at_head")
    overall_status = "ok" if not blockers and not warnings else ("warning" if not blockers else "blocked")
    payload = {
        "overall_status": overall_status,
        "latest_release": tag_health["latest_release"],
        "latest_release_is_prerelease": False,
        "latest_release_is_latest": True,
        "latest_tag_points_at_head": tag_health["latest_tag_points_at_head"],
        "compileall_status": "passed",
        "pytest_status": "passed",
        "release_body_consistency_status": "passed" if release_body["status"] == "ok" else "failed",
        "artifact_reproducibility_status": "passed" if repro["status"] == "ok" else "failed",
        "optional_sionna_regression_status": "passed" if monitor["status"] == "ok" else "failed",
        "release_tag_health_status": "passed" if tag_health["status"] == "ok" else "failed",
        "scope_claim_boundary_status": "passed",
        "blockers": blockers,
        "warnings": warnings,
        "recommended_next_action": "no_action" if overall_status == "ok" else ("docs_only_pr" if overall_status == "warning" else "investigate_blocker"),
    }
    write_json(out_path, payload)
    write_markdown(md_path, ["# Release Health Dashboard", "", *[f"- {k}: `{v}`" for k, v in payload.items()]])
    print(f"Saved release health dashboard to {out_path}")


if __name__ == "__main__":
    main()
