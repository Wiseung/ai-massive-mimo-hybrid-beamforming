#!/usr/bin/env python
"""Generate a stable-candidate readiness report for the interface-first Sionna RC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-consistency", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--smoke-matrix", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    release_consistency = _load(args.release_consistency)
    provenance = _load(args.provenance)
    smoke = _load(args.smoke_matrix)
    blocking_issues = []
    if not release_consistency["artifact_paths_exist"]:
        blocking_issues.append("missing_artifact_paths")
    if not provenance["all_artifacts_exist"]:
        blocking_issues.append("artifact_provenance_missing_files")
    if not provenance["all_full_native_only_false"]:
        blocking_issues.append("full_native_only_violation")
    if not provenance["all_rt_ray_5g_flags_false"]:
        blocking_issues.append("forbidden_scope_flag_violation")
    if not all(row["status"] == "ok" for row in smoke["scenarios"]):
        blocking_issues.append("smoke_matrix_failure")
    payload = {
        "ready_for_v1_0_0_final": len(blocking_issues) == 0,
        "blocking_issues": blocking_issues,
        "nonblocking_issues": [],
        "documentation_issues": [],
        "artifact_issues": [] if provenance["all_artifacts_exist"] else ["missing_artifacts"],
        "reproduction_issues": [] if all(row["status"] == "ok" for row in smoke["scenarios"]) else ["smoke_matrix_failure"],
        "recommended_next_action": "release_v1_0_0_final" if len(blocking_issues) == 0 else "fix_blockers",
        "final_statement": "interface-first release candidate only; not full native-only benchmark",
    }
    write_json(out_dir / "stable_readiness_report.json", payload)
    write_markdown(
        out_dir / "stable_readiness_report.md",
        [
            "# Stable Readiness Report",
            "",
            f"- ready_for_v1_0_0_final: `{payload['ready_for_v1_0_0_final']}`",
            f"- blocking_issues: `{payload['blocking_issues']}`",
            f"- recommended_next_action: `{payload['recommended_next_action']}`",
            f"- final_statement: `{payload['final_statement']}`",
        ],
    )
    print(f"Saved stable readiness report to {out_dir}")


if __name__ == "__main__":
    main()
