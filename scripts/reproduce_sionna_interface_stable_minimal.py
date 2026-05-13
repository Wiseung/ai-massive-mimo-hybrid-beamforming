#!/usr/bin/env python
"""Stable-stage minimal wrapper for the interface-first Sionna bridge."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=str(cwd))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    _run(["scripts/reproduce_sionna_interface_rc_minimal.py", "--out", "outputs/repro/sionna_interface_rc_minimal_summary.json"], cwd=repo_root)
    _run(["scripts/audit_sionna_interface_rc_release_consistency.py", "--out", "outputs/sionna_interface_rc/interface_rc_release_consistency.json"], cwd=repo_root)
    _run(["scripts/audit_sionna_interface_rc_artifact_provenance.py", "--out", "outputs/sionna_interface_rc/interface_rc_artifact_provenance.json"], cwd=repo_root)
    _run(["scripts/run_sionna_interface_rc_smoke_matrix.py", "--out", "outputs/sionna_interface_rc/interface_rc_smoke_matrix.json"], cwd=repo_root)
    _run(
        [
            "scripts/generate_sionna_interface_stable_readiness_report.py",
            "--release-consistency",
            "outputs/sionna_interface_rc/interface_rc_release_consistency.json",
            "--provenance",
            "outputs/sionna_interface_rc/interface_rc_artifact_provenance.json",
            "--smoke-matrix",
            "outputs/sionna_interface_rc/interface_rc_smoke_matrix.json",
            "--out",
            "outputs/sionna_interface_rc",
        ],
        cwd=repo_root,
    )

    rc_min = _load(repo_root / "outputs/repro/sionna_interface_rc_minimal_summary.json")
    rel = _load(repo_root / "outputs/sionna_interface_rc/interface_rc_release_consistency.json")
    prov = _load(repo_root / "outputs/sionna_interface_rc/interface_rc_artifact_provenance.json")
    smoke = _load(repo_root / "outputs/sionna_interface_rc/interface_rc_smoke_matrix.json")
    ready = _load(repo_root / "outputs/sionna_interface_rc/stable_readiness_report.json")

    payload = {
        "status": "ok",
        "sionna_import_ok": rc_min.get("sionna_import_ok"),
        "rc_minimal_status": rc_min.get("status"),
        "release_consistency_passed": rel.get("status") == "ok" and rel.get("artifact_paths_exist"),
        "artifact_provenance_passed": prov.get("status") == "ok" and prov.get("all_artifacts_exist"),
        "smoke_matrix_passed": smoke.get("status") == "ok",
        "stable_readiness_passed": ready.get("ready_for_v1_0_0_final"),
        "ready_for_v1_0_0_final": ready.get("ready_for_v1_0_0_final"),
        "blocking_issues": ready.get("blocking_issues"),
        "nonblocking_issues": ready.get("nonblocking_issues"),
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved stable minimal summary to {out_path}")


if __name__ == "__main__":
    main()
