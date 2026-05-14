#!/usr/bin/env python
"""Manual-first pip-audit runner with optional temporary venv installation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-install-in-venv", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> tuple[str, int]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).strip(), proc.returncode


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    warnings: list[str] = []
    blockers: list[str] = []
    pip_audit_available = shutil.which("pip-audit") is not None
    pip_audit_installed_in_temp_venv = False
    audit_status = "skipped"
    vulnerabilities_found = None
    vulnerability_count = None
    fix_versions_available_count = None
    skipped_reason = ""
    recommended_next_action = "install_pip_audit_and_rerun"
    manual_audit_attempted = True

    runner = None
    if pip_audit_available:
        runner = ["pip-audit"]
    elif args.allow_install_in_venv:
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="pip-audit-venv-"))
            venv.EnvBuilder(with_pip=True).create(tmp_dir)
            py = tmp_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            install_out, install_code = _run([str(py), "-m", "pip", "install", "pip-audit"])
            if install_code == 0:
                pip_audit_available = True
                pip_audit_installed_in_temp_venv = True
                runner = [str(py), "-m", "pip_audit"]
            else:
                warnings.append("pip_audit_temp_venv_install_failed")
                skipped_reason = install_out or "pip_audit_temp_venv_install_failed"
        except Exception as exc:  # pragma: no cover
            warnings.append("pip_audit_temp_venv_install_failed")
            skipped_reason = f"{type(exc).__name__}: {exc}"
    else:
        warnings.append("pip_audit_not_installed")
        skipped_reason = "pip_audit_not_installed"

    if runner is not None:
        output, code = _run(runner)
        if code == 0:
            audit_status = "passed"
            recommended_next_action = "no_action"
        else:
            audit_status = "warning"
            vulnerabilities_found = output
            recommended_next_action = "review_dependency_alerts"
            warnings.append("pip_audit_reported_findings_or_failed")

    payload = {
        "manual_audit_attempted": manual_audit_attempted,
        "pip_audit_available": pip_audit_available,
        "pip_audit_installed_in_temp_venv": pip_audit_installed_in_temp_venv,
        "audit_status": audit_status,
        "vulnerabilities_found": vulnerabilities_found,
        "vulnerability_count": vulnerability_count,
        "fix_versions_available_count": fix_versions_available_count,
        "skipped_reason": skipped_reason,
        "blockers": blockers,
        "warnings": warnings,
        "recommended_next_action": recommended_next_action,
    }
    write_json(out_path, payload)
    write_markdown(md_path, ["# Manual pip-audit", "", *[f"- {k}: `{v}`" for k, v in payload.items()]])
    print(f"Saved manual pip-audit summary to {out_path}")


if __name__ == "__main__":
    main()
