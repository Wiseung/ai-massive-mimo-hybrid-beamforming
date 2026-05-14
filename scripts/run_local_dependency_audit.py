#!/usr/bin/env python
"""Run a local/manual dependency audit without modifying the environment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _run(cmd: list[str]) -> tuple[str, int]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).strip(), proc.returncode


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    pip_check_output, pip_check_code = _run([sys.executable, "-m", "pip", "check"])
    pip_audit_available = shutil.which("pip-audit") is not None
    pip_audit_status = "skipped"
    skipped_reason = ""
    vulnerabilities_found = None
    advisory = "Run manually in trusted environments only."
    recommended_next_action = "no_action" if pip_check_code == 0 else "run_manual_audit"
    if pip_audit_available:
        pip_audit_output, pip_audit_code = _run(["pip-audit"])
        pip_audit_status = "passed" if pip_audit_code == 0 else "warning"
        vulnerabilities_found = None if pip_audit_code == 0 else pip_audit_output
        if pip_audit_code != 0:
            recommended_next_action = "review_dependency_alerts"
    else:
        skipped_reason = "pip_audit_not_installed"
        if pip_check_code == 0:
            recommended_next_action = "install_pip_audit_and_rerun"
    payload = {
        "pip_check_status": "passed" if pip_check_code == 0 else "failed",
        "pip_audit_available": pip_audit_available,
        "pip_audit_status": pip_audit_status,
        "skipped_reason": skipped_reason,
        "vulnerabilities_found": vulnerabilities_found,
        "advisory": advisory,
        "recommended_next_action": recommended_next_action,
    }
    write_json(out_path, payload)
    write_markdown(md_path, ["# Local Dependency Audit", "", *[f"- {k}: `{v}`" for k, v in payload.items()]])
    print(f"Saved local dependency audit to {out_path}")


if __name__ == "__main__":
    main()
