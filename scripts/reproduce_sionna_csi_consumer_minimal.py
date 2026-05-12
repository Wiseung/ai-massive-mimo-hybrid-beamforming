#!/usr/bin/env python
"""Run a very small CSI-consumer-unification reproduction check."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_ofdm_experiment_helpers import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=str(cwd))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = collect_sionna_env_info()
    if not env["sionna_import_ok"]:
        payload = {
            "status": "skipped",
            "sionna_import_ok": False,
            "csi_object_created": False,
            "all_consumers_accept_csi": False,
            "raw_only_high_priority_paths": None,
            "no_new_fallback_introduced": None,
            "native_receiver_success": False,
            "teacher_used_during_inference": False,
            "strict_equivalence_claim_allowed": False,
            "full_native_only": False,
            "reason": "Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`.",
            "sionna_rt_used": False,
            "ray_tracing_used": False,
            "fiveg_full_stack_used": False,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved minimal Sionna CSI-consumer summary to {out_path}")
        return

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "outputs/repro/debug/sionna_csi_consumer_minimal"
    audit_out = base_dir / "csi_consumer_audit.json"
    demo_out = base_dir / "unified_csi_consumers_summary.json"

    _run(["scripts/audit_csi_consumers.py", "--out", str(audit_out)], cwd=repo_root)
    _run(["scripts/demo_unified_csi_consumers.py", "--out", str(demo_out), "--seed", "0"], cwd=repo_root)

    audit_summary = load_json(audit_out)
    demo_summary = load_json(demo_out)

    payload = {
        "status": "ok",
        "sionna_import_ok": True,
        "sionna_version": env["sionna_version"],
        "csi_object_created": bool(demo_summary.get("csi_object_created", False)),
        "all_consumers_accept_csi": bool(demo_summary.get("all_consumers_accept_csi", False)),
        "raw_only_high_priority_paths": audit_summary.get("raw_only_high_priority_paths"),
        "no_new_fallback_introduced": bool(demo_summary.get("no_new_fallback_introduced", False)),
        "native_receiver_success": bool(demo_summary.get("native_receiver_success", False)),
        "teacher_used_during_inference": bool(demo_summary.get("teacher_used_during_inference", False)),
        "strict_equivalence_claim_allowed": False,
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
        "notes": [
            "This is a reviewer-oriented minimal reproduction check for CSI consumer unification.",
            "It does not download data and does not train any model.",
            "Unified-vs-baseline comparison semantics remain cross-run only; this minimal path is not a strict equivalence test.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved minimal Sionna CSI-consumer summary to {out_path}")


if __name__ == "__main__":
    main()
