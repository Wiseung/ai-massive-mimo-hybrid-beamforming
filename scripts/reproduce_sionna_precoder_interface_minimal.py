#!/usr/bin/env python
"""Run a very small PrecoderOutput-bridge reproduction check."""

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
            "csi_interface_used": False,
            "precoder_interface_used": False,
            "precoder_output_created": False,
            "all_receiver_consumers_accept_precoder_output": False,
            "same_batch_equivalence_passed": False,
            "numeric_consistency_within_tolerance": False,
            "max_abs_diff_raw_f_f_vs_precoder_output": None,
            "native_receiver_success": False,
            "teacher_used_during_inference": False,
            "project_side_precoder": True,
            "sionna_native_precoder": False,
            "full_native_only": False,
            "reason": "Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`.",
            "sionna_rt_used": False,
            "ray_tracing_used": False,
            "fiveg_full_stack_used": False,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved minimal Sionna PrecoderOutput summary to {out_path}")
        return

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "outputs/repro/debug/sionna_precoder_interface_minimal"
    base_dir.mkdir(parents=True, exist_ok=True)
    demo_out = base_dir / "unified_csi_precoder_summary.json"
    equivalence_out = base_dir / "precoder_output_same_batch_equivalence.json"

    _run(["scripts/demo_unified_csi_and_precoder_interfaces.py", "--out", str(demo_out), "--seed", "0"], cwd=repo_root)
    _run(["scripts/validate_precoder_output_same_batch_equivalence.py", "--out", str(equivalence_out), "--seed", "0"], cwd=repo_root)

    demo_summary = load_json(demo_out)
    equivalence_summary = load_json(equivalence_out)

    payload = {
        "status": "ok",
        "sionna_import_ok": True,
        "sionna_version": env["sionna_version"],
        "csi_interface_used": bool(demo_summary.get("csi_interface_used", False)),
        "precoder_interface_used": bool(demo_summary.get("precoder_interface_used", False)),
        "precoder_output_created": bool(demo_summary.get("all_precoders_emit_precoder_output", False)),
        "all_receiver_consumers_accept_precoder_output": bool(demo_summary.get("all_receiver_consumers_accept_precoder_output", False)),
        "same_batch_equivalence_passed": bool(
            equivalence_summary.get("status") == "ok"
            and equivalence_summary.get("strict_equivalence_claim_allowed", False)
        ),
        "numeric_consistency_within_tolerance": bool(equivalence_summary.get("numeric_consistency_within_tolerance", False)),
        "max_abs_diff_raw_f_f_vs_precoder_output": equivalence_summary.get("max_abs_diff_raw_f_f_vs_precoder_output"),
        "native_receiver_success": bool(demo_summary.get("native_receiver_success", False)),
        "teacher_used_during_inference": bool(demo_summary.get("teacher_used_during_inference", False)),
        "project_side_precoder": True,
        "sionna_native_precoder": False,
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
        "notes": [
            "This is a reviewer-oriented minimal reproduction check for the ExtractedCSI + PrecoderOutput bridge.",
            "It does not download data and does not train any model.",
            "The supported interpretation remains project-side precoder bridge plus Sionna-native channel/receiver path, not full native-only.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved minimal Sionna PrecoderOutput summary to {out_path}")


if __name__ == "__main__":
    main()
