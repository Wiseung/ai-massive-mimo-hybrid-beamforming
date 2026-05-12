#!/usr/bin/env python
"""Run a very small Sionna channel-extraction reproduction check."""

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


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *cmd], check=True, cwd=str(cwd))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = collect_sionna_env_info()
    if not env["sionna_import_ok"]:
        payload = {
            "status": "skipped",
            "sionna_import_ok": False,
            "reason": "Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`.",
            "sionna_rt_used": False,
            "ray_tracing_used": False,
            "fiveg_full_stack_used": False,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved minimal Sionna channel-extraction summary to {out_path}")
        return

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "outputs/repro/debug/sionna_channel_extraction_minimal"
    extract_out = base_dir / "extract_h_f_demo_summary.json"
    axis_out = base_dir / "hf_axis_validation.json"
    beamforming_out = base_dir / "native_channel_beamforming_summary.json"

    _run(
        ["scripts/sionna_extract_channel_hf_demo.py", "--out", str(extract_out)],
        cwd=repo_root,
    )
    _run(
        ["scripts/validate_sionna_extracted_hf_axes.py", "--out", str(axis_out)],
        cwd=repo_root,
    )
    _run(
        [
            "scripts/sionna_native_channel_beamforming_chain.py",
            "--out",
            str(beamforming_out),
            "--receiver-mode",
            "auto",
        ],
        cwd=repo_root,
    )

    extract_summary = load_json(extract_out)
    axis_summary = load_json(axis_out)
    beamforming_summary = load_json(beamforming_out)

    payload = {
        "status": "ok",
        "sionna_import_ok": True,
        "sionna_version": env["sionna_version"],
        "extraction_success": bool(extract_summary.get("extraction_success")),
        "extracted_h_f_shape": extract_summary.get("extracted_h_f_shape"),
        "axis_validation_passed": bool(axis_summary.get("axis_spot_check_passed")),
        "native_receiver_success": bool(beamforming_summary.get("native_receiver_success")),
        "project_h_f_assisted": bool(beamforming_summary.get("project_h_f_assisted", True)),
        "full_native_only": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "fiveg_full_stack_used": False,
        "notes": [
            "This is a reviewer-oriented minimal reproduction check for the Sionna channel-extraction bridge.",
            "It does not download data and does not train any model.",
            "The supported interpretation remains native-channel-assisted plus native-receiver-assisted, not full native-only.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved minimal Sionna channel-extraction summary to {out_path}")


if __name__ == "__main__":
    main()
