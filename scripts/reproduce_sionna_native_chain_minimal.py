#!/usr/bin/env python
"""Run a very small Sionna-native receiver-chain reproduction check."""

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
            "reason": "Sionna is not installed. Install the optional dependency with `pip install sionna-no-rt`.",
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved minimal Sionna native-chain summary to {out_path}")
        return

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "outputs/repro/debug/sionna_native_chain_minimal"
    baseline_out = base_dir / "beamforming_receiver_chain_v2_summary.json"
    learned_out = base_dir / "learned_beamforming_receiver_summary.json"

    _run(
        [
            "scripts/sionna_native_ofdm_beamforming_chain.py",
            "--out",
            str(baseline_out),
            "--enable-receiver-chain",
            "--receiver-mode",
            "auto",
        ],
        cwd=repo_root,
    )
    baseline_summary = load_json(baseline_out)

    learned_ckpt = repo_root / "outputs/runs/sionna_ofdm_residual_rzf/best.pt"
    learned_checkpoint_found = learned_ckpt.exists()
    learned_summary = None
    if learned_checkpoint_found:
        _run(
            [
                "scripts/sionna_native_ofdm_learned_beamforming_chain.py",
                "--out",
                str(learned_out),
                "--receiver-mode",
                "auto",
            ],
            cwd=repo_root,
        )
        learned_summary = load_json(learned_out)

    payload = {
        "status": "ok",
        "sionna_import_ok": True,
        "sionna_version": env["sionna_version"],
        "baseline_receiver_check": {
            "native_receiver_success": baseline_summary["native_receiver_success"],
            "used_sionna_channel": baseline_summary["used_sionna_channel"],
            "used_sionna_estimator": baseline_summary["used_sionna_estimator"],
            "used_sionna_equalizer": baseline_summary["used_sionna_equalizer"],
            "used_sionna_demapper": baseline_summary["used_sionna_demapper"],
            "project_h_f_assisted": True,
        },
        "learned_checkpoint_found": learned_checkpoint_found,
        "learned_receiver_check": None,
        "notes": [
            "This is a reviewer-oriented minimal reproduction check for the native receiver chain.",
            "It does not download data and does not train any model.",
            "The receiver path is real Sionna, while the precoder/H_f side remains project-assisted where noted.",
        ],
    }
    if learned_summary is not None:
        learned_rows = {row["method"]: row for row in learned_summary["metrics"]}
        learned_row = learned_rows.get("learned_residual_rzf")
        payload["learned_receiver_check"] = {
            "native_receiver_success": bool(learned_row and learned_row["native_receiver_success"]),
            "used_sionna_channel": learned_summary["used_sionna_channel"],
            "used_sionna_estimator": learned_summary["used_sionna_estimator"],
            "used_sionna_equalizer": learned_summary["used_sionna_equalizer"],
            "used_sionna_demapper": learned_summary["used_sionna_demapper"],
            "teacher_used_during_inference": bool(learned_row and learned_row["teacher_used_during_inference"]),
            "project_h_f_assisted": True,
            "checkpoint_path": str(learned_ckpt),
        }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved minimal Sionna native-chain summary to {out_path}")


if __name__ == "__main__":
    main()
