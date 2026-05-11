#!/usr/bin/env python
"""Run a very short optional Sionna learned-training reproduction check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_ofdm_experiment_helpers import run_python_command, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


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
        print(f"Saved minimal Sionna training summary to {out_path}")
        return

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "outputs/runs/sionna_training_minimal_repro"
    eval_dir = repo_root / "outputs/comparisons/sionna_training_minimal_repro"

    run_python_command(
        [
            "scripts/train_sionna_ofdm_beamformer.py",
            "--config",
            "configs/sionna_ofdm_residual_rzf.yaml",
            "--out",
            str(run_dir),
            "--smoke",
        ],
        cwd=repo_root,
    )
    run_python_command(
        [
            "scripts/evaluate_sionna_ofdm_beamformer.py",
            "--config",
            "configs/sionna_ofdm_residual_rzf.yaml",
            "--ckpt",
            str(run_dir / "best.pt"),
            "--out",
            str(eval_dir),
        ],
        cwd=repo_root,
    )

    train_summary = load_json(run_dir / "smoke_summary.json")
    eval_summary = load_json(eval_dir / "summary.json")
    payload = {
        "status": "ok",
        "sionna_import_ok": True,
        "sionna_version": env["sionna_version"],
        "model_name": "sionna_ofdm_residual_rzf",
        "train_summary": {
            "best_epoch": train_summary["best_epoch"],
            "best_val_loss": train_summary["best_val_loss"],
            "best_val_mean_sum_rate": train_summary["best_val_mean_sum_rate"],
            "used_sionna_ofdm": train_summary["used_sionna_ofdm"],
            "used_sionna_channel": train_summary["used_sionna_channel"],
        },
        "eval_summary": {
            "learned_mean_sum_rate": eval_summary["learned_mean_sum_rate"],
            "learned_mean_gap_to_rzf": eval_summary["learned_mean_gap_to_rzf"],
            "learned_mean_gap_to_wmmse_iter_5": eval_summary["learned_mean_gap_to_wmmse_iter_5"],
        },
        "notes": [
            "This is a reviewer-oriented minimal reproduction check.",
            "It uses residual-RZF smoke training only.",
            "It does not download data and does not run the full benchmark suite.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved minimal Sionna training summary to {out_path}")


if __name__ == "__main__":
    main()
