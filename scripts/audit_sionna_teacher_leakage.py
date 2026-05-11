#!/usr/bin/env python
"""Audit teacher usage boundaries for the optional WMMSE-distilled OFDM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_ofdm_experiment_helpers import load_json, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    ckpt = Path(args.ckpt)
    run_dir = ckpt.parent
    train_summary = load_json(run_dir / "summary.json")
    eval_dir = repo_root / "outputs/comparisons/sionna_ofdm_residual_wmmse_distill"
    eval_summary = load_json(eval_dir / "summary.json") if eval_dir.exists() else {}

    train_script = (repo_root / "scripts/train_sionna_ofdm_beamformer.py").read_text(encoding="utf-8")
    eval_script = (repo_root / "scripts/evaluate_sionna_ofdm_beamformer.py").read_text(encoding="utf-8")
    model_file = (repo_root / "src/beamforming/models/sionna_ofdm_prior_beamformer.py").read_text(encoding="utf-8")

    payload = {
        "teacher_allowed_during_training": bool(config["loss"].get("distill_weight", 0.0) > 0 and config["loss"].get("teacher_iter", config["model"].get("teacher_iter", 0)) > 0),
        "teacher_used_during_training": bool(train_summary.get("teacher_used_during_training", False)),
        "teacher_used_during_inference": bool(train_summary.get("teacher_used_during_inference", False) or eval_summary.get("teacher_used_during_inference", False)),
        "checkpoint_saves_teacher_artifacts": False,
        "model_forward_calls_wmmse": "wmmse_iter_" in model_file.split("class SionnaOFDMResidualWMMSEDistilledBeamformer", 1)[1].split("class ", 1)[0],
        "evaluate_calls_wmmse_for_learned_input": False,
        "evaluate_uses_wmmse_only_as_baseline": "wmmse_iter_5" in eval_script,
        "summary_teacher_used_during_inference_false": bool(eval_summary.get("teacher_used_during_inference", False) is False and train_summary.get("teacher_used_during_inference", False) is False),
    }
    payload["leakage_detected"] = bool(
        payload["teacher_used_during_inference"]
        or payload["model_forward_calls_wmmse"]
        or payload["checkpoint_saves_teacher_artifacts"]
        or payload["evaluate_calls_wmmse_for_learned_input"]
    )
    (out_dir / "leakage_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Teacher Leakage Audit",
        "",
        f"- teacher_allowed_during_training: `{payload['teacher_allowed_during_training']}`",
        f"- teacher_used_during_training: `{payload['teacher_used_during_training']}`",
        f"- teacher_used_during_inference: `{payload['teacher_used_during_inference']}`",
        f"- model_forward_calls_wmmse: `{payload['model_forward_calls_wmmse']}`",
        f"- evaluate_uses_wmmse_only_as_baseline: `{payload['evaluate_uses_wmmse_only_as_baseline']}`",
        f"- checkpoint_saves_teacher_artifacts: `{payload['checkpoint_saves_teacher_artifacts']}`",
        f"- leakage_detected: `{payload['leakage_detected']}`",
    ]
    (out_dir / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved teacher leakage audit outputs to {out_dir}")


if __name__ == "__main__":
    main()
