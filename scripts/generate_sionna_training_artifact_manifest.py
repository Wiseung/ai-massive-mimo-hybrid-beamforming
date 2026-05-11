#!/usr/bin/env python
"""Generate a manifest for optional Sionna learned-training artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ARTIFACTS = [
    {
        "path": "outputs/runs/sionna_ofdm_learned_beamformer/summary.json",
        "description": "TinyNeuralBeamformer full training summary.",
        "command": "python scripts/train_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_learned_beamformer.yaml --out outputs/runs/sionna_ofdm_learned_beamformer",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_learned_beamformer/summary.json",
        "description": "TinyNeuralBeamformer full evaluation summary.",
        "command": "python scripts/evaluate_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_learned_beamformer.yaml --ckpt outputs/runs/sionna_ofdm_learned_beamformer/best.pt --out outputs/comparisons/sionna_ofdm_learned_beamformer",
        "quick": False,
    },
    {
        "path": "outputs/runs/sionna_ofdm_residual_rzf/summary.json",
        "description": "Residual-RZF full training summary.",
        "command": "python scripts/train_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_residual_rzf.yaml --out outputs/runs/sionna_ofdm_residual_rzf",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_residual_rzf/summary.json",
        "description": "Residual-RZF full evaluation summary.",
        "command": "python scripts/evaluate_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_residual_rzf.yaml --ckpt outputs/runs/sionna_ofdm_residual_rzf/best.pt --out outputs/comparisons/sionna_ofdm_residual_rzf",
        "quick": False,
    },
    {
        "path": "outputs/runs/sionna_ofdm_unfolded_lite/summary.json",
        "description": "Unfolded-Lite full training summary.",
        "command": "python scripts/train_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_unfolded_lite.yaml --out outputs/runs/sionna_ofdm_unfolded_lite",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_unfolded_lite/summary.json",
        "description": "Unfolded-Lite full evaluation summary.",
        "command": "python scripts/evaluate_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_unfolded_lite.yaml --ckpt outputs/runs/sionna_ofdm_unfolded_lite/best.pt --out outputs/comparisons/sionna_ofdm_unfolded_lite",
        "quick": False,
    },
    {
        "path": "outputs/runs/sionna_ofdm_residual_wmmse_distill/summary.json",
        "description": "Residual WMMSE-distill full training summary.",
        "command": "python scripts/train_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_residual_wmmse_distill.yaml --out outputs/runs/sionna_ofdm_residual_wmmse_distill",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_residual_wmmse_distill/summary.json",
        "description": "Residual WMMSE-distill full evaluation summary.",
        "command": "python scripts/evaluate_sionna_ofdm_beamformer.py --config configs/sionna_ofdm_residual_wmmse_distill.yaml --ckpt outputs/runs/sionna_ofdm_residual_wmmse_distill/best.pt --out outputs/comparisons/sionna_ofdm_residual_wmmse_distill",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_training_family/family_summary.md",
        "description": "Family comparison for tiny/residual/unfolded learned methods.",
        "command": "python scripts/compare_sionna_ofdm_training_runs.py --tiny outputs/comparisons/sionna_ofdm_learned_beamformer --residual outputs/comparisons/sionna_ofdm_residual_rzf --unfolded outputs/comparisons/sionna_ofdm_unfolded_lite --out outputs/comparisons/sionna_ofdm_training_family",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_training_family_v2/family_summary.md",
        "description": "Family v2 comparison including the residual WMMSE-distilled model.",
        "command": "python scripts/compare_sionna_ofdm_training_runs.py --tiny outputs/comparisons/sionna_ofdm_learned_beamformer --residual outputs/comparisons/sionna_ofdm_residual_rzf --unfolded outputs/comparisons/sionna_ofdm_unfolded_lite --wmmse-distill outputs/comparisons/sionna_ofdm_residual_wmmse_distill --out outputs/comparisons/sionna_ofdm_training_family_v2",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_multiseed/multiseed_summary.csv",
        "description": "Quick multi-seed robustness summary.",
        "command": "python scripts/run_sionna_ofdm_multiseed_benchmark.py --configs configs/sionna_ofdm_learned_beamformer.yaml configs/sionna_ofdm_residual_rzf.yaml configs/sionna_ofdm_unfolded_lite.yaml --seeds 1 2 3 --out outputs/comparisons/sionna_ofdm_multiseed --quick",
        "quick": True,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_latency/latency_table.csv",
        "description": "Latency and parameter benchmark for learned OFDM methods and baselines.",
        "command": "python scripts/benchmark_sionna_ofdm_models.py --out outputs/comparisons/sionna_ofdm_latency",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_residual_analysis/summary.md",
        "description": "Residual correction analysis for the residual-RZF model.",
        "command": "python scripts/analyze_sionna_residual_corrections.py --config configs/sionna_ofdm_residual_rzf.yaml --ckpt outputs/runs/sionna_ofdm_residual_rzf/best.pt --out outputs/comparisons/sionna_ofdm_residual_analysis",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_scale_sweep/scale_sweep.md",
        "description": "Quick scale sweep over OFDM dimensions and antenna/user counts.",
        "command": "python scripts/sweep_sionna_ofdm_scale.py --quick --out outputs/comparisons/sionna_ofdm_scale_sweep",
        "quick": True,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_snr_ablation/snr_ablation.md",
        "description": "Quick train-SNR ablation for residual-RZF.",
        "command": "python scripts/sweep_sionna_train_snr.py --model sionna_ofdm_residual_rzf --quick --out outputs/comparisons/sionna_ofdm_snr_ablation",
        "quick": True,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_teacher_leakage_audit/leakage_audit.json",
        "description": "Teacher leakage audit for the WMMSE-distilled residual model.",
        "command": "python scripts/audit_sionna_teacher_leakage.py --config configs/sionna_ofdm_residual_wmmse_distill.yaml --ckpt outputs/runs/sionna_ofdm_residual_wmmse_distill/best.pt --out outputs/comparisons/sionna_ofdm_teacher_leakage_audit",
        "quick": False,
    },
    {
        "path": "outputs/comparisons/sionna_ofdm_wmmse_distill_sweep/distill_sweep.csv",
        "description": "Quick WMMSE distillation-weight sweep.",
        "command": "python scripts/sweep_sionna_wmmse_distill_weight.py --config configs/sionna_ofdm_residual_wmmse_distill.yaml --weights 0.0 0.05 0.1 0.5 1.0 --quick --out outputs/comparisons/sionna_ofdm_wmmse_distill_sweep",
        "quick": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_json_if_possible(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifact_row(item: dict[str, Any], commit: str) -> dict[str, Any]:
    path = Path(item["path"])
    row: dict[str, Any] = {
        "path": item["path"],
        "description": item["description"],
        "generating_command": item["command"],
        "generated_from_commit": commit,
        "exists": path.exists(),
        "quick": bool(item["quick"]),
        "teacher_used_during_training": False,
        "teacher_used_during_inference": False,
        "sionna_rt_used": False,
        "ray_tracing_used": False,
        "nr_full_stack_used": False,
    }
    if not path.exists():
        return row

    payload = _load_json_if_possible(path)
    if payload is not None:
        row["teacher_used_during_training"] = bool(payload.get("teacher_used_during_training", False))
        row["teacher_used_during_inference"] = bool(payload.get("teacher_used_during_inference", False))
    return row


def main() -> None:
    args = parse_args()
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_json.with_suffix(".md")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    rows = [_artifact_row(item, commit) for item in ARTIFACTS]
    payload = {
        "generated_from_commit": commit,
        "note": "Sionna learned-training artifacts are optional synthetic OFDM outputs only. No Sionna RT, no ray tracing, and no 5G NR full stack are used.",
        "artifacts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sionna Learned Training Artifact Manifest",
        "",
        f"- generated_from_commit: `{commit}`",
        "- note: optional synthetic-OFDM learned-training artifacts only.",
        "",
        "| path | exists | quick | teacher_train | teacher_infer | RT | ray tracing | 5G NR | command |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['path']} | {row['exists']} | {row['quick']} | {row['teacher_used_during_training']} | {row['teacher_used_during_inference']} | {row['sionna_rt_used']} | {row['ray_tracing_used']} | {row['nr_full_stack_used']} | `{row['generating_command']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved Sionna learned-training artifact manifest to {out_json}")


if __name__ == "__main__":
    main()
