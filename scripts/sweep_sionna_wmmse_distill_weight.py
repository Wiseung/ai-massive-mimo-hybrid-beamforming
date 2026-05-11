#!/usr/bin/env python
"""Sweep distillation weight for the optional WMMSE-distilled residual OFDM model."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_ofdm_experiment_helpers import apply_quick_overrides, dump_yaml, load_json, load_yaml, update_nested_dict, run_python_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_root = out_dir / "configs"
    run_root = out_dir / "runs"
    eval_root = out_dir / "evals"
    cfg_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(args.config)
    rows: list[dict[str, object]] = []
    best_eval_dir: Path | None = None
    best_gap = float("-inf")

    for weight in args.weights:
        cfg = update_nested_dict(base_cfg, {"loss": {"distill_weight": float(weight), "teacher_iter": 5, "rate_weight": 1.0}})
        if args.quick:
            cfg = apply_quick_overrides(cfg)
        tag = str(weight).replace(".", "p")
        cfg_path = cfg_root / f"distill_{tag}.yaml"
        dump_yaml(cfg, cfg_path)
        run_dir = run_root / f"distill_{tag}"
        eval_dir = eval_root / f"distill_{tag}"
        run_python_command(
            [
                "scripts/train_sionna_ofdm_beamformer.py",
                "--config",
                str(cfg_path),
                "--out",
                str(run_dir),
                *(["--smoke"] if args.quick else []),
            ],
            cwd=repo_root,
        )
        run_python_command(
            [
                "scripts/evaluate_sionna_ofdm_beamformer.py",
                "--config",
                str(cfg_path),
                "--ckpt",
                str(run_dir / "best.pt"),
                "--out",
                str(eval_dir),
            ],
            cwd=repo_root,
        )
        train_summary = load_json(run_dir / ("smoke_summary.json" if args.quick else "summary.json"))
        eval_summary = load_json(eval_dir / "summary.json")
        rows.append(
            {
                "distill_weight": float(weight),
                "quick": bool(args.quick),
                "learned_mean_sum_rate": float(eval_summary["learned_mean_sum_rate"]),
                "mean_gap_to_rzf": float(eval_summary["learned_mean_gap_to_rzf"]),
                "mean_gap_to_wmmse_iter_5": float(eval_summary["learned_mean_gap_to_wmmse_iter_5"]),
                "high_snr_gap_to_wmmse_iter_5": float(eval_summary["high_snr_gap_to_wmmse_iter_5"]),
                "best_val_loss": float(train_summary["best_val_loss"]),
                "best_val_distill_loss": float(train_summary.get("best_val_distill_loss", 0.0) or 0.0),
            }
        )
        if float(eval_summary["learned_mean_gap_to_wmmse_iter_5"]) > best_gap:
            best_gap = float(eval_summary["learned_mean_gap_to_wmmse_iter_5"])
            best_eval_dir = eval_dir

    frame = pd.DataFrame(rows).sort_values("distill_weight")
    frame.to_csv(out_dir / "distill_sweep.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(frame["distill_weight"], frame["mean_gap_to_wmmse_iter_5"], marker="o")
    plt.xlabel("Distill weight")
    plt.ylabel("Gap to WMMSE-iter5")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "gap_to_wmmse_iter5_vs_weight.png", dpi=160)
    plt.close()

    if best_eval_dir is not None:
        metrics = pd.read_csv(best_eval_dir / "metrics.csv")
        learned = metrics[metrics["method"] == "sionna_ofdm_residual_wmmse_distill"].copy()
        plt.figure(figsize=(7, 4.5))
        plt.plot(learned["snr_db"], learned["mean_sum_rate"], marker="o")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean sum-rate (bit/s/Hz)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "se_vs_snr_best_weight.png", dpi=160)
        plt.close()

    best_row = frame.sort_values("mean_gap_to_wmmse_iter_5", ascending=False).iloc[0]
    lines = [
        "# WMMSE Distillation Weight Sweep",
        "",
        f"- quick_mode: `{bool(args.quick)}`",
        f"- recommended_distill_weight: `{best_row['distill_weight']}`",
        f"- best mean_gap_to_wmmse_iter_5: `{best_row['mean_gap_to_wmmse_iter_5']:+.6%}`",
        "",
        "## Answers",
        "",
        f"1. distill_weight improves gap_to_wmmse_iter5: `{frame['mean_gap_to_wmmse_iter_5'].max() > frame['mean_gap_to_wmmse_iter_5'].min()}`",
        f"2. stronger distillation clearly harms validation loss: `{frame.sort_values('distill_weight')['best_val_loss'].iloc[-1] > frame.sort_values('distill_weight')['best_val_loss'].iloc[0]}`",
        f"3. teacher matching without SE gain is visible: `{(frame['best_val_distill_loss'].idxmin() != frame['learned_mean_sum_rate'].idxmax()) if len(frame) > 1 else False}`",
        f"4. recommended weight: `{best_row['distill_weight']}`",
    ]
    (out_dir / "distill_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved distillation sweep outputs to {out_dir}")


if __name__ == "__main__":
    main()
