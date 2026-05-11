#!/usr/bin/env python
"""Train-SNR ablation for the optional Sionna OFDM residual-RZF beamformer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_ofdm_experiment_helpers import (
    apply_quick_overrides,
    dump_yaml,
    load_json,
    load_yaml,
    update_nested_dict,
    run_python_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != "sionna_ofdm_residual_rzf":
        raise SystemExit("Only sionna_ofdm_residual_rzf is supported in this ablation stage.")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_root = out_dir / "configs"
    run_root = out_dir / "runs"
    eval_root = out_dir / "evals"
    cfg_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    snr_sets = {
        "low_mid": [0, 5, 10],
        "high_only": [15, 20],
        "mixed_default": [0, 5, 10, 15, 20],
        "wide": [-5, 0, 5, 10, 15, 20],
    }
    base_cfg = load_yaml("configs/sionna_ofdm_residual_rzf.yaml")
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for name, snr_list in snr_sets.items():
        cfg = update_nested_dict(base_cfg, {"dataset": {"snr_db_train": snr_list}})
        if args.quick:
            cfg = apply_quick_overrides(cfg)
        cfg_path = cfg_root / f"{name}.yaml"
        dump_yaml(cfg, cfg_path)
        run_dir = run_root / name
        eval_dir = eval_root / name
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
        summary = load_json(eval_dir / "summary.json")
        metrics = pd.read_csv(eval_dir / "metrics.csv")
        learned = metrics[metrics["method"] == "sionna_ofdm_residual_rzf"].copy()
        learned["train_snr_set"] = name
        rows.extend(learned.to_dict(orient="records"))
        summary_rows.append(
            {
                "train_snr_set": name,
                "quick": bool(args.quick),
                "training_snr_list": ",".join(str(x) for x in snr_list),
                "learned_mean_sum_rate": float(summary["learned_mean_sum_rate"]),
                "mean_gap_to_rzf": float(summary["learned_mean_gap_to_rzf"]),
                "mean_gap_to_wmmse_iter_5": float(summary["learned_mean_gap_to_wmmse_iter_5"]),
                "high_snr_gap_to_rzf": float(summary["high_snr_gap_to_rzf"]),
                "high_snr_gap_to_wmmse_iter_5": float(summary["high_snr_gap_to_wmmse_iter_5"]),
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out_dir / "snr_ablation.csv", index=False)
    metrics_frame = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4.5))
    for name, group in metrics_frame.groupby("train_snr_set"):
        plt.plot(group["snr_db"], group["gap_to_rzf"], marker="o", label=name)
    plt.xlabel("Eval SNR (dB)")
    plt.ylabel("Gap to RZF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gap_vs_eval_snr.png", dpi=160)
    plt.close()

    best = summary_frame.sort_values("mean_gap_to_rzf", ascending=False).iloc[0]
    lines = [
        "# Sionna OFDM Train-SNR Ablation",
        "",
        f"- quick_mode: `{bool(args.quick)}`",
        f"- best mean gap to RZF set: `{best['train_snr_set']}` with `{best['mean_gap_to_rzf']:+.6%}`",
        f"- best high-SNR gap to RZF set: `{summary_frame.sort_values('high_snr_gap_to_rzf', ascending=False).iloc[0]['train_snr_set']}`",
        "",
        "## Answers",
        "",
        f"- high-SNR gap depends on train SNR set: `{summary_frame['high_snr_gap_to_rzf'].max() - summary_frame['high_snr_gap_to_rzf'].min() > 0.01}`",
        f"- mixed training is best by mean gap to RZF: `{best['train_snr_set'] == 'mixed_default'}`",
        "- A curriculum decision is not justified yet from this quick ablation alone; it needs a larger run if the SNR-set sensitivity is material.",
    ]
    (out_dir / "snr_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved SNR ablation outputs to {out_dir}")


if __name__ == "__main__":
    main()
