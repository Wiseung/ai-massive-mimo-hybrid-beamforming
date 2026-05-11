#!/usr/bin/env python
"""Run optional Sionna OFDM learned beamformer training across multiple seeds."""

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
    override_seed,
    run_python_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _load_metrics(eval_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(eval_dir / "metrics.csv")
    if "model_name" not in frame.columns:
        aliases = {
            "tiny_neural_beamformer": ["tiny_neural_beamformer", "learned"],
            "sionna_ofdm_residual_rzf": ["sionna_ofdm_residual_rzf"],
            "sionna_ofdm_unfolded_lite": ["sionna_ofdm_unfolded_lite"],
        }
        for canonical, names in aliases.items():
            matched = frame[frame["method"].isin(names)].copy()
            if not matched.empty:
                matched["method"] = canonical
                return matched
    return frame


def _save_plot(frame: pd.DataFrame, y: str, out_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for model, group in frame.groupby("model"):
        ordered = group.sort_values("snr_db")
        plt.plot(ordered["snr_db"], ordered[f"{y}_mean"], marker="o", label=model)
        plt.fill_between(
            ordered["snr_db"],
            ordered[f"{y}_mean"] - ordered[f"{y}_std"],
            ordered[f"{y}_mean"] + ordered[f"{y}_std"],
            alpha=0.2,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    evals_dir = out_dir / "evals"
    configs_dir = out_dir / "configs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    snr_rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        base_cfg = load_yaml(config_path)
        model_name = str(base_cfg["model"]["name"])
        for seed in args.seeds:
            cfg = override_seed(base_cfg, seed)
            if args.quick:
                cfg = apply_quick_overrides(cfg)
            cfg_path = configs_dir / f"{model_name}_seed{seed}.yaml"
            dump_yaml(cfg, cfg_path)

            run_dir = runs_dir / f"{model_name}_seed{seed}"
            eval_dir = evals_dir / f"{model_name}_seed{seed}"
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
            metrics = _load_metrics(eval_dir)
            learned = metrics[metrics["method"] == model_name].copy()
            learned["model"] = model_name
            learned["seed"] = seed
            snr_rows.extend(learned.to_dict(orient="records"))

            seed_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "quick": bool(args.quick),
                    "mean_sum_rate": float(eval_summary["learned_mean_sum_rate"]),
                    "gap_to_rzf": float(eval_summary["learned_mean_gap_to_rzf"]),
                    "gap_to_wmmse_iter_5": float(eval_summary["learned_mean_gap_to_wmmse_iter_5"]),
                    "high_snr_gap_to_rzf": float(eval_summary["high_snr_gap_to_rzf"]),
                    "high_snr_gap_to_wmmse_iter_5": float(eval_summary["high_snr_gap_to_wmmse_iter_5"]),
                    "train_time_sec": float(train_summary["train_time_sec"]),
                    "used_sionna_ofdm": bool(eval_summary["used_sionna_ofdm"]),
                    "used_sionna_channel": bool(eval_summary["used_sionna_channel"]),
                    "fallback_used": bool(eval_summary["fallback_used"]),
                }
            )

    seed_frame = pd.DataFrame(seed_rows)
    snr_frame = pd.DataFrame(snr_rows)
    summary = seed_frame.groupby("model", as_index=False).agg(
        num_seeds=("seed", "nunique"),
        mean_sum_rate_mean=("mean_sum_rate", "mean"),
        mean_sum_rate_std=("mean_sum_rate", "std"),
        gap_to_rzf_mean=("gap_to_rzf", "mean"),
        gap_to_rzf_std=("gap_to_rzf", "std"),
        gap_to_wmmse_iter_5_mean=("gap_to_wmmse_iter_5", "mean"),
        gap_to_wmmse_iter_5_std=("gap_to_wmmse_iter_5", "std"),
        high_snr_gap_to_rzf_mean=("high_snr_gap_to_rzf", "mean"),
        high_snr_gap_to_rzf_std=("high_snr_gap_to_rzf", "std"),
        train_time_sec_mean=("train_time_sec", "mean"),
        train_time_sec_std=("train_time_sec", "std"),
    ).fillna(0.0)
    summary.to_csv(out_dir / "multiseed_summary.csv", index=False)

    high_snr = snr_frame[snr_frame["snr_db"].isin([10.0, 15.0, 20.0])].groupby(["model", "seed"], as_index=False).agg(
        mean_high_snr_gap_to_rzf=("gap_to_rzf", "mean"),
        mean_high_snr_gap_to_wmmse_iter_5=("gap_to_wmmse_iter_5", "mean"),
        mean_high_snr_sum_rate=("mean_sum_rate", "mean"),
    )
    high_snr.to_csv(out_dir / "high_snr_gap_summary.csv", index=False)

    per_snr = snr_frame.groupby(["model", "snr_db"], as_index=False).agg(
        mean_sum_rate_mean=("mean_sum_rate", "mean"),
        mean_sum_rate_std=("mean_sum_rate", "std"),
        gap_to_rzf_mean=("gap_to_rzf", "mean"),
        gap_to_rzf_std=("gap_to_rzf", "std"),
    ).fillna(0.0)
    _save_plot(per_snr, "mean_sum_rate", out_dir / "se_vs_snr_mean_std.png", "Mean sum-rate (bit/s/Hz)")
    _save_plot(per_snr, "gap_to_rzf", out_dir / "gap_to_rzf_mean_std.png", "Gap to RZF")

    best_row = summary.sort_values("mean_sum_rate_mean", ascending=False).iloc[0]
    lines = [
        "# Sionna OFDM Multi-seed Benchmark",
        "",
        f"- quick_mode: `{bool(args.quick)}`",
        f"- seeds: `{args.seeds}`",
        f"- best mean model: `{best_row['model']}`",
        f"- best mean_sum_rate_mean: `{best_row['mean_sum_rate_mean']:.6f}`",
        f"- best gap_to_rzf_mean: `{best_row['gap_to_rzf_mean']:+.6%}`",
        f"- best gap_to_wmmse_iter_5_mean: `{best_row['gap_to_wmmse_iter_5_mean']:+.6%}`",
        "",
        "## Notes",
        "",
        "- This benchmark remains optional and synthetic-OFDM only.",
        "- Quick mode is explicitly marked and must not be interpreted as a full multi-seed study.",
        "- Learned models still do not claim a fair win over WMMSE-iter5 unless the measured gap is positive.",
    ]
    (out_dir / "multiseed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved multiseed outputs to {out_dir}")


if __name__ == "__main__":
    main()
