#!/usr/bin/env python
"""Aggregate learned-beamformer ablations into one table and SE-vs-SNR plot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


DEFAULT_RUNS: list[dict[str, str]] = [
    {
        "label": "original_cnn",
        "summary": "outputs/runs/cnn_synthetic_eval_fair/summary.yaml",
        "curve": "outputs/runs/cnn_synthetic_eval_fair/evaluation_by_snr.csv",
        "train_summary": "outputs/runs/cnn_synthetic/train_summary.yaml",
    },
    {
        "label": "rzf_warm_start",
        "summary": "outputs/runs/cnn_finetune_rzf_no_snr_eval/summary.yaml",
        "curve": "outputs/runs/cnn_finetune_rzf_no_snr_eval/evaluation_by_snr.csv",
        "train_summary": "outputs/runs/cnn_finetune_rzf_no_snr/train_summary.yaml",
    },
    {
        "label": "rzf_warm_start_snr_conditioned",
        "summary": "outputs/runs/cnn_finetune_rzf_eval/summary.yaml",
        "curve": "outputs/runs/cnn_finetune_rzf_eval/evaluation_by_snr.csv",
        "train_summary": "outputs/runs/cnn_finetune_rzf/train_summary.yaml",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Comparison output directories from evaluate_all.py")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _find_csv(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*_all_methods.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_all_methods.csv found in {run_dir}")
    return matches[0]


def _curve_from_summary(summary: dict[str, Any]) -> pd.DataFrame:
    se_by_snr = summary.get("se_by_snr", {})
    rows = [
        {"snr_db": float(snr), "se": float(value)}
        for snr, value in se_by_snr.items()
    ]
    if not rows:
        raise ValueError("summary.yaml does not contain se_by_snr")
    return pd.DataFrame(rows).sort_values("snr_db")


def _load_default_entry(entry: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    summary_path = Path(entry["summary"])
    curve_path = Path(entry["curve"])
    train_summary_path = Path(entry["train_summary"])
    summary = _read_yaml(summary_path)
    train_summary = _read_yaml(train_summary_path) if train_summary_path.exists() else {}
    if curve_path.exists():
        curve_df = pd.read_csv(curve_path)
        if "method" in curve_df.columns:
            curve_df = curve_df[curve_df["method"] == "cnn"].copy()
        if "snr_db" not in curve_df.columns or "se" not in curve_df.columns or curve_df.empty:
            curve_df = _curve_from_summary(summary)
    else:
        curve_df = _curve_from_summary(summary)
    return curve_df.sort_values("snr_db"), summary, train_summary


def _load_comparison_run(run_dir: Path) -> tuple[str, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    csv_path = _find_csv(run_dir)
    summary_path = run_dir / "summary.yaml"
    df = pd.read_csv(csv_path)
    cnn_df = df[df["method"] == "cnn"].copy().sort_values("snr_db")
    if cnn_df.empty:
        raise ValueError(f"No cnn rows found in {csv_path}")
    summary = _read_yaml(summary_path) if summary_path.exists() else {}
    train_dir_map = {
        "synthetic_cnn_finetune": Path("outputs/runs/cnn_finetune_rzf"),
        "synthetic_cnn_highsnr": Path("outputs/runs/cnn_finetune_highsnr"),
        "synthetic_cnn_mixed_teacher": Path("outputs/runs/cnn_finetune_mixed_teacher"),
        "deepmimo_cnn_finetune": Path("outputs/runs/deepmimo_cnn_finetune"),
    }
    train_dir = train_dir_map.get(run_dir.name)
    train_summary = {}
    if train_dir is not None:
        summary_candidates = [train_dir / "train_summary.yaml", train_dir / "pretrain_summary.yaml"]
        for candidate in summary_candidates:
            if candidate.exists():
                train_summary = _read_yaml(candidate)
                break
    return run_dir.name, cnn_df, summary, train_summary


def _make_row(
    label: str,
    curve_df: pd.DataFrame,
    summary: dict[str, Any],
    train_summary: dict[str, Any],
) -> dict[str, Any]:
    curve_df = curve_df.sort_values("snr_db")
    snr_to_se = {float(row.snr_db): float(row.se) for row in curve_df.itertuples()}
    rzf_reference = {
        -10.0: 0.164250,
        -5.0: 0.505333,
        0.0: 1.331687,
        5.0: 3.102838,
        10.0: 6.302627,
        15.0: 11.158300,
        20.0: 16.474419,
    }
    mean_se = float(summary.get("mean_se", curve_df["se"].mean()))
    mean_gap_to_rzf = float(
        summary.get(
            "mean_gap_to_rzf",
            summary.get("mean_relative_gap_to_rzf", float("nan")),
        )
    )
    mean_gap_to_best = float(
        summary.get(
            "mean_gap_to_best_baseline",
            summary.get("mean_relative_gap_to_best_baseline", float("nan")),
        )
    )
    def gap_at(snr: float, key: str) -> float:
        if key in summary:
            return float(summary[key])
        se = snr_to_se.get(snr)
        ref = rzf_reference.get(snr)
        if se is None or ref is None:
            return float("nan")
        return float((se - ref) / max(abs(ref), 1e-12))

    gap_10 = gap_at(10.0, "gap_10db")
    gap_15 = gap_at(15.0, "gap_15db")
    gap_20 = gap_at(20.0, "gap_20db")
    high_snr_mean_gap = float(
        summary.get(
            "mean_gap_high_snr",
            pd.Series([gap_10, gap_15, gap_20], dtype=float).mean(),
        )
    )
    return {
        "run": label,
        "mean_se": mean_se,
        "mean_gap_to_rzf": mean_gap_to_rzf,
        "mean_gap_to_best_baseline": mean_gap_to_best,
        "gap_10db": gap_10,
        "gap_15db": gap_15,
        "gap_20db": gap_20,
        "high_snr_mean_gap": high_snr_mean_gap,
        "best_epoch": train_summary.get("best_epoch"),
        "train_time": train_summary.get("train_time_sec"),
        "se_-10db": snr_to_se.get(-10.0),
        "se_-5db": snr_to_se.get(-5.0),
        "se_0db": snr_to_se.get(0.0),
        "se_5db": snr_to_se.get(5.0),
        "se_10db": snr_to_se.get(10.0),
        "se_15db": snr_to_se.get(15.0),
        "se_20db": snr_to_se.get(20.0),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    plot_entries: list[tuple[str, pd.DataFrame]] = []

    for entry in DEFAULT_RUNS:
        curve_df, summary, train_summary = _load_default_entry(entry)
        rows.append(_make_row(entry["label"], curve_df, summary, train_summary))
        plot_entries.append((entry["label"], curve_df))

    for run in args.runs:
        label, curve_df, summary, train_summary = _load_comparison_run(Path(run))
        if label == "synthetic_cnn_finetune":
            continue
        rows.append(_make_row(label, curve_df, summary, train_summary))
        plot_entries.append((label, curve_df))

    ablation_df = pd.DataFrame(rows)
    order = [
        "original_cnn",
        "rzf_warm_start",
        "rzf_warm_start_snr_conditioned",
        "synthetic_cnn_highsnr",
        "synthetic_cnn_mixed_teacher",
    ]
    ablation_df["run"] = pd.Categorical(ablation_df["run"], categories=order, ordered=True)
    ablation_df = ablation_df.sort_values("run").reset_index(drop=True)
    ablation_df.to_csv(out_dir / "ablation_table.csv", index=False)

    plt.figure(figsize=(7.5, 4.8))
    for label, curve_df in plot_entries:
        ordered = curve_df.sort_values("snr_db")
        plt.plot(ordered["snr_db"], ordered["se"], marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
    plt.title("CNN Ablation: SE vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_se_vs_snr.png")
    plt.close()

    print(f"Saved ablation table to {out_dir / 'ablation_table.csv'}")
    print(f"Saved ablation plot to {out_dir / 'ablation_se_vs_snr.png'}")


if __name__ == "__main__":
    main()
