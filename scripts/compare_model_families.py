#!/usr/bin/env python
"""Compare baseline and learned beamformer families from saved outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keep = [col for col in ["method", "snr_db", "se", "runtime_sec"] if col in df.columns]
    return df[keep].copy()


def _summary_yaml(path: Path) -> dict | None:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    unfolded_csv = Path("outputs/comparisons/synthetic_unfolded_rzf/synthetic_all_methods.csv")
    residual_csv = Path("outputs/comparisons/synthetic_residual_rzf/synthetic_all_methods.csv")
    cnn_csv = Path("outputs/comparisons/synthetic_cnn_finetune/synthetic_all_methods.csv")

    frames = [df for df in [_load_if_exists(unfolded_csv), _load_if_exists(residual_csv), _load_if_exists(cnn_csv)] if df is not None]
    if not frames:
        raise FileNotFoundError("No comparison CSVs found for model family comparison.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["method", "snr_db"], keep="last")

    base_df = _load_if_exists(Path("outputs/runs/baselines_synthetic_wmmse/metrics/baseline_results.csv"))
    if base_df is not None:
        base_df = base_df.rename(columns={"sum_rate": "se"})
        base_subset = base_df[["method", "snr_db", "se", "runtime_sec"]].drop_duplicates(subset=["method", "snr_db"], keep="last")
        combined = pd.concat([combined, base_subset], ignore_index=True)
        combined = combined.drop_duplicates(subset=["method", "snr_db"], keep="last")
    if "rzf" not in set(combined["method"].unique()):
        raise FileNotFoundError("RZF rows are not available in the loaded comparison inputs.")

    strongest = combined.groupby("snr_db", as_index=False)["se"].max().rename(columns={"se": "best_ref"})
    rzf = combined[combined["method"] == "rzf"][["snr_db", "se"]].rename(columns={"se": "rzf_se"})
    wmmse = combined[combined["method"] == "wmmse"][["snr_db", "se"]].rename(columns={"se": "wmmse_se"})
    merged = combined.merge(strongest, on="snr_db", how="left").merge(rzf, on="snr_db", how="left").merge(wmmse, on="snr_db", how="left")
    merged["gap_to_rzf"] = (merged["se"] - merged["rzf_se"]) / merged["rzf_se"].abs().clip(lower=1e-12)
    merged["gap_to_wmmse"] = (merged["se"] - merged["wmmse_se"]) / merged["wmmse_se"].abs().clip(lower=1e-12)
    merged["gap_to_best_reference"] = (merged["se"] - merged["best_ref"]) / merged["best_ref"].abs().clip(lower=1e-12)

    train_summaries = {
        "cnn": _summary_yaml(Path("outputs/runs/cnn_finetune_rzf/train_summary.yaml")),
        "residual_rzf": _summary_yaml(Path("outputs/runs/synthetic_residual_rzf/train_summary.yaml")),
        "unfolded_rzf": _summary_yaml(Path("outputs/runs/synthetic_unfolded_rzf/train_summary.yaml")),
    }
    param_counts = {
        "cnn": None,
        "residual_rzf": None,
        "unfolded_rzf": None,
    }

    rows = []
    for method, group in merged.groupby("method"):
        rows.append(
            {
                "method": method,
                "mean_se": float(group["se"].mean()),
                "gap_to_rzf": float(group["gap_to_rzf"].mean()) if group["gap_to_rzf"].notna().any() else None,
                "gap_to_wmmse": float(group["gap_to_wmmse"].mean()) if group["gap_to_wmmse"].notna().any() else None,
                "gap_to_best_reference": float(group["gap_to_best_reference"].mean()),
                "high_snr_mean_gap": float(group[group["snr_db"].isin([10.0, 15.0, 20.0])]["gap_to_rzf"].mean()),
                "inference_latency_ms": float(group["runtime_sec"].mean() * 1000.0),
                "train_time": None if train_summaries.get(method) is None else train_summaries[method].get("train_time_sec"),
                "num_params": param_counts.get(method),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_se", ascending=False)
    table.to_csv(out_dir / "model_family_table.csv", index=False)

    plt.figure(figsize=(7.2, 4.6))
    for method, group in merged.groupby("method"):
        ordered = group.sort_values("snr_db")
        plt.plot(ordered["snr_db"], ordered["se"], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
    plt.title("Model Family SE vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_family_se_vs_snr.png")
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    runtime_table = table[["method", "inference_latency_ms"]].sort_values("inference_latency_ms")
    plt.bar(runtime_table["method"], runtime_table["inference_latency_ms"])
    plt.ylabel("Inference latency (ms)")
    plt.title("Model Family Runtime")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "model_family_runtime.png")
    plt.close()
    print(f"Saved model family comparison to {out_dir}")


if __name__ == "__main__":
    main()
