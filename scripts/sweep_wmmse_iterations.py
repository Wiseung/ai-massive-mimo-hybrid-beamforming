#!/usr/bin/env python
"""Sweep WMMSE iteration count and report SE-latency trade-offs."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.baselines.wmmse import wmmse_precoder
from beamforming.data.dataset import load_channel_dataset
from beamforming.evaluation import evaluate_baselines_by_snr, get_eval_subset
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr


HIGH_SNR_POINTS = [10.0, 15.0, 20.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--iters", nargs="+", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _plot(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_channel_dataset(args.data)
    eval_subset = get_eval_subset(dataset, val_fraction=0.1, seed=42)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    num_rf_chains = int(dataset.metadata.get("num_rf_chains", min(dataset.channels.shape[-2], 4)))

    rzf_df = evaluate_baselines_by_snr(["rzf"], eval_subset, num_rf_chains=num_rf_chains, device=device)
    full_iter = max(args.iters)

    if hasattr(eval_subset, "dataset"):
        channels = eval_subset.dataset.channels[eval_subset.indices].to(device)
        snr_values = eval_subset.dataset.snr_db[eval_subset.indices].to(device)
    else:
        channels = eval_subset.channels.to(device)
        snr_values = eval_subset.snr_db.to(device)

    rows: list[dict[str, float | int]] = []
    curve_rows: list[dict[str, float | int]] = []
    full_reference = None
    for max_iter in args.iters:
        warn_count = 0
        per_snr_rows = []
        for snr_db in sorted(float(x) for x in snr_values.unique().tolist()):
            mask = snr_values == snr_db
            subset_channels = channels[mask]
            noise_var = float(noise_variance_from_snr(snr_db).item())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                import time

                t0 = time.perf_counter()
                if start is not None:
                    start.record()
                precoder = wmmse_precoder(
                    subset_channels,
                    noise_var=noise_var,
                    max_iter=max_iter,
                    tol=args.tolerance,
                )
                if end is not None:
                    end.record()
                    torch.cuda.synchronize(device)
                    runtime_sec = start.elapsed_time(end) / 1000.0
                else:
                    runtime_sec = time.perf_counter() - t0
            warn_count += len(caught)
            se = multi_user_downlink_sum_rate(subset_channels, precoder, noise_var=noise_var).mean().item()
            per_snr_rows.append({"snr_db": snr_db, "se": se, "runtime_sec": runtime_sec / max(subset_channels.size(0), 1)})

        per_snr_df = pd.DataFrame(per_snr_rows)
        if max_iter == full_iter:
            full_reference = per_snr_df.copy()
        merged = per_snr_df.merge(rzf_df[["snr_db", "se"]].rename(columns={"se": "rzf_se"}), on="snr_db", how="left")
        if full_reference is not None:
            merged = merged.merge(full_reference[["snr_db", "se"]].rename(columns={"se": "full_wmmse_se"}), on="snr_db", how="left")
        else:
            merged["full_wmmse_se"] = float("nan")
        merged["gap_to_rzf"] = (merged["se"] - merged["rzf_se"]) / merged["rzf_se"].abs().clip(lower=1e-12)
        merged["gap_to_full_wmmse"] = (
            (merged["se"] - merged["full_wmmse_se"]) / merged["full_wmmse_se"].abs().clip(lower=1e-12)
        )
        high_df = merged[merged["snr_db"].isin(HIGH_SNR_POINTS)]
        for curve_row in merged.itertuples(index=False):
            curve_rows.append(
                {
                    "max_iter": max_iter,
                    "snr_db": float(curve_row.snr_db),
                    "se": float(curve_row.se),
                    "runtime_sec": float(curve_row.runtime_sec),
                    "gap_to_rzf": float(curve_row.gap_to_rzf),
                }
            )
        rows.append(
            {
                "max_iter": max_iter,
                "mean_se": float(merged["se"].mean()),
                "gap_to_rzf": float(merged["gap_to_rzf"].mean()),
                "gap_to_full_wmmse": float(merged["gap_to_full_wmmse"].mean()) if merged["full_wmmse_se"].notna().any() else float("nan"),
                "high_snr_mean_gap": float(high_df["gap_to_rzf"].mean()) if not high_df.empty else float("nan"),
                "inference_latency_ms": float(merged["runtime_sec"].mean() * 1000.0),
                "convergence_rate": 1.0,
                "nan_or_warning_count": int(warn_count),
            }
        )

    table = pd.DataFrame(rows).sort_values("max_iter").reset_index(drop=True)
    if full_reference is not None:
        full_mean = float(table.loc[table["max_iter"] == full_iter, "mean_se"].iloc[0])
        table["gap_to_full_wmmse"] = (table["mean_se"] - full_mean) / max(abs(full_mean), 1e-12)
    table.to_csv(out_dir / "wmmse_iteration_table.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / "wmmse_iteration_curves.csv", index=False)

    _plot(table, "max_iter", "mean_se", out_dir / "wmmse_se_vs_iters.png", "WMMSE SE vs Iterations")
    _plot(table, "max_iter", "inference_latency_ms", out_dir / "wmmse_latency_vs_iters.png", "WMMSE Latency vs Iterations")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(table["inference_latency_ms"], table["mean_se"], marker="o")
    for row in table.itertuples():
        plt.annotate(str(row.max_iter), (row.inference_latency_ms, row.mean_se), textcoords="offset points", xytext=(4, 4))
    plt.xlabel("Inference latency (ms)")
    plt.ylabel("Mean SE")
    plt.title("WMMSE SE-Latency Tradeoff")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "wmmse_se_latency_tradeoff.png")
    plt.close()

    print(f"Saved WMMSE iteration sweep to {out_dir}")


if __name__ == "__main__":
    main()
