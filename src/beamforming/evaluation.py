"""Shared evaluation helpers for fair method comparison."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from beamforming.baselines.common import evaluate_baseline
from beamforming.data.dataset import ChannelDataset, split_dataset
from beamforming.metrics.sum_rate import multi_user_downlink_sum_rate, noise_variance_from_snr


def get_eval_subset(dataset: ChannelDataset, val_fraction: float, seed: int) -> Subset:
    """Return the deterministic evaluation subset used across train/eval/baselines."""
    _, val_subset = split_dataset(dataset, val_fraction=val_fraction, seed=seed)
    return val_subset


def evaluate_model_by_snr(
    model: torch.nn.Module,
    dataset: ChannelDataset | Subset,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate a learned model on the same channels grouped by SNR."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            channel = batch["channel"].to(device)
            channel_real = batch["channel_real"].to(device)
            snr_db = batch["snr_db"].to(device)
            start = time.perf_counter()
            outputs = model(channel_real, snr_db=snr_db, channel_complex=channel)
            runtime = time.perf_counter() - start
            noise_var = noise_variance_from_snr(snr_db).to(device)
            se = multi_user_downlink_sum_rate(channel, outputs["precoder"], noise_var)
            for sample_idx in range(channel.size(0)):
                rows.append(
                    {
                        "snr_db": float(snr_db[sample_idx].item()),
                        "se": float(se[sample_idx].item()),
                        "runtime_sec": runtime / channel.size(0),
                    }
                )
    df = pd.DataFrame(rows)
    return df.groupby("snr_db", as_index=False).agg(se=("se", "mean"), runtime_sec=("runtime_sec", "mean"))


def evaluate_baselines_by_snr(
    methods: list[str],
    dataset: ChannelDataset | Subset,
    num_rf_chains: int,
) -> pd.DataFrame:
    """Evaluate baseline methods on a common subset and SNR grid."""
    if isinstance(dataset, Subset):
        channels = dataset.dataset.channels[dataset.indices]
        snr_values = dataset.dataset.snr_db[dataset.indices]
    else:
        channels = dataset.channels
        snr_values = dataset.snr_db
    rows: list[dict[str, Any]] = []
    for snr_db in snr_values.unique(sorted=True).tolist():
        mask = snr_values == snr_db
        subset_channels = channels[mask]
        for method in methods:
            result = evaluate_baseline(method, subset_channels, float(snr_db), num_rf_chains=num_rf_chains)
            rows.append(
                {
                    "method": method,
                    "snr_db": float(snr_db),
                    "se": float(result["sum_rate"].mean().item()),
                    "runtime_sec": float(result["runtime_sec"]) / max(subset_channels.size(0), 1),
                }
            )
    return pd.DataFrame(rows)


def add_relative_gaps(df: pd.DataFrame, reference_method: str = "rzf") -> pd.DataFrame:
    """Compute relative gaps against RZF and best baseline for each SNR."""
    result = df.copy()
    ref = result[result["method"] == reference_method][["snr_db", "se"]].rename(columns={"se": "rzf_se"})
    result = result.merge(ref, on="snr_db", how="left")
    baseline_best = (
        result[result["method"].isin(["mrt", "zf", "rzf", "dft", "omp"])]
        .groupby("snr_db", as_index=False)["se"]
        .max()
        .rename(columns={"se": "best_baseline_se"})
    )
    result = result.merge(baseline_best, on="snr_db", how="left")
    result["relative_gap_to_rzf"] = (result["se"] - result["rzf_se"]) / result["rzf_se"].abs().clip(lower=1e-12)
    result["relative_gap_to_best_baseline"] = (
        (result["se"] - result["best_baseline_se"]) / result["best_baseline_se"].abs().clip(lower=1e-12)
    )
    for snr_value in (10.0, 15.0, 20.0):
        col = f"gap_{int(snr_value)}db"
        result[col] = result["relative_gap_to_rzf"].where(result["snr_db"] == snr_value)
    high_mask = result["snr_db"].isin([10.0, 15.0, 20.0])
    result["mean_gap_high_snr"] = result["relative_gap_to_rzf"].where(high_mask)
    return result


def save_comparison_outputs(df: pd.DataFrame, out_dir: str | Path, prefix: str) -> tuple[Path, Path]:
    """Save unified CSV and SE-vs-SNR plot."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{prefix}_all_methods.csv"
    fig_path = out_path / f"{prefix}_se_vs_snr.png"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(7, 4.5))
    for method, group in df.groupby("method"):
        ordered = group.sort_values("snr_db")
        plt.plot(ordered["snr_db"], ordered["se"], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
    plt.title("SE vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return csv_path, fig_path
