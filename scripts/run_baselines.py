#!/usr/bin/env python
"""Run classical beamforming baselines on a saved dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.baselines.common import evaluate_baseline
from beamforming.data.dataset import load_channel_dataset
from beamforming.data.deepmimo_loader import load_deepmimo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-rf-chains", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--bs-idx", type=int, default=0)
    parser.add_argument("--deepmimo-users", type=int, default=4)
    parser.add_argument("--subcarrier-idx", type=int, default=None)
    return parser.parse_args()


def _plot_metric(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    for method, group in df.groupby("method"):
        ordered = group.sort_values(x)
        plt.plot(ordered[x], ordered[y], marker="o", label=method)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_runtime_bar(df: pd.DataFrame, out_path: Path) -> None:
    summary = df.groupby("method", as_index=False)["runtime_sec"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(summary["method"], summary["runtime_sec"])
    plt.ylabel("runtime_sec")
    plt.title("Runtime Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_type == "deepmimo":
        dataset = load_deepmimo_dataset(
            args.data,
            bs_idx=args.bs_idx,
            num_users=args.deepmimo_users,
            subcarrier_idx=args.subcarrier_idx,
        )
    else:
        dataset = load_channel_dataset(args.data)
    channels = dataset.channels if args.max_samples <= 0 else dataset.channels[: args.max_samples]
    metadata = dataset.metadata
    snr_list = dataset.snr_db.unique(sorted=True).tolist()
    num_users = int(metadata.get("num_users", channels.shape[-2]))
    num_rf_chains = int(metadata.get("num_rf_chains", args.num_rf_chains))

    rows = []
    for snr_db in snr_list:
        subset = channels
        for method in args.methods:
            result = evaluate_baseline(method, subset, float(snr_db), num_rf_chains=num_rf_chains)
            rows.append(
                {
                    "method": method,
                    "snr_db": float(snr_db),
                    "num_users": num_users,
                    "num_rf_chains": num_rf_chains,
                    "sum_rate": float(result["sum_rate"].mean().item()),
                    "se": float(result["se"].mean().item()),
                    "runtime_sec": float(result["runtime_sec"]),
                }
            )

    rzf_users = []
    for users in range(1, min(8, channels.shape[-2]) + 1):
        subset = channels[:, :users, :]
        result = evaluate_baseline("rzf", subset, 10.0, num_rf_chains=min(users, num_rf_chains))
        rzf_users.append({"method": "rzf", "num_users": users, "sum_rate": float(result["sum_rate"].mean().item())})

    dft_rf = []
    for rf in range(1, min(8, channels.shape[-2]) + 1):
        result = evaluate_baseline("dft", channels, 10.0, num_rf_chains=rf)
        dft_rf.append({"method": "dft", "num_rf_chains": rf, "sum_rate": float(result["sum_rate"].mean().item())})

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    csv_path = metrics_dir / "baseline_results.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    _plot_metric(df, "snr_db", "se", out_dir / "se_vs_snr.png", "SE vs SNR")
    _plot_runtime_bar(df, out_dir / "runtime_comparison.png")
    _plot_metric(pd.DataFrame(rzf_users), "num_users", "sum_rate", out_dir / "sum_rate_vs_users.png", "Sum-Rate vs Users")
    _plot_metric(pd.DataFrame(dft_rf), "num_rf_chains", "sum_rate", out_dir / "sum_rate_vs_rf_chains.png", "Sum-Rate vs RF Chains")
    print(f"Saved baseline metrics to {csv_path}")


if __name__ == "__main__":
    main()
