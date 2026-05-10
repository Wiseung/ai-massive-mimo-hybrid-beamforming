#!/usr/bin/env python
"""Run classical beamforming baselines on a saved dataset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset
from beamforming.data.deepmimo_loader import load_deepmimo_dataset
from beamforming.evaluation import evaluate_baselines_by_snr, save_comparison_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-rf-chains", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dataset-type", choices=["auto", "tensor", "deepmimo"], default="auto")
    parser.add_argument("--bs-idx", type=int, default=0)
    parser.add_argument("--deepmimo-users", type=int, default=4)
    parser.add_argument("--subcarrier-idx", type=int, default=None)
    parser.add_argument("--num-subcarriers", type=int, default=None)
    parser.add_argument("--narrowband", action="store_true")
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
    if args.dataset_type == "deepmimo" or args.scenario is not None:
        dataset = load_deepmimo_dataset(
            scenario_path=args.data,
            scenario=args.scenario,
            download=args.download,
            bs_idx=args.bs_idx,
            num_users=args.deepmimo_users,
            subcarrier_idx=args.subcarrier_idx,
            num_subcarriers=args.num_subcarriers,
            narrowband=args.narrowband or args.num_subcarriers in (None, 1),
        )
    else:
        dataset = load_channel_dataset(args.data)
    if args.max_samples > 0:
        dataset.channels = dataset.channels[: args.max_samples]
        dataset.snr_db = dataset.snr_db[: args.max_samples]

    metadata = dataset.metadata
    num_users = int(metadata.get("num_users", dataset.channels.shape[-2]))
    num_rf_chains = int(metadata.get("num_rf_chains", args.num_rf_chains))

    df = evaluate_baselines_by_snr(args.methods, dataset, num_rf_chains=num_rf_chains)
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    csv_path = metrics_dir / "baseline_results.csv"
    df.assign(num_users=num_users, num_rf_chains=num_rf_chains).to_csv(csv_path, index=False)

    prefix = "deepmimo" if args.dataset_type == "deepmimo" or args.scenario is not None else "synthetic"
    save_comparison_outputs(df, out_dir / "figures", prefix=prefix)
    _plot_runtime_bar(df, figures_dir / "runtime_comparison.png")

    if dataset.channels.ndim == 3:
        from beamforming.evaluation import evaluate_baselines_by_snr as _eval
        rzf_users = []
        for users in range(1, min(8, dataset.channels.shape[-2]) + 1):
            subset = copy.deepcopy(dataset)
            subset.channels = dataset.channels[:, :users, :]
            subset.metadata = {**dataset.metadata, "num_users": users}
            result = _eval(["rzf"], subset, num_rf_chains=min(users, num_rf_chains))
            rzf_users.append({"method": "rzf", "num_users": users, "sum_rate": float(result["se"].mean())})
        dft_rf = []
        for rf in range(1, min(8, dataset.channels.shape[-2]) + 1):
            result = _eval(["dft"], dataset, num_rf_chains=rf)
            dft_rf.append({"method": "dft", "num_rf_chains": rf, "sum_rate": float(result["se"].mean())})
        _plot_metric(pd.DataFrame(rzf_users), "num_users", "sum_rate", figures_dir / "sum_rate_vs_users.png", "Sum-Rate vs Users")
        _plot_metric(pd.DataFrame(dft_rf), "num_rf_chains", "sum_rate", figures_dir / "sum_rate_vs_rf_chains.png", "Sum-Rate vs RF Chains")
    print(f"Saved baseline metrics to {csv_path}")


if __name__ == "__main__":
    main()
