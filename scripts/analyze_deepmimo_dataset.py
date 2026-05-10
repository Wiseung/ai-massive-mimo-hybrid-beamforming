#!/usr/bin/env python
"""Analyze a saved DeepMIMO tensor and export dataset diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset
from beamforming.data.splits import make_dataset_split, split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="Output stem, e.g. outputs/reports/deepmimo_dataset_summary")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _snr_distribution(snr_db: torch.Tensor) -> dict[str, int]:
    return {str(float(v.item())): int((snr_db == v).sum().item()) for v in torch.unique(snr_db, sorted=True)}


def _split_summary(num_samples: int, seed: int, mode: str) -> dict[str, object]:
    payload = make_dataset_split(num_samples, split_mode=mode, seed=seed)
    return {
        "mode": mode,
        "seed": seed,
        "counts": split_counts(payload),
    }


def main() -> None:
    args = parse_args()
    dataset = load_channel_dataset(args.data)
    channels = dataset.channels
    metadata = dataset.metadata
    powers = (torch.abs(channels) ** 2).sum(dim=tuple(range(1, channels.ndim)))
    norms = torch.sqrt(powers.clamp_min(1e-12))
    low_power_threshold = 1e-6

    raw_shape = metadata.get("raw_channel_shape")
    grouped_before = None
    if raw_shape is not None and len(raw_shape) > 0:
        grouped_before = int(raw_shape[0]) // int(metadata.get("num_users", channels.shape[-2]))
    invalid_ratio = float(metadata.get("invalid_group_ratio", 0.0))
    filtered_out = int(round(grouped_before * invalid_ratio)) if grouped_before is not None else None

    summary = {
        "dataset_path": args.data,
        "num_samples": int(len(dataset)),
        "num_users": int(metadata.get("num_users", channels.shape[-2])),
        "num_bs_ant": int(metadata.get("num_bs_ant", channels.shape[-1])),
        "num_subcarriers": int(metadata.get("num_subcarriers", 1)),
        "raw_channel_shape": raw_shape,
        "channel_norm_distribution": {
            "mean": float(norms.mean().item()),
            "std": float(norms.std(unbiased=False).item()),
            "min": float(norms.min().item()),
            "p10": float(torch.quantile(norms, 0.10).item()),
            "median": float(torch.quantile(norms, 0.50).item()),
            "p90": float(torch.quantile(norms, 0.90).item()),
            "max": float(norms.max().item()),
        },
        "power_distribution": {
            "mean": float(powers.mean().item()),
            "std": float(powers.std(unbiased=False).item()),
            "zero_power_ratio": float((powers <= 1e-12).float().mean().item()),
            "low_power_ratio": float((powers <= low_power_threshold).float().mean().item()),
        },
        "filtering": {
            "group_count_before_filter": grouped_before,
            "group_count_after_filter": int(len(dataset)),
            "filtered_group_count": filtered_out,
            "invalid_group_ratio": invalid_ratio,
        },
        "user_group_construction": (
            "User groups are formed by contiguous blocks of K users in the DeepMIMO receiver ordering after selecting one BS, "
            "then zero-power groups are filtered out before saving the project tensor."
        ),
        "snr_distribution": _snr_distribution(dataset.snr_db),
        "split_summaries": {
            "random_seed42": _split_summary(len(dataset), args.seed, "random"),
            "contiguous_seed42": _split_summary(len(dataset), args.seed, "contiguous"),
        },
    }

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    hist_path = figures_dir / "deepmimo_channel_norm_hist.png"
    power_path = figures_dir / "deepmimo_user_group_power.png"

    plt.figure(figsize=(6.5, 4.2))
    plt.hist(norms.cpu().numpy(), bins=50, color="#2a6fbb", alpha=0.85)
    plt.xlabel("Channel Frobenius Norm")
    plt.ylabel("Count")
    plt.title("DeepMIMO Channel Norm Distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(powers.cpu().numpy(), linewidth=0.8, alpha=0.8)
    plt.xlabel("User Group Index")
    plt.ylabel("Group Power")
    plt.title("DeepMIMO User Group Power After Filtering")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(power_path)
    plt.close()

    print(f"Saved dataset summary to {json_path}")
    print(f"Saved histogram to {hist_path}")
    print(f"Saved power trace to {power_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
