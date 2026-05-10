#!/usr/bin/env python
"""Aggregate and compare baseline or model result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    combined_path = out_dir / f"{args.prefix}_results.csv"
    df.to_csv(combined_path, index=False)
    if {"snr_db", "se", "method"} <= set(df.columns):
        plt.figure(figsize=(6, 4))
        for method, group in df.groupby("method"):
            ordered = group.sort_values("snr_db")
            plt.plot(ordered["snr_db"], ordered["se"], marker="o", label=method)
        plt.xlabel("snr_db")
        plt.ylabel("se")
        plt.title("SE vs SNR")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{args.prefix}_se_vs_snr.png")
        plt.close()
    print(f"Saved combined results to {combined_path}")


if __name__ == "__main__":
    main()
