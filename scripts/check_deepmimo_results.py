#!/usr/bin/env python
"""Check DeepMIMO benchmark consistency and compare random vs contiguous results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", required=True)
    parser.add_argument("--contiguous", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "method",
        "num_seeds",
        "mean_se_mean",
        "mean_se_std",
        "inference_latency_ms_mean",
        "inference_latency_ms_std",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if (df["num_seeds"] != 3).any():
        raise ValueError(f"{path} is not a 3-seed benchmark.")
    return df


def _infer_dataset_path(path: Path) -> Path:
    return Path("outputs/data/deepmimo_asu_campus_3p5_narrowband.pt")


def main() -> None:
    args = parse_args()
    random_path = Path(args.random)
    contiguous_path = Path(args.contiguous)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_csv.with_suffix(".md")
    out_png = out_csv.with_suffix(".png")

    random_df = _load_table(args.random).copy()
    contiguous_df = _load_table(args.contiguous).copy()
    random_df["split_mode"] = "random"
    contiguous_df["split_mode"] = "contiguous"

    dataset = load_channel_dataset(_infer_dataset_path(random_path))
    k = int(dataset.metadata.get("num_users", dataset.channels.shape[-2]))
    nt = int(dataset.metadata.get("num_bs_ant", dataset.channels.shape[-1]))
    nsc = int(dataset.metadata.get("num_subcarriers", 1))

    merged = random_df.merge(
        contiguous_df,
        on="method",
        suffixes=("_random", "_contiguous"),
    )
    merged["mean_se_delta_contiguous_minus_random"] = (
        merged["mean_se_mean_contiguous"] - merged["mean_se_mean_random"]
    )
    merged["relative_position_generalization_gap"] = (
        merged["mean_se_mean_contiguous"] - merged["mean_se_mean_random"]
    ) / merged["mean_se_mean_random"].abs().clip(lower=1e-12)
    merged["latency_source_consistent"] = (
        merged["inference_latency_ms_mean_random"].notna() & merged["inference_latency_ms_mean_contiguous"].notna()
    )
    merged["K"] = k
    merged["Nt"] = nt
    merged["Nsc"] = nsc
    merged.to_csv(out_csv, index=False)

    lines = [
        "# DeepMIMO Random vs Contiguous",
        "",
        f"- num_seeds(random) = {int(random_df['num_seeds'].iloc[0])}",
        f"- num_seeds(contiguous) = {int(contiguous_df['num_seeds'].iloc[0])}",
        f"- scale: K={k}, Nt={nt}, Nsc={nsc}",
        "",
        "| Method | Random mean±std | Contiguous mean±std | Relative gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in merged.sort_values("mean_se_mean_random", ascending=False).itertuples():
        lines.append(
            f"| {row.method} | {row.mean_se_mean_random:.6f} ± {row.mean_se_std_random:.6f} | "
            f"{row.mean_se_mean_contiguous:.6f} ± {row.mean_se_std_contiguous:.6f} | "
            f"{row.relative_position_generalization_gap:+.4%} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_df = merged.sort_values("mean_se_mean_random", ascending=False)
    x = range(len(plot_df))
    plt.figure(figsize=(9.0, 4.8))
    plt.plot(x, plot_df["mean_se_mean_random"], marker="o", label="random")
    plt.plot(x, plot_df["mean_se_mean_contiguous"], marker="o", label="contiguous")
    plt.xticks(list(x), plot_df["method"], rotation=20)
    plt.ylabel("Mean SE")
    plt.title("DeepMIMO Random vs Contiguous")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    summary = {
        "random_num_seeds": int(random_df["num_seeds"].iloc[0]),
        "contiguous_num_seeds": int(contiguous_df["num_seeds"].iloc[0]),
        "K": k,
        "Nt": nt,
        "Nsc": nsc,
        "wideband_allowed": nsc > 1,
    }
    with open(out_csv.with_suffix(".yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle)
    print(f"Saved comparison CSV to {out_csv}")


if __name__ == "__main__":
    main()
