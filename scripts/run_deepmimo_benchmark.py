#!/usr/bin/env python
"""Run a reproducible DeepMIMO smoke/quick benchmark across seeds."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _summary(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "summary.yaml").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        split_path = Path("outputs/splits") / f"deepmimo_seed{seed}.pt"
        _run(
            [
                "python",
                "scripts/make_splits.py",
                "--dataset-type",
                "deepmimo",
                "--data",
                args.data,
                "--split-mode",
                "random",
                "--seed",
                str(seed),
                "--out",
                str(split_path),
            ]
        )
        pretrain_out = out_dir / f"seed_{seed}_pretrain"
        train_out = out_dir / f"seed_{seed}_finetune"
        eval_out = out_dir / f"seed_{seed}_eval"

        pretrain_cfg = "configs/deepmimo_cnn_pretrain.yaml"
        finetune_cfg = "configs/deepmimo_cnn_finetune.yaml"
        if args.quick:
            quick_pretrain = out_dir / f"seed_{seed}_pretrain_quick.yaml"
            quick_finetune = out_dir / f"seed_{seed}_finetune_quick.yaml"
            for src, dst in ((pretrain_cfg, quick_pretrain), (finetune_cfg, quick_finetune)):
                cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
                cfg["training"]["epochs"] = 3 if "pretrain" in dst.name else 5
                dst.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            pretrain_cfg = str(quick_pretrain)
            finetune_cfg = str(quick_finetune)

        _run(
            [
                "python",
                "scripts/pretrain.py",
                "--dataset-type",
                "deepmimo",
                "--data",
                args.data,
                "--split",
                str(split_path),
                "--config",
                pretrain_cfg,
                "--teacher",
                "rzf",
                "--out",
                str(pretrain_out),
            ]
        )
        _run(
            [
                "python",
                "scripts/train.py",
                "--dataset-type",
                "deepmimo",
                "--data",
                args.data,
                "--split",
                str(split_path),
                "--config",
                finetune_cfg,
                "--init-ckpt",
                str(pretrain_out / "best.pt"),
                "--out",
                str(train_out),
            ]
        )
        _run(
            [
                "python",
                "scripts/evaluate_all.py",
                "--dataset-type",
                "deepmimo",
                "--data",
                args.data,
                "--split",
                str(split_path),
                "--ckpt",
                str(train_out / "best.pt"),
                "--config",
                finetune_cfg,
                "--methods",
                "mrt",
                "zf",
                "rzf",
                "dft",
                "cnn",
                "--out",
                str(eval_out),
            ]
        )
        summary = _summary(eval_out)
        summary["seed"] = seed
        rows.append(summary)
        curve_df = pd.read_csv(eval_out / "deepmimo_all_methods.csv")
        cnn_df = curve_df[curve_df["method"] == "cnn"].copy()
        cnn_df["seed"] = seed
        curve_rows.append(cnn_df)
        if args.quick:
            break

    summary_df = pd.DataFrame(rows)
    numeric_cols = summary_df.select_dtypes(include=["number"]).columns.tolist()
    aggregate = (
        summary_df[numeric_cols]
        .agg(["mean", "std"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    aggregate.to_csv(out_dir / "deepmimo_benchmark_summary.csv", index=False)

    md_lines = ["# DeepMIMO Benchmark Summary", "", f"quick_mode: {args.quick}", "", aggregate.to_markdown(index=False)]
    (out_dir / "deepmimo_benchmark_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    if curve_rows:
        curve_df = pd.concat(curve_rows, ignore_index=True)
        stats_df = curve_df.groupby("snr_db", as_index=False).agg(se_mean=("se", "mean"), se_std=("se", "std"))
        plt.figure(figsize=(7.0, 4.5))
        plt.plot(stats_df["snr_db"], stats_df["se_mean"], marker="o", label="cnn mean")
        lower = stats_df["se_mean"] - stats_df["se_std"].fillna(0.0)
        upper = stats_df["se_mean"] + stats_df["se_std"].fillna(0.0)
        plt.fill_between(stats_df["snr_db"], lower, upper, alpha=0.2, label="cnn std")
        plt.xlabel("SNR (dB)")
        plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
        plt.title("DeepMIMO CNN SE vs SNR Mean/Std")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "deepmimo_se_vs_snr_mean_std.png")
        plt.close()
    print(f"Saved benchmark summary to {out_dir}")


if __name__ == "__main__":
    main()
