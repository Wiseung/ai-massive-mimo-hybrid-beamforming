#!/usr/bin/env python
"""Run a reproducible DeepMIMO benchmark across seeds."""

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
    parser.add_argument("--model-family", choices=["cnn", "residual_rzf"], default="cnn")
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
    model_name = args.model_family
    methods = ["mrt", "zf", "rzf", "dft", model_name]

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
        baseline_out = out_dir / f"seed_{seed}_baselines"

        if model_name == "cnn":
            pretrain_cfg = "configs/deepmimo_cnn_pretrain.yaml"
            finetune_cfg = "configs/deepmimo_cnn_finetune.yaml"
            teacher = "rzf"
        else:
            pretrain_cfg = None
            finetune_cfg = "configs/deepmimo_residual_rzf.yaml"
            teacher = None
        if args.quick:
            quick_finetune = out_dir / f"seed_{seed}_finetune_quick.yaml"
            cfg = yaml.safe_load(Path(finetune_cfg).read_text(encoding="utf-8"))
            cfg["training"]["epochs"] = 5
            quick_finetune.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            finetune_cfg = str(quick_finetune)
            if pretrain_cfg is not None:
                quick_pretrain = out_dir / f"seed_{seed}_pretrain_quick.yaml"
                cfg = yaml.safe_load(Path(pretrain_cfg).read_text(encoding="utf-8"))
                cfg["training"]["epochs"] = 3
                quick_pretrain.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                pretrain_cfg = str(quick_pretrain)

        _run(
            [
                "python",
                "scripts/run_baselines.py",
                "--dataset-type",
                "deepmimo",
                "--data",
                args.data,
                "--split",
                str(split_path),
                "--methods",
                "mrt",
                "zf",
                "rzf",
                "dft",
                "fd_zf",
                "fd_rzf",
                "--out",
                str(baseline_out),
            ]
        )

        init_ckpt = None
        if pretrain_cfg is not None and teacher is not None:
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
                    teacher,
                    "--out",
                    str(pretrain_out),
                ]
            )
            init_ckpt = str(pretrain_out / "best.pt")
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
                "--out",
                str(train_out),
            ]
            + (["--init-ckpt", init_ckpt] if init_ckpt is not None else [])
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
                *methods,
                "--out",
                str(eval_out),
            ]
        )
        summary = _summary(eval_out)
        summary["seed"] = seed
        summary["model_family"] = model_name
        rows.append(summary)
        curve_df = pd.read_csv(eval_out / "deepmimo_all_methods.csv")
        learned_df = curve_df[curve_df["method"] == model_name].copy()
        learned_df["seed"] = seed
        curve_rows.append(learned_df)
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
    aggregate["num_seeds"] = len(rows)
    aggregate.to_csv(out_dir / "deepmimo_benchmark_summary.csv", index=False)

    md_lines = [
        "# DeepMIMO Benchmark Summary",
        "",
        f"quick_mode: {args.quick}",
        f"model_family: {model_name}",
        f"num_seeds: {len(rows)}",
        "",
        aggregate.to_markdown(index=False),
    ]
    (out_dir / "deepmimo_benchmark_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    if curve_rows:
        curve_df = pd.concat(curve_rows, ignore_index=True)
        stats_df = curve_df.groupby("snr_db", as_index=False).agg(se_mean=("se", "mean"), se_std=("se", "std"))
        plt.figure(figsize=(7.0, 4.5))
        plt.plot(stats_df["snr_db"], stats_df["se_mean"], marker="o", label=f"{model_name} mean")
        lower = stats_df["se_mean"] - stats_df["se_std"].fillna(0.0)
        upper = stats_df["se_mean"] + stats_df["se_std"].fillna(0.0)
        plt.fill_between(stats_df["snr_db"], lower, upper, alpha=0.2, label=f"{model_name} std")
        plt.xlabel("SNR (dB)")
        plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
        plt.title(f"DeepMIMO {model_name} SE vs SNR Mean/Std")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "deepmimo_se_vs_snr_mean_std.png")
        plt.close()
    print(f"Saved benchmark summary to {out_dir}")


if __name__ == "__main__":
    main()
