#!/usr/bin/env python
"""Run a DeepMIMO model-family benchmark across seeds and split modes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.evaluation import add_relative_gaps


LEARNED_METHODS = {"cnn", "residual_rzf", "residual_wmmse", "unfolded_rzf", "unfolded_wmmse_lite"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split-mode", choices=["random", "contiguous"], required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _curve_summary(curve_df: pd.DataFrame, method: str) -> dict[str, float]:
    ordered = curve_df.sort_values("snr_db")
    method_df = ordered[ordered["method"] == method].copy()
    return {
        "mean_se": float(method_df["se"].mean()),
        "mean_gap_to_rzf": float(method_df["relative_gap_to_rzf"].mean()),
        "mean_gap_to_best_baseline": float(method_df["relative_gap_to_best_baseline"].mean()),
        "mean_gap_to_strongest_reference": float(method_df["relative_gap_to_strongest_reference"].mean()),
        "gap_10db": float(method_df.loc[method_df["snr_db"] == 10.0, "relative_gap_to_rzf"].iloc[0]),
        "gap_15db": float(method_df.loc[method_df["snr_db"] == 15.0, "relative_gap_to_rzf"].iloc[0]),
        "gap_20db": float(method_df.loc[method_df["snr_db"] == 20.0, "relative_gap_to_rzf"].iloc[0]),
        "high_snr_mean_gap": float(
            method_df[method_df["snr_db"].isin([10.0, 15.0, 20.0])]["relative_gap_to_rzf"].mean()
        ),
    }


def _artifact_spec_for(method: str, train_out: Path, config_path: str) -> str:
    return f"{method}={method},{config_path},{train_out / 'best.pt'}"


def _train_learned_method(
    method: str,
    args: argparse.Namespace,
    split_path: Path,
    out_dir: Path,
) -> tuple[Path | None, Path]:
    if method == "cnn":
        pretrain_cfg = "configs/deepmimo_cnn_pretrain.yaml"
        finetune_cfg = "configs/deepmimo_cnn_finetune.yaml"
        pretrain_out = out_dir / "cnn_pretrain"
        train_out = out_dir / "cnn_finetune"
        if not (pretrain_out / "best.pt").exists():
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
                    "--device",
                    args.device,
                ]
            )
        if not (train_out / "best.pt").exists():
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
                    "--device",
                    args.device,
                ]
            )
        return pretrain_out, train_out

    config_map = {
        "residual_rzf": "configs/deepmimo_residual_rzf.yaml",
        "residual_wmmse": "configs/deepmimo_residual_wmmse.yaml",
        "unfolded_rzf": "configs/deepmimo_unfolded_rzf.yaml",
        "unfolded_wmmse_lite": "configs/deepmimo_unfolded_wmmse_lite.yaml",
    }
    config_path = config_map[method]
    train_out = out_dir / method
    if not (train_out / "best.pt").exists():
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
                config_path,
                "--out",
                str(train_out),
                "--device",
                args.device,
            ]
        )
    return None, train_out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_methods = [method for method in args.methods if method not in LEARNED_METHODS]
    learned_methods = [method for method in args.methods if method in LEARNED_METHODS]
    all_rows: list[dict[str, object]] = []
    all_curves: list[pd.DataFrame] = []
    latency_rows: list[pd.DataFrame] = []

    for seed in args.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        split_path = seed_dir / f"deepmimo_{args.split_mode}_seed{seed}.pt"
        _run(
            [
                "python",
                "scripts/make_splits.py",
                "--dataset-type",
                "deepmimo",
                "--data",
                args.data,
                "--split-mode",
                args.split_mode,
                "--seed",
                str(seed),
                "--out",
                str(split_path),
            ]
        )

        baseline_out = seed_dir / "baselines"
        if baseline_methods and not (baseline_out / "metrics" / "baseline_results.csv").exists():
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
                    *baseline_methods,
                    "--out",
                    str(baseline_out),
                    "--device",
                    args.device,
                ]
            )
        baseline_df = pd.read_csv(baseline_out / "metrics" / "baseline_results.csv")[
            ["method", "snr_db", "se", "runtime_sec"]
        ].copy()

        learned_curves: list[pd.DataFrame] = []
        artifact_specs: list[str] = []
        for method in learned_methods:
            pretrain_out, train_out = _train_learned_method(method, args, split_path, seed_dir / "runs")
            eval_out = seed_dir / "eval" / method
            eval_out.mkdir(parents=True, exist_ok=True)
            config_map = {
                "cnn": "configs/deepmimo_cnn_finetune.yaml",
                "residual_rzf": "configs/deepmimo_residual_rzf.yaml",
                "residual_wmmse": "configs/deepmimo_residual_wmmse.yaml",
                "unfolded_rzf": "configs/deepmimo_unfolded_rzf.yaml",
                "unfolded_wmmse_lite": "configs/deepmimo_unfolded_wmmse_lite.yaml",
            }
            config_path = config_map[method]
            if not (eval_out / "deepmimo_all_methods.csv").exists():
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
                        config_path,
                        "--methods",
                        *(baseline_methods + [method]),
                        "--out",
                        str(eval_out),
                        "--device",
                        args.device,
                    ]
                )
            curve_df = pd.read_csv(eval_out / "deepmimo_all_methods.csv")
            learned_curves.append(curve_df[curve_df["method"] == method][["method", "snr_db", "se", "runtime_sec"]].copy())
            artifact_specs.append(_artifact_spec_for(method, train_out, config_path))

        combined = pd.concat([baseline_df] + learned_curves, ignore_index=True).drop_duplicates(
            subset=["method", "snr_db"], keep="last"
        )
        combined = add_relative_gaps(combined)
        combined["seed"] = seed
        combined["split_mode"] = args.split_mode
        all_curves.append(combined)

        latency_out = seed_dir / "latency"
        if not (latency_out / "latency_table.csv").exists():
            _run(
                [
                    "python",
                    "scripts/benchmark_latency.py",
                    "--dataset-type",
                    "deepmimo",
                    "--data",
                    args.data,
                    "--split",
                    str(split_path),
                    "--methods",
                    *args.methods,
                    "--batch-size",
                    "512",
                    "--warmup-runs",
                    "20",
                    "--timed-runs",
                    "100",
                    "--device",
                    args.device,
                    "--out",
                    str(latency_out),
                    *sum([["--artifact-spec", spec] for spec in artifact_specs], []),
                ]
            )
        latency_df = pd.read_csv(latency_out / "latency_table.csv")
        latency_df["seed"] = seed
        latency_rows.append(latency_df)

        for method in combined["method"].unique().tolist():
            metrics = _curve_summary(combined, method)
            runtime_match = latency_df[latency_df["method"] == method]
            all_rows.append(
                {
                    "seed": seed,
                    "split_mode": args.split_mode,
                    "method": method,
                    **metrics,
                    "inference_latency_ms": float(runtime_match["inference_latency_ms"].iloc[0]) if not runtime_match.empty else None,
                }
            )

    table = pd.DataFrame(all_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    table.to_csv(out_dir / "deepmimo_model_family_table.csv", index=False)

    mean_std = (
        table.groupby("method", as_index=False)
        .agg(
            num_seeds=("seed", "nunique"),
            mean_se_mean=("mean_se", "mean"),
            mean_se_std=("mean_se", "std"),
            gap_to_rzf_mean=("mean_gap_to_rzf", "mean"),
            gap_to_rzf_std=("mean_gap_to_rzf", "std"),
            gap_to_best_reference_mean=("mean_gap_to_strongest_reference", "mean"),
            gap_to_best_reference_std=("mean_gap_to_strongest_reference", "std"),
            high_snr_mean_gap_mean=("high_snr_mean_gap", "mean"),
            high_snr_mean_gap_std=("high_snr_mean_gap", "std"),
            inference_latency_ms_mean=("inference_latency_ms", "mean"),
            inference_latency_ms_std=("inference_latency_ms", "std"),
        )
        .sort_values("mean_se_mean", ascending=False)
        .reset_index(drop=True)
    )
    mean_std.to_csv(out_dir / "deepmimo_model_family_mean_std.csv", index=False)

    curve_df = pd.concat(all_curves, ignore_index=True)
    curve_stats = (
        curve_df.groupby(["method", "snr_db"], as_index=False)
        .agg(se_mean=("se", "mean"), se_std=("se", "std"))
        .sort_values(["method", "snr_db"])
    )
    plt.figure(figsize=(8.0, 4.8))
    for method, group in curve_stats.groupby("method"):
        plt.plot(group["snr_db"], group["se_mean"], marker="o", label=method)
        lower = group["se_mean"] - group["se_std"].fillna(0.0)
        upper = group["se_mean"] + group["se_std"].fillna(0.0)
        plt.fill_between(group["snr_db"], lower, upper, alpha=0.15)
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
    plt.title(f"DeepMIMO Model Family SE vs SNR ({args.split_mode})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "deepmimo_model_family_se_vs_snr.png")
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    runtime_df = mean_std.sort_values("inference_latency_ms_mean")
    plt.bar(runtime_df["method"], runtime_df["inference_latency_ms_mean"], yerr=runtime_df["inference_latency_ms_std"].fillna(0.0), capsize=3)
    plt.ylabel("Inference latency (ms)")
    plt.title(f"DeepMIMO Model Family Runtime ({args.split_mode})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "deepmimo_model_family_runtime.png")
    plt.close()

    print(f"Saved DeepMIMO model family benchmark to {out_dir}")


if __name__ == "__main__":
    main()
