#!/usr/bin/env python
"""Compare baseline and learned beamformer families on a common fair benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.data.dataset import load_channel_dataset
from beamforming.evaluation import add_relative_gaps, evaluate_baselines_by_snr, get_eval_subset


HIGH_SNR_POINTS = [10.0, 15.0, 20.0]
GAP_FORMULA = "(method_se - reference_se) / reference_se"
SYNTHETIC_DATA = Path("outputs/data/synthetic_narrowband.pt")
REFERENCE_CONFIG = Path("configs/synthetic_residual_rzf.yaml")


MODEL_RUNS: list[dict[str, str]] = [
    {
        "method": "cnn",
        "summary": "outputs/comparisons/synthetic_cnn_finetune/summary.yaml",
        "curve": "outputs/comparisons/synthetic_cnn_finetune/synthetic_all_methods.csv",
        "train_summary": "outputs/runs/cnn_finetune_rzf/train_summary.yaml",
    },
    {
        "method": "residual_rzf",
        "summary": "outputs/comparisons/synthetic_residual_rzf/summary.yaml",
        "curve": "outputs/comparisons/synthetic_residual_rzf/synthetic_all_methods.csv",
        "train_summary": "outputs/runs/synthetic_residual_rzf/train_summary.yaml",
    },
    {
        "method": "unfolded_rzf",
        "summary": "outputs/comparisons/synthetic_unfolded_rzf/summary.yaml",
        "curve": "outputs/comparisons/synthetic_unfolded_rzf/synthetic_all_methods.csv",
        "train_summary": "outputs/runs/synthetic_unfolded_rzf/train_summary.yaml",
    },
    {
        "method": "residual_wmmse",
        "summary": "outputs/comparisons/synthetic_residual_wmmse/summary.yaml",
        "curve": "outputs/comparisons/synthetic_residual_wmmse/synthetic_all_methods.csv",
        "train_summary": "outputs/runs/synthetic_residual_wmmse_finetune/train_summary.yaml",
    },
    {
        "method": "unfolded_wmmse_lite",
        "summary": "outputs/comparisons/synthetic_unfolded_wmmse_lite/summary.yaml",
        "curve": "outputs/comparisons/synthetic_unfolded_wmmse_lite/synthetic_all_methods.csv",
        "train_summary": "outputs/runs/synthetic_unfolded_wmmse_lite/train_summary.yaml",
    },
]
WMMSE_SWEEP_TABLE = Path("outputs/comparisons/wmmse_iteration_sweep/wmmse_iteration_table.csv")
WMMSE_SWEEP_CURVES = Path("outputs/comparisons/wmmse_iteration_sweep/wmmse_iteration_curves.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_fair_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"method", "snr_db", "se", "runtime_sec"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing required columns: {sorted(required - set(df.columns))}")
    return df.copy()


def _load_baseline_reference() -> pd.DataFrame:
    dataset = load_channel_dataset(SYNTHETIC_DATA)
    cfg = _read_yaml(REFERENCE_CONFIG)
    eval_subset = get_eval_subset(
        dataset,
        val_fraction=float(cfg["training"].get("val_fraction", 0.1)),
        seed=int(cfg["training"]["seed"]),
    )
    num_rf_chains = int(dataset.metadata.get("num_rf_chains", min(dataset.channels.shape[-2], 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_df = evaluate_baselines_by_snr(["mrt", "zf", "rzf", "dft", "wmmse"], eval_subset, num_rf_chains=num_rf_chains, device=device)
    return baseline_df


def _merge_curves() -> pd.DataFrame:
    frames: list[pd.DataFrame] = [_load_baseline_reference()]
    for entry in MODEL_RUNS:
        curve_path = Path(entry["curve"])
        if not curve_path.exists():
            continue
        df = _load_fair_curve(curve_path)
        keep_cols = [col for col in ["method", "snr_db", "se", "runtime_sec"] if col in df.columns]
        frames.append(df[keep_cols].copy())
    if not frames:
        raise FileNotFoundError("No fair evaluation comparison CSVs found for model family comparison.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["method", "snr_db"], keep="last")
    combined = add_relative_gaps(combined)
    return combined


def _summary_row(method: str, curve: pd.DataFrame, train_summary: dict[str, Any], reference_method: str) -> dict[str, Any]:
    ordered = curve.sort_values("snr_db")
    method_df = ordered[ordered["method"] == method].copy()
    if method_df.empty:
        raise ValueError(f"Method {method} not found in fair comparison CSV.")
    mean_se = float(method_df["se"].mean())
    wmmse_ref = ordered[ordered["method"] == "wmmse"][["snr_db", "se"]].rename(columns={"se": "wmmse_se"})
    method_df = method_df.merge(wmmse_ref, on="snr_db", how="left")
    method_df["gap_to_wmmse"] = (method_df["se"] - method_df["wmmse_se"]) / method_df["wmmse_se"].abs().clip(lower=1e-12)
    method_df["gap_to_best_reference"] = method_df["relative_gap_to_strongest_reference"]
    method_df["gap_to_rzf"] = method_df["relative_gap_to_rzf"]
    high_df = method_df[method_df["snr_db"].isin(HIGH_SNR_POINTS)]
    gap_map = {f"gap_{int(row.snr_db)}db": float(row.gap_to_rzf) for row in high_df.itertuples()}
    return {
        "method": method,
        "reference_method": reference_method,
        "gap_formula": GAP_FORMULA,
        "num_snr_points": int(method_df["snr_db"].nunique()),
        "high_snr_points_used": ",".join(str(int(x)) for x in HIGH_SNR_POINTS),
        "mean_se": mean_se,
        "gap_to_rzf": float(method_df["gap_to_rzf"].mean()),
        "gap_to_wmmse": float(method_df["gap_to_wmmse"].mean()) if method_df["wmmse_se"].notna().any() else None,
        "gap_to_best_reference": float(method_df["gap_to_best_reference"].mean()),
        "high_snr_mean_gap": float(high_df["gap_to_rzf"].mean()) if not high_df.empty else None,
        "inference_latency_ms": float(method_df["runtime_sec"].mean() * 1000.0),
        "train_time": train_summary.get("train_time_sec"),
        "best_epoch": train_summary.get("best_epoch"),
        "num_params": train_summary.get("num_params"),
        **gap_map,
    }


def _select_best_latency_wmmse(table: pd.DataFrame) -> pd.Series:
    within_one_percent = table[table["gap_to_full_wmmse"] >= -0.01].sort_values("inference_latency_ms")
    if not within_one_percent.empty:
        return within_one_percent.iloc[0]
    return table.sort_values(["gap_to_full_wmmse", "inference_latency_ms"], ascending=[False, True]).iloc[0]


def _pareto_frontier(table: pd.DataFrame) -> pd.DataFrame:
    valid = table.dropna(subset=["inference_latency_ms", "mean_se"]).sort_values(["inference_latency_ms", "mean_se"])
    frontier_rows = []
    best_so_far = float("-inf")
    for row in valid.itertuples(index=False):
        if float(row.mean_se) > best_so_far + 1e-12:
            frontier_rows.append(row.method)
            best_so_far = float(row.mean_se)
    return valid[valid["method"].isin(frontier_rows)].copy()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = _merge_curves()
    reference_method = "rzf"
    methods = list(dict.fromkeys(combined["method"].tolist()))
    rows = []
    for method in methods:
        train_summary = {}
        for entry in MODEL_RUNS:
            if entry["method"] == method:
                train_summary = _read_yaml(Path(entry["train_summary"]))
                break
        rows.append(_summary_row(method, combined, train_summary, reference_method))
    table = pd.DataFrame(rows).sort_values("mean_se", ascending=False).reset_index(drop=True)
    if WMMSE_SWEEP_TABLE.exists():
        sweep_df = pd.read_csv(WMMSE_SWEEP_TABLE)
        if not sweep_df.empty:
            best_latency = _select_best_latency_wmmse(sweep_df)
            wmmse_row = table[table["method"] == "wmmse"]
            if not wmmse_row.empty:
                wmmse_row = wmmse_row.iloc[0]
                extra = {
                    "method": f"wmmse_iter_{int(best_latency['max_iter'])}",
                    "reference_method": reference_method,
                    "gap_formula": GAP_FORMULA,
                    "num_snr_points": int(wmmse_row["num_snr_points"]),
                    "high_snr_points_used": wmmse_row["high_snr_points_used"],
                    "mean_se": float(best_latency["mean_se"]),
                    "gap_to_rzf": float(best_latency["gap_to_rzf"]),
                    "gap_to_wmmse": float(best_latency["gap_to_full_wmmse"]),
                    "gap_to_best_reference": float(best_latency["gap_to_full_wmmse"]),
                    "high_snr_mean_gap": float(best_latency["high_snr_mean_gap"]),
                    "inference_latency_ms": float(best_latency["inference_latency_ms"]),
                    "train_time": None,
                    "best_epoch": None,
                    "num_params": 0,
                }
                table = pd.concat([table, pd.DataFrame([extra])], ignore_index=True)
                table = table.sort_values("mean_se", ascending=False).reset_index(drop=True)
    table.to_csv(out_dir / "model_family_table.csv", index=False)

    plt.figure(figsize=(7.4, 4.8))
    for method, group in combined.groupby("method"):
        ordered = group.sort_values("snr_db")
        plt.plot(ordered["snr_db"], ordered["se"], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
    plt.title("Model Family SE vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_family_se_vs_snr.png")
    plt.close()

    plt.figure(figsize=(7.4, 4.8))
    runtime_table = table[["method", "inference_latency_ms"]].dropna().sort_values("inference_latency_ms")
    plt.bar(runtime_table["method"], runtime_table["inference_latency_ms"])
    plt.ylabel("Inference latency (ms)")
    plt.title("Model Family Runtime")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "model_family_runtime.png")
    plt.close()

    frontier = _pareto_frontier(table)
    plt.figure(figsize=(7.4, 4.8))
    plt.scatter(table["inference_latency_ms"], table["mean_se"], s=70)
    for row in table.itertuples():
        plt.annotate(row.method, (row.inference_latency_ms, row.mean_se), textcoords="offset points", xytext=(4, 4))
    if not frontier.empty:
        ordered_frontier = frontier.sort_values("inference_latency_ms")
        plt.plot(ordered_frontier["inference_latency_ms"], ordered_frontier["mean_se"], linestyle="--")
    plt.xlabel("Inference latency (ms)")
    plt.ylabel("Mean SE")
    plt.title("SE-Latency Pareto")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_se_latency.png")
    plt.close()

    print(f"Saved model family comparison to {out_dir}")


if __name__ == "__main__":
    main()
