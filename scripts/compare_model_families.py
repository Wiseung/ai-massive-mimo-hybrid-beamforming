#!/usr/bin/env python
"""Compare baseline and learned beamformer families with a unified latency table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.evaluation import GAP_FORMULA


HIGH_SNR_POINTS = [10.0, 15.0, 20.0]
SYNTHETIC_CURVE_FALLBACK = Path("outputs/comparisons/synthetic_residual_wzf_missing.csv")


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
        "summary": "outputs/comparisons/synthetic_residual_wmmse_v2/summary.yaml",
        "curve": "outputs/comparisons/synthetic_residual_wmmse_v2/synthetic_all_methods.csv",
        "train_summary": "outputs/runs/synthetic_residual_wmmse_finetune/train_summary.yaml",
    },
    {
        "method": "unfolded_wmmse_lite",
        "summary": "outputs/comparisons/synthetic_unfolded_wmmse_lite_iter2/summary.yaml",
        "curve": "outputs/comparisons/synthetic_unfolded_wmmse_lite_iter2/synthetic_all_methods.csv",
        "train_summary": "outputs/runs/synthetic_unfolded_wmmse_lite_iter2/train_summary.yaml",
    },
]
WMMSE_SWEEP_TABLE = Path("outputs/comparisons/wmmse_iteration_sweep/wmmse_iteration_table.csv")
WMMSE_SWEEP_CURVES = Path("outputs/comparisons/wmmse_iteration_sweep/wmmse_iteration_curves.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-table", required=False, default="outputs/comparisons/latency/latency_table.csv")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"method", "snr_db", "se"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} is missing required columns: {sorted(required - set(df.columns))}")
    keep_cols = [col for col in ["method", "snr_db", "se", "runtime_sec", "relative_gap_to_rzf", "relative_gap_to_best_baseline", "relative_gap_to_strongest_reference"] if col in df.columns]
    return df[keep_cols].copy()


def _merge_curves() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in MODEL_RUNS:
        curve_path = Path(entry["curve"])
        if curve_path.exists():
            frames.append(_load_curve(curve_path))
    if WMMSE_SWEEP_CURVES.exists():
        sweep_curves = pd.read_csv(WMMSE_SWEEP_CURVES).copy()
        if not sweep_curves.empty:
            for iter_value in sorted(sweep_curves["max_iter"].unique().tolist()):
                sub = sweep_curves[sweep_curves["max_iter"] == iter_value].copy()
                sub["method"] = f"wmmse_iter_{int(iter_value)}"
                frames.append(sub[[col for col in ["method", "snr_db", "se", "runtime_sec", "gap_to_rzf"] if col in sub.columns]].rename(columns={"gap_to_rzf": "relative_gap_to_rzf"}))
    if not frames:
        raise FileNotFoundError("No model-family comparison curves found.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["method", "snr_db"], keep="last")
    return combined


def _get_baseline_curve(combined: pd.DataFrame) -> pd.DataFrame:
    baseline_methods = ["mrt", "zf", "rzf", "dft", "wmmse"]
    parts = []
    for method in baseline_methods:
        sub = combined[combined["method"] == method]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        raise ValueError("Baseline methods are missing from the comparison curves.")
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["method", "snr_db"], keep="last")


def _ensure_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    rzf_ref = result[result["method"] == "rzf"][["snr_db", "se"]].rename(columns={"se": "rzf_se"})
    wmmse_ref = result[result["method"] == "wmmse"][["snr_db", "se"]].rename(columns={"se": "wmmse_se"})
    best_ref = result.groupby("snr_db", as_index=False)["se"].max().rename(columns={"se": "best_reference_se"})
    baseline_ref = (
        result[result["method"].isin(["mrt", "zf", "rzf", "dft", "wmmse"]) | result["method"].astype(str).str.startswith("wmmse_iter_")]
        .groupby("snr_db", as_index=False)["se"]
        .max()
        .rename(columns={"se": "best_baseline_se"})
    )
    result = result.merge(rzf_ref, on="snr_db", how="left").merge(wmmse_ref, on="snr_db", how="left").merge(best_ref, on="snr_db", how="left").merge(baseline_ref, on="snr_db", how="left")
    if "relative_gap_to_rzf" not in result.columns:
        result["relative_gap_to_rzf"] = (result["se"] - result["rzf_se"]) / result["rzf_se"].abs().clip(lower=1e-12)
    if "relative_gap_to_best_baseline" not in result.columns:
        result["relative_gap_to_best_baseline"] = (result["se"] - result["best_baseline_se"]) / result["best_baseline_se"].abs().clip(lower=1e-12)
    if "relative_gap_to_strongest_reference" not in result.columns:
        result["relative_gap_to_strongest_reference"] = (result["se"] - result["best_reference_se"]) / result["best_reference_se"].abs().clip(lower=1e-12)
    return result


def _summary_row(method: str, curve: pd.DataFrame, train_summary: dict[str, Any], latency_df: pd.DataFrame) -> dict[str, Any]:
    method_df = curve[curve["method"] == method].sort_values("snr_db").copy()
    if method_df.empty:
        raise ValueError(f"Method {method} not found in fair comparison CSV.")
    method_df["gap_to_wmmse"] = (method_df["se"] - method_df["wmmse_se"]) / method_df["wmmse_se"].abs().clip(lower=1e-12)
    high_df = method_df[method_df["snr_db"].isin(HIGH_SNR_POINTS)]
    gap_map = {f"gap_{int(row.snr_db)}db": float(row.relative_gap_to_rzf) for row in high_df.itertuples()}
    latency_match = latency_df[latency_df["method"] == method]
    return {
        "method": method,
        "reference_method": "rzf",
        "gap_formula": GAP_FORMULA,
        "num_snr_points": int(method_df["snr_db"].nunique()),
        "high_snr_points_used": ",".join(str(int(x)) for x in HIGH_SNR_POINTS),
        "mean_se": float(method_df["se"].mean()),
        "gap_to_rzf": float(method_df["relative_gap_to_rzf"].mean()),
        "gap_to_wmmse": float(method_df["gap_to_wmmse"].mean()) if method_df["wmmse_se"].notna().any() else None,
        "gap_to_best_reference": float(method_df["relative_gap_to_strongest_reference"].mean()),
        "gap_to_best_baseline": float(method_df["relative_gap_to_best_baseline"].mean()),
        "high_snr_mean_gap": float(high_df["relative_gap_to_rzf"].mean()) if not high_df.empty else None,
        "inference_latency_ms": float(latency_match["inference_latency_ms"].iloc[0]) if not latency_match.empty else None,
        "train_time": train_summary.get("train_time_sec"),
        "best_epoch": train_summary.get("best_epoch"),
        "num_params": train_summary.get("num_params"),
        **gap_map,
    }


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

    latency_path = Path(args.latency_table)
    if not latency_path.exists():
        raise FileNotFoundError(f"Latency table not found: {latency_path}")
    latency_df = pd.read_csv(latency_path)

    combined = _ensure_gap_columns(_merge_curves())
    methods = list(dict.fromkeys(combined["method"].tolist()))

    rows = []
    for method in methods:
        train_summary = {}
        for entry in MODEL_RUNS:
            if entry["method"] == method:
                train_summary = _read_yaml(Path(entry["train_summary"]))
                break
        rows.append(_summary_row(method, combined, train_summary, latency_df))
    table = pd.DataFrame(rows).sort_values("mean_se", ascending=False).reset_index(drop=True)
    table.to_csv(out_dir / "model_family_table_v3.csv", index=False)

    plt.figure(figsize=(8.0, 4.8))
    for method, group in combined.groupby("method"):
        ordered = group.sort_values("snr_db")
        plt.plot(ordered["snr_db"], ordered["se"], marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE / Sum-Rate (bit/s/Hz)")
    plt.title("Model Family SE vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_family_se_vs_snr_v3.png")
    plt.close()

    runtime_table = table[["method", "inference_latency_ms"]].dropna().sort_values("inference_latency_ms")
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(runtime_table["method"], runtime_table["inference_latency_ms"])
    plt.ylabel("Inference latency (ms)")
    plt.title("Model Family Runtime")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "model_family_runtime_v3.png")
    plt.close()

    frontier = _pareto_frontier(table)
    plt.figure(figsize=(8.0, 4.8))
    plt.scatter(table["inference_latency_ms"], table["mean_se"], s=70)
    for row in table.itertuples():
        if pd.notna(row.inference_latency_ms):
            plt.annotate(row.method, (row.inference_latency_ms, row.mean_se), textcoords="offset points", xytext=(4, 4))
    if not frontier.empty:
        ordered_frontier = frontier.sort_values("inference_latency_ms")
        plt.plot(ordered_frontier["inference_latency_ms"], ordered_frontier["mean_se"], linestyle="--")
    plt.xlabel("Inference latency (ms)")
    plt.ylabel("Mean SE")
    plt.title("SE-Latency Pareto")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_se_latency_v3.png")
    plt.close()

    print(f"Saved model family comparison to {out_dir}")


if __name__ == "__main__":
    main()
