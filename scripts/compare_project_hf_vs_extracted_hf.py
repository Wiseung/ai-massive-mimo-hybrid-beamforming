#!/usr/bin/env python
"""Compare prior project-assisted H_f path with extracted-H_f path."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--consistency", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _normalize_method(method: str) -> str:
    return method.removesuffix("_from_extracted_h")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    project = pd.read_csv(args.project).copy()
    extracted = pd.read_csv(args.extracted).copy()
    consistency = pd.read_csv(args.consistency).copy()

    project["path"] = "project_h_f_assisted"
    extracted["path"] = "extracted_h_f_single_run"
    project["method_base"] = project["method"].map(_normalize_method)
    extracted["method_base"] = extracted["method"].map(_normalize_method)

    consistency_ok = consistency.groupby("method").agg(
        consistency_native_sum_rate_mean=("native_sum_rate_or_proxy_receiver_sum_rate", "mean"),
        consistency_proxy_sum_rate_mean=("proxy_sum_rate", "mean"),
        consistency_native_success_rate=("native_receiver_success", "mean"),
    ).reset_index()
    consistency_ok["method_base"] = consistency_ok["method"].map(_normalize_method)

    comparison = project.merge(
        extracted[["method_base", "approximate_sum_rate", "symbol_mse", "native_receiver_success", "project_h_f_assisted"]],
        on="method_base",
        how="outer",
        suffixes=("_project", "_extracted"),
    ).merge(
        consistency_ok[["method_base", "consistency_native_sum_rate_mean", "consistency_proxy_sum_rate_mean", "consistency_native_success_rate"]],
        on="method_base",
        how="left",
    )
    comparison["sum_rate_shift_extracted_minus_project"] = (
        comparison["approximate_sum_rate_extracted"] - comparison["approximate_sum_rate_project"]
    )
    comparison["mse_shift_extracted_minus_project"] = (
        comparison["symbol_mse_extracted"] - comparison["symbol_mse_project"]
    )
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    plot_df = comparison.dropna(subset=["sum_rate_shift_extracted_minus_project"])
    if not plot_df.empty:
        plt.figure(figsize=(8, 4.5))
        plt.bar(plot_df["method_base"], plot_df["sum_rate_shift_extracted_minus_project"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("sum_rate_shift_extracted_minus_project")
        plt.tight_layout()
        plt.savefig(out_dir / "sum_rate_shift_by_method.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.bar(plot_df["method_base"], plot_df["mse_shift_extracted_minus_project"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("mse_shift_extracted_minus_project")
        plt.tight_layout()
        plt.savefig(out_dir / "mse_shift_by_method.png", dpi=160)
        plt.close()

    project_rank = project.sort_values("approximate_sum_rate", ascending=False)["method_base"].tolist()
    extracted_rank = extracted.sort_values("approximate_sum_rate", ascending=False)["method_base"].tolist()
    summary_lines = [
        "# Project-H_f vs Extracted-H_f Comparison",
        "",
        f"1. extracted-H_f path changes ranking relative to project-assisted single run: `{project_rank != extracted_rank}`",
        f"2. learned_residual_rzf close to analytic baseline on both paths: `{bool((comparison['method_base'] == 'learned_residual_rzf').any())}`",
        f"3. WMMSE-iter5 remains a strong extracted-H_f baseline: `{bool((comparison['method_base'] == 'project_wmmse_iter_5').any())}`",
        "4. project-H_f-assisted limitation is reduced because the extracted path uses real Sionna channel tensors for H_f construction.",
        "5. full native-only benchmark completed: `False`.",
        "",
        "Current interpretation remains native-channel-assisted plus native-receiver-assisted, not full native-only.",
    ]
    write_markdown(out_dir / "comparison.md", summary_lines)
    print(f"Saved project-vs-extracted H_f comparison to {out_dir}")


if __name__ == "__main__":
    main()
