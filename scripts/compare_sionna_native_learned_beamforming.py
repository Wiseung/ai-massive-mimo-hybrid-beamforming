#!/usr/bin/env python
"""Compare analytic and learned beamformers in the Sionna-native receiver chain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import matplotlib.pyplot as plt
import pandas as pd

add_src_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analytic-summary", required=True)
    parser.add_argument("--analytic-metrics", required=True)
    parser.add_argument("--learned-summary", required=True)
    parser.add_argument("--learned-metrics", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    analytic_summary = json.loads(Path(args.analytic_summary).read_text(encoding="utf-8"))
    analytic = pd.read_csv(args.analytic_metrics)
    learned_summary = json.loads(Path(args.learned_summary).read_text(encoding="utf-8"))
    learned = pd.read_csv(args.learned_metrics)
    comparison = learned.copy()

    if "project_rzf" in comparison["method"].values:
        rzf_sr = float(comparison[comparison["method"] == "project_rzf"]["approximate_sum_rate"].iloc[0])
        comparison["gap_to_project_rzf"] = (comparison["approximate_sum_rate"] - rzf_sr) / rzf_sr
    else:
        comparison["gap_to_project_rzf"] = float("nan")
    if "project_wmmse_iter_5" in comparison["method"].values:
        wmmse5_sr = float(comparison[comparison["method"] == "project_wmmse_iter_5"]["approximate_sum_rate"].iloc[0])
        comparison["gap_to_project_wmmse_iter_5"] = (comparison["approximate_sum_rate"] - wmmse5_sr) / wmmse5_sr
    else:
        comparison["gap_to_project_wmmse_iter_5"] = float("nan")

    comparison.to_csv(out_dir / "native_learned_comparison.csv", index=False)
    plt.figure(figsize=(8, 4.5))
    plt.bar(comparison["method"], comparison["symbol_mse"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("symbol_mse")
    plt.tight_layout()
    plt.savefig(out_dir / "native_learned_mse_by_method.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(comparison["method"], comparison["effective_sinr_db"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("effective_sinr_db")
    plt.tight_layout()
    plt.savefig(out_dir / "native_learned_sinr_by_method.png", dpi=160)
    plt.close()

    def _row(method: str):
        frame = comparison[comparison["method"] == method]
        return frame.iloc[0] if not frame.empty else None

    residual = _row("learned_residual_rzf")
    distill = _row("learned_residual_wmmse_distill")
    lines = [
        "# Native Learned Beamforming Comparison",
        "",
        f"1. learned_residual_rzf native receiver success: `{None if residual is None else bool(residual['native_receiver_success'])}`",
        f"2. learned_residual_wmmse_distill native receiver success: `{None if distill is None else bool(distill['native_receiver_success'])}`",
        f"3. learned_residual_rzf gap to project_rzf: `{None if residual is None else float(residual['gap_to_project_rzf']):+.6%}`",
        f"4. learned_residual_rzf gap to project_wmmse_iter_5: `{None if residual is None else float(residual['gap_to_project_wmmse_iter_5']):+.6%}`",
        f"5. learned_residual_wmmse_distill gap to project_rzf: `{None if distill is None else float(distill['gap_to_project_rzf']):+.6%}`",
        f"6. learned_residual_wmmse_distill gap to project_wmmse_iter_5: `{None if distill is None else float(distill['gap_to_project_wmmse_iter_5']):+.6%}`",
        f"7. teacher_used_during_inference false for learned methods: `{bool(comparison[comparison['method_type']=='learned']['teacher_used_during_inference'].fillna(False).eq(False).all())}`",
        f"8. recommend learned_residual_rzf as native-chain mainline: `{bool(residual is not None and residual['native_receiver_success'])}`",
        f"9. current result remains synthetic/project-H_f-assisted native receiver benchmark: `True`",
        "",
        "## Notes",
        f"- analytic native receiver success methods from prior pure-analytic run: `{analytic_summary['methods_successful_under_native_receiver']}`",
        f"- learned native receiver success methods: `{learned_summary['methods_successful_under_native_receiver']}`",
        f"- learned skipped missing checkpoint: `{learned_summary['methods_skipped_missing_checkpoint']}`",
        "- gap calculations are referenced to the analytic methods inside the same learned native-chain run, not to a separate earlier analytic artifact.",
    ]
    (out_dir / "native_learned_comparison.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved native learned comparison to {out_dir}")


if __name__ == "__main__":
    main()
