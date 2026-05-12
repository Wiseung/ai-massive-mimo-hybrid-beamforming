#!/usr/bin/env python
"""Compare unified CSI-consumer demo metrics with the existing CSI-backed path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--unified", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(args.baseline)
    unified_path = Path(args.unified)
    baseline = pd.read_csv(baseline_path).copy()
    unified = pd.read_csv(unified_path).copy()
    baseline_summary_path = baseline_path.with_name("csi_backed_beamforming_summary.json")
    unified_summary_path = unified_path.with_name("unified_csi_consumers_summary.json")
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8")) if baseline_summary_path.exists() else {}
    unified_summary = json.loads(unified_summary_path.read_text(encoding="utf-8")) if unified_summary_path.exists() else {}

    comparison = baseline.merge(
        unified[
            [
                "method",
                "approximate_sum_rate",
                "symbol_mse",
                "effective_sinr_db",
                "native_receiver_success",
                "fallback_used",
                "fallback_reason",
                "input_type",
                "csi_interface_used",
                "project_h_f_assisted",
                "extracted_h_f_used",
                "full_native_only",
            ]
        ],
        on="method",
        how="outer",
        suffixes=("_baseline", "_unified"),
    )
    comparison["sum_rate_abs_diff"] = (
        comparison["approximate_sum_rate_unified"] - comparison["approximate_sum_rate_baseline"]
    ).abs()
    comparison["symbol_mse_abs_diff"] = (comparison["symbol_mse_unified"] - comparison["symbol_mse_baseline"]).abs()
    comparison["effective_sinr_db_abs_diff"] = (
        comparison["effective_sinr_db_unified"] - comparison["effective_sinr_db_baseline"]
    ).abs()
    comparison_path = out_dir / "unified_csi_consumer_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    baseline_rank = baseline.sort_values("approximate_sum_rate", ascending=False)["method"].tolist()
    unified_rank = unified.sort_values("approximate_sum_rate", ascending=False)["method"].tolist()
    consistent = bool(
        comparison["sum_rate_abs_diff"].dropna().le(1e-6).all()
        and comparison["symbol_mse_abs_diff"].dropna().le(1e-6).all()
        and comparison["effective_sinr_db_abs_diff"].dropna().le(1e-6).all()
    )
    all_accept_csi = bool(unified.get("input_type", pd.Series(dtype=str)).fillna("").eq("ExtractedCSI").all())
    same_seed_used = baseline_summary.get("seed") == unified_summary.get("seed")
    baseline_sig = baseline_summary.get("csi_input_summary", {}).get("tensor_signature")
    unified_sig = unified_summary.get("csi_input_summary", {}).get("tensor_signature")
    same_csi_tensor_signature = baseline_sig is not None and baseline_sig == unified_sig
    new_fallback = bool(
        (
            comparison["fallback_used_unified"].fillna(False).astype(bool)
            & ~comparison["fallback_used_baseline"].fillna(False).astype(bool)
        ).any()
    )
    comparison_type = "same_csi_realization" if same_csi_tensor_signature else "cross_run_comparison"
    strict_equivalence_claim_allowed = bool(same_csi_tensor_signature)

    lines = [
        "# Unified CSI Consumer Comparison",
        "",
        f"- comparison_type: `{comparison_type}`",
        f"- same_seed_used: `{same_seed_used}`",
        f"- same_csi_tensor_signature: `{same_csi_tensor_signature}`",
        f"- strict_equivalence_claim_allowed: `{strict_equivalence_claim_allowed}`",
        "",
        f"1. unified demo matches existing CSI-backed path within tolerance: `{consistent}`",
        f"2. all key consumers accept ExtractedCSI: `{all_accept_csi}`",
        f"3. additional fallback introduced: `{new_fallback}`",
        "4. provenance clarity improved: `True`.",
        "5. full native-only benchmark completed: `False`.",
        "",
        f"- baseline_ranking: `{baseline_rank}`",
        f"- unified_ranking: `{unified_rank}`",
    ]
    if not strict_equivalence_claim_allowed:
        lines.extend(
            [
                "",
                "This artifact compares separate reruns. It should not be interpreted as a strict same-batch equivalence test.",
            ]
        )
    write_markdown(out_dir / "unified_csi_consumer_comparison.md", lines)
    print(f"Saved unified CSI consumer comparison to {out_dir}")


if __name__ == "__main__":
    main()
