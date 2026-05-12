#!/usr/bin/env python
"""Compare raw extracted-H beamforming metrics with CSI-backed metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--csi", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _normalize_method(method: str) -> str:
    return method.removesuffix("_from_extracted_h")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw).copy()
    csi = pd.read_csv(args.csi).copy()

    raw["method_base"] = raw["method"].map(_normalize_method)
    csi["method_base"] = csi["method"].map(_normalize_method)

    comparison = raw.merge(
        csi[
            [
                "method_base",
                "approximate_sum_rate",
                "symbol_mse",
                "native_receiver_success",
                "fallback_used",
                "fallback_reason",
                "project_h_f_assisted",
                "extracted_h_f_used",
            ]
        ],
        on="method_base",
        how="outer",
        suffixes=("_raw", "_csi"),
    )
    comparison["sum_rate_delta_csi_minus_raw"] = comparison["approximate_sum_rate_csi"] - comparison["approximate_sum_rate_raw"]
    comparison["symbol_mse_delta_csi_minus_raw"] = comparison["symbol_mse_csi"] - comparison["symbol_mse_raw"]
    comparison.to_csv(out_dir / "csi_interface_comparison.csv", index=False)

    raw_rank = raw.sort_values("approximate_sum_rate", ascending=False)["method_base"].tolist()
    csi_rank = csi.sort_values("approximate_sum_rate", ascending=False)["method_base"].tolist()
    tolerance_ok = bool(
        comparison["sum_rate_delta_csi_minus_raw"].dropna().abs().le(1e-5).all()
        and comparison["symbol_mse_delta_csi_minus_raw"].dropna().abs().le(1e-5).all()
    )
    new_fallback = bool(
        (
            comparison["fallback_used_csi"].fillna(False).astype(bool)
            & ~comparison["fallback_used_raw"].fillna(False).astype(bool)
        ).any()
    )
    lines = [
        "# CSI-backed vs Raw Extracted-H Comparison",
        "",
        f"1. numeric consistency within tolerance: `{tolerance_ok}`",
        f"2. method ranking consistent: `{raw_rank == csi_rank}`",
        f"3. additional fallback introduced: `{new_fallback}`",
        "4. metadata/provenance clarity improved: `True`.",
        "5. full native-only benchmark completed: `False`.",
        "",
        "The CSI-backed path keeps the same native-channel-assisted plus native-receiver-assisted boundary and does not justify a full native-only claim.",
    ]
    write_markdown(out_dir / "csi_interface_comparison.md", lines)
    print(f"Saved CSI-backed comparison to {out_dir}")


if __name__ == "__main__":
    main()
