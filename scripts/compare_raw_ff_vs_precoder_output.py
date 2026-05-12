#!/usr/bin/env python
"""Compare legacy raw-F_f metrics with the PrecoderOutput-backed unified demo."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import pandas as pd

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--precoder-output", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _normalize_method(method: str) -> str:
    return method.removesuffix("_from_extracted_h")


def _parse_precoder_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.raw)
    precoder_path = Path(args.precoder_output)
    raw = pd.read_csv(raw_path).copy()
    precoder = pd.read_csv(precoder_path).copy()
    raw_summary_path = raw_path.with_name("csi_backed_beamforming_summary.json")
    precoder_summary_path = precoder_path.with_name("unified_csi_precoder_summary.json")
    raw_summary = json.loads(raw_summary_path.read_text(encoding="utf-8")) if raw_summary_path.exists() else {}
    precoder_summary = json.loads(precoder_summary_path.read_text(encoding="utf-8")) if precoder_summary_path.exists() else {}

    raw["method_base"] = raw["method"].map(_normalize_method)
    precoder["method_base"] = precoder["method"].map(_normalize_method)

    raw_precoder_signatures = {
        row["method_base"]: (_parse_precoder_summary(row.get("precoder_summary")) or {}).get("tensor_signature")
        for _, row in raw.iterrows()
    }
    precoder_signatures = {
        row["method_base"]: (_parse_precoder_summary(row.get("precoder_summary")) or {}).get("tensor_signature")
        for _, row in precoder.iterrows()
    }
    same_seed_used = raw_summary.get("seed") == precoder_summary.get("seed")
    same_csi_object_used = (raw_summary.get("csi_input_summary") or {}).get("tensor_signature") == (
        (precoder_summary.get("csi_input_summary") or {}).get("tensor_signature")
    )
    shared_methods = sorted(set(raw_precoder_signatures) & set(precoder_signatures))
    same_raw_f_f_used = bool(shared_methods) and all(
        raw_precoder_signatures.get(method) == precoder_signatures.get(method) for method in shared_methods
    )

    comparison = raw.merge(
        precoder[
            [
                "method_base",
                "approximate_sum_rate",
                "symbol_mse",
                "effective_sinr_db",
                "native_receiver_success",
                "fallback_used",
                "fallback_reason",
                "precoder_interface_used",
                "precoder_input_type",
                "precoder_source",
                "project_side_precoder",
                "sionna_native_precoder",
                "full_native_only",
            ]
        ],
        on="method_base",
        how="outer",
        suffixes=("_raw", "_precoder_output"),
    )
    comparison["sum_rate_abs_diff"] = (
        comparison["approximate_sum_rate_precoder_output"] - comparison["approximate_sum_rate_raw"]
    ).abs()
    comparison["symbol_mse_abs_diff"] = (
        comparison["symbol_mse_precoder_output"] - comparison["symbol_mse_raw"]
    ).abs()
    comparison["sinr_db_abs_diff"] = (
        comparison["effective_sinr_db_precoder_output"] - comparison["effective_sinr_db_raw"]
    ).abs()
    comparison["comparison_type"] = "cross_run_comparison"
    comparison["same_batch_comparison"] = False
    comparison["not_strict_equivalence_test"] = True
    comparison["strict_equivalence_claim_allowed"] = False
    comparison["same_csi_object_used"] = same_csi_object_used
    comparison["same_raw_f_f_used"] = same_raw_f_f_used
    comparison["same_seed_used"] = same_seed_used
    comparison["equivalence_claim_allowed"] = False
    comparison["interface_bug_evidence"] = False
    comparison.to_csv(out_dir / "precoder_output_comparison.csv", index=False)

    raw_rank = raw.sort_values("approximate_sum_rate", ascending=False)["method_base"].tolist()
    precoder_rank = precoder.sort_values("approximate_sum_rate", ascending=False)["method_base"].tolist()
    additional_fallback = bool(
        (
            comparison["fallback_used_precoder_output"].fillna(False).astype(bool)
            & ~comparison["fallback_used_raw"].fillna(False).astype(bool)
        ).any()
    )
    lines = [
        "# Raw F_f vs PrecoderOutput Comparison",
        "",
        "- comparison_type: `cross_run_comparison`",
        "- same_batch_comparison: `false`",
        "- not_strict_equivalence_test: `true`",
        f"- same_seed_used: `{same_seed_used}`",
        f"- same_csi_object_used: `{same_csi_object_used}`",
        f"- same_raw_f_f_used: `{same_raw_f_f_used}`",
        "- strict_equivalence_claim_allowed: `false`",
        "- equivalence_claim_allowed: `false`",
        "- interface_bug_evidence: `false`",
        "",
        "1. same-batch comparison: `False`.",
        "2. cross-run comparison caveat: `True`.",
        f"3. additional fallback introduced: `{additional_fallback}`.",
        f"4. method ranking consistent: `{raw_rank == precoder_rank}`.",
        "5. strict equivalence claim allowed: `False`.",
        "6. full native-only benchmark completed: `False`.",
        "",
        "This artifact compares separate reruns. It should not be interpreted as a same-batch or strict numerical equivalence test, and ranking mismatch here is not treated as PrecoderOutput bug evidence.",
    ]
    write_markdown(out_dir / "precoder_output_comparison.md", lines)
    print(f"Saved raw-vs-PrecoderOutput comparison to {out_dir}")


if __name__ == "__main__":
    main()
