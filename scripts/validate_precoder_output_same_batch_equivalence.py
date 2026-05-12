#!/usr/bin/env python
"""Validate raw-F_f and PrecoderOutput paths on one deterministic shared batch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.precoder_interface import compare_precoder_outputs, create_shared_precoder_output_batch
from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import clone_native_receiver_context, run_native_receiver_with_precoder


TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _method_rank(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: row[key], reverse=True)
    return {row["method"]: idx + 1 for idx, row in enumerate(ordered)}


def _md(summary: dict[str, Any]) -> list[str]:
    return [
        "# PrecoderOutput Same-batch Equivalence",
        "",
        f"- same_csi_object_used: `{summary['same_csi_object_used']}`",
        f"- same_raw_f_f_used: `{summary['same_raw_f_f_used']}`",
        f"- same_bits_used: `{summary['same_bits_used']}`",
        f"- same_noise_config_used: `{summary['same_noise_config_used']}`",
        f"- same_receiver_config_used: `{summary['same_receiver_config_used']}`",
        f"- precoder_output_f_f_matches_raw: `{summary['precoder_output_f_f_matches_raw']}`",
        f"- numeric_consistency_within_tolerance: `{summary['numeric_consistency_within_tolerance']}`",
        f"- ranking_consistent: `{summary['ranking_consistent']}`",
        f"- max_abs_diff_raw_f_f_vs_precoder_output: `{summary['max_abs_diff_raw_f_f_vs_precoder_output']}`",
        f"- max_abs_diff_sum_rate: `{summary['max_abs_diff_sum_rate']}`",
        f"- max_abs_diff_symbol_mse: `{summary['max_abs_diff_symbol_mse']}`",
        f"- max_abs_diff_sinr_db: `{summary['max_abs_diff_sinr_db']}`",
        "",
        f"- root_cause_if_failed: `{summary['root_cause_if_failed']}`",
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    csv_path = out_path.with_suffix(".csv")
    env = collect_sionna_env_info()
    summary: dict[str, Any] = {
        "status": "skipped",
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "seed": int(args.seed),
        "same_csi_object_used": False,
        "same_raw_f_f_used": False,
        "same_bits_used": False,
        "same_noise_config_used": False,
        "same_receiver_config_used": False,
        "precoder_output_f_f_matches_raw": False,
        "numeric_consistency_within_tolerance": False,
        "ranking_consistent": False,
        "max_abs_diff_raw_f_f_vs_precoder_output": None,
        "max_abs_diff_sum_rate": None,
        "max_abs_diff_symbol_mse": None,
        "max_abs_diff_sinr_db": None,
        "strict_equivalence_claim_allowed": False,
        "root_cause_if_failed": "",
        "rows": [],
        "notes": [],
    }
    if not env["sionna_import_ok"]:
        write_json(out_path, summary)
        write_markdown(md_path, _md(summary))
        print(f"Saved same-batch equivalence summary to {out_path}")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]
    shared = create_shared_precoder_output_batch(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
        repo_root=repo_root,
        seed=args.seed,
    )
    base_context = shared.native_context
    if base_context is None:
        raise RuntimeError("shared_precoder_output_batch_missing_native_context")

    rows: list[dict[str, Any]] = []
    for method, artifacts in shared.method_artifacts.items():
        method_context = clone_native_receiver_context(
            base_context,
            h_f=shared.raw_h_f,
            csi=shared.csi,
            h_full=shared.h_full,
            context_meta_updates={
                "shared_rx_noise_grid": shared.rx_noise_grid,
                "csi_interface_used": True,
                "project_h_f_assisted": False,
                "extracted_h_f_used": True,
                "full_native_only": False,
                "csi_summary": shared.csi.summary_dict(),
                "receiver_config": shared.receiver_config,
            },
        )
        raw_row, _, _ = run_native_receiver_with_precoder(
            method=f"{method}_raw",
            method_type=artifacts.method_type,
            precoder_f=artifacts.raw_f_f,
            context=method_context,
            runtime_ms=artifacts.runtime_ms,
            checkpoint_path=artifacts.checkpoint_path,
            teacher_used_during_inference=artifacts.teacher_used_during_inference,
            trace_shapes=False,
        )
        precoder_row, _, _ = run_native_receiver_with_precoder(
            method=f"{method}_precoder_output",
            method_type=artifacts.method_type,
            precoder_f=artifacts.precoder_output,
            context=method_context,
            runtime_ms=artifacts.runtime_ms,
            checkpoint_path=artifacts.checkpoint_path,
            teacher_used_during_inference=artifacts.teacher_used_during_inference,
            trace_shapes=False,
        )
        comparison = compare_precoder_outputs(artifacts.raw_f_f, artifacts.precoder_output)
        row = {
            "method": method,
            "same_csi_object_used": True,
            "same_raw_f_f_used": True,
            "same_bits_used": True,
            "same_noise_config_used": True,
            "same_receiver_config_used": True,
            "max_abs_diff_raw_f_f_vs_precoder_output": float(comparison["max_abs_diff"] or 0.0),
            "raw_sum_rate": float(raw_row["approximate_sum_rate"]),
            "precoder_output_sum_rate": float(precoder_row["approximate_sum_rate"]),
            "abs_diff_sum_rate": float(abs(raw_row["approximate_sum_rate"] - precoder_row["approximate_sum_rate"])),
            "raw_symbol_mse": float(raw_row["symbol_mse"]),
            "precoder_output_symbol_mse": float(precoder_row["symbol_mse"]),
            "abs_diff_symbol_mse": float(abs(raw_row["symbol_mse"] - precoder_row["symbol_mse"])),
            "raw_effective_sinr_db": float(raw_row["effective_sinr_db"]),
            "precoder_output_effective_sinr_db": float(precoder_row["effective_sinr_db"]),
            "abs_diff_sinr_db": float(abs(raw_row["effective_sinr_db"] - precoder_row["effective_sinr_db"])),
            "ranking_raw": None,
            "ranking_precoder_output": None,
            "ranking_consistent": None,
            "strict_equivalence_claim_allowed": False,
            "teacher_used_during_inference": bool(artifacts.teacher_used_during_inference),
            "checkpoint_path": artifacts.checkpoint_path,
        }
        rows.append(row)

    rank_raw = _method_rank(rows, "raw_sum_rate")
    rank_precoder = _method_rank(rows, "precoder_output_sum_rate")
    for row in rows:
        row["ranking_raw"] = rank_raw[row["method"]]
        row["ranking_precoder_output"] = rank_precoder[row["method"]]
        row["ranking_consistent"] = row["ranking_raw"] == row["ranking_precoder_output"]
        row["strict_equivalence_claim_allowed"] = bool(
            row["max_abs_diff_raw_f_f_vs_precoder_output"] <= TOL
            and row["abs_diff_sum_rate"] <= TOL
            and row["abs_diff_symbol_mse"] <= TOL
            and row["abs_diff_sinr_db"] <= TOL
            and row["ranking_consistent"]
        )

    summary["status"] = "ok"
    summary["rows"] = rows
    summary["same_csi_object_used"] = True
    summary["same_raw_f_f_used"] = True
    summary["same_bits_used"] = True
    summary["same_noise_config_used"] = True
    summary["same_receiver_config_used"] = True
    summary["max_abs_diff_raw_f_f_vs_precoder_output"] = max(
        (row["max_abs_diff_raw_f_f_vs_precoder_output"] for row in rows),
        default=None,
    )
    summary["max_abs_diff_sum_rate"] = max((row["abs_diff_sum_rate"] for row in rows), default=None)
    summary["max_abs_diff_symbol_mse"] = max((row["abs_diff_symbol_mse"] for row in rows), default=None)
    summary["max_abs_diff_sinr_db"] = max((row["abs_diff_sinr_db"] for row in rows), default=None)
    summary["precoder_output_f_f_matches_raw"] = bool(
        rows and all(row["max_abs_diff_raw_f_f_vs_precoder_output"] <= TOL for row in rows)
    )
    summary["numeric_consistency_within_tolerance"] = bool(
        rows
        and all(
            row["abs_diff_sum_rate"] <= TOL
            and row["abs_diff_symbol_mse"] <= TOL
            and row["abs_diff_sinr_db"] <= TOL
            for row in rows
        )
    )
    summary["ranking_consistent"] = bool(rows and all(row["ranking_consistent"] for row in rows))
    summary["strict_equivalence_claim_allowed"] = bool(rows and all(row["strict_equivalence_claim_allowed"] for row in rows))
    if not summary["strict_equivalence_claim_allowed"]:
        if not summary["precoder_output_f_f_matches_raw"]:
            summary["root_cause_if_failed"] = "precoder_output_f_f_mismatch"
        elif not summary["numeric_consistency_within_tolerance"]:
            summary["root_cause_if_failed"] = "native_receiver_metric_mismatch"
        elif not summary["ranking_consistent"]:
            summary["root_cause_if_failed"] = "method_ranking_mismatch"
        else:
            summary["root_cause_if_failed"] = "unknown_same_batch_equivalence_failure"
        summary["notes"].append("Inspect per-method diffs in CSV for the exact failure stage.")

    _write_csv(csv_path, rows if rows else [{"method": "none"}])
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved same-batch equivalence summary to {out_path}")


if __name__ == "__main__":
    main()
