#!/usr/bin/env python
"""Validate raw-H and CSI-backed paths on the same deterministic shared batch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.csi_interface import tensor_signature
from beamforming.utils.seed import set_seed
from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import compute_project_precoder_per_subcarrier
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import (
    build_native_receiver_context,
    clone_native_receiver_context,
    default_checkpoint_path,
    generate_shared_sionna_channel_bundle,
    infer_learned_precoder,
    load_learned_beamformer_checkpoint,
    run_native_receiver_with_precoder,
)


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
        "# CSI Same-batch Equivalence",
        "",
        f"- same_channel_tensor_used: `{summary['same_channel_tensor_used']}`",
        f"- same_bits_used: `{summary['same_bits_used']}`",
        f"- same_noise_config_used: `{summary['same_noise_config_used']}`",
        f"- same_receiver_config_used: `{summary['same_receiver_config_used']}`",
        f"- numeric_consistency_within_tolerance: `{summary['numeric_consistency_within_tolerance']}`",
        f"- ranking_consistent: `{summary['ranking_consistent']}`",
        f"- max_abs_diff_sum_rate: `{summary['max_abs_diff_sum_rate']}`",
        f"- max_abs_diff_symbol_mse: `{summary['max_abs_diff_symbol_mse']}`",
        f"- max_abs_diff_sinr_db: `{summary['max_abs_diff_sinr_db']}`",
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
        "same_channel_tensor_used": False,
        "same_bits_used": False,
        "same_noise_config_used": False,
        "same_receiver_config_used": False,
        "numeric_consistency_within_tolerance": False,
        "ranking_consistent": False,
        "max_abs_diff_sum_rate": None,
        "max_abs_diff_symbol_mse": None,
        "max_abs_diff_sinr_db": None,
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
    channel_bundle = generate_shared_sionna_channel_bundle(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        noise_var=float(10.0 ** (-10.0 / 10.0)),
        device=device,
        seed=args.seed,
    )
    raw_context = build_native_receiver_context(
        batch_size=16,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
        channel_bundle=channel_bundle,
    )
    csi_context = clone_native_receiver_context(
        raw_context,
        h_f=channel_bundle.csi.to_project_h_f() if channel_bundle.csi is not None else raw_context.h_f,
        csi=channel_bundle.csi,
        h_full=raw_context.h_full,
        context_meta_updates={
            "csi_interface_used": True,
            "csi_summary": channel_bundle.csi.summary_dict() if channel_bundle.csi is not None else None,
        },
    )

    methods = [
        ("project_rzf", "analytic"),
        ("project_wmmse_iter_5", "analytic"),
        ("learned_residual_rzf", "learned"),
        ("learned_residual_wmmse_distill", "learned"),
    ]
    rows: list[dict[str, Any]] = []
    for method, method_type in methods:
        checkpoint_path = None
        teacher_flag = False
        runtime_ms = 0.0
        if method_type == "analytic":
            raw_precoder = compute_project_precoder_per_subcarrier(method.removeprefix("project_"), raw_context.h_f, raw_context.noise_var)
            csi_precoder = compute_project_precoder_per_subcarrier(method.removeprefix("project_"), csi_context.csi.to_project_h_f(), csi_context.noise_var)
        else:
            ckpt = default_checkpoint_path(method, repo_root)
            if not ckpt.exists():
                continue
            bundle = load_learned_beamformer_checkpoint(ckpt, device, method_name=method)
            snr_tensor = torch.full((raw_context.h_f.size(0),), raw_context.snr_db, dtype=torch.float32, device=device)
            raw_precoder, infer_meta, runtime_ms = infer_learned_precoder(bundle, raw_context.h_f, snr_tensor, native_receiver_path=True)
            csi_precoder, _, _ = infer_learned_precoder(bundle, csi_context.csi.to_project_h_f(), snr_tensor, native_receiver_path=True)
            checkpoint_path = str(ckpt)
            teacher_flag = bool(infer_meta["teacher_used_during_inference"])

        raw_row, _, _ = run_native_receiver_with_precoder(
            method=f"{method}_raw",
            method_type=method_type,
            precoder_f=raw_precoder,
            context=raw_context,
            runtime_ms=runtime_ms,
            checkpoint_path=checkpoint_path,
            teacher_used_during_inference=teacher_flag,
            trace_shapes=False,
        )
        csi_row, _, _ = run_native_receiver_with_precoder(
            method=f"{method}_csi",
            method_type=method_type,
            precoder_f=csi_precoder,
            context=csi_context,
            runtime_ms=runtime_ms,
            checkpoint_path=checkpoint_path,
            teacher_used_during_inference=teacher_flag,
            trace_shapes=False,
        )
        rows.append(
            {
                "method": method,
                "raw_sum_rate": float(raw_row["approximate_sum_rate"]),
                "csi_sum_rate": float(csi_row["approximate_sum_rate"]),
                "abs_diff_sum_rate": float(abs(raw_row["approximate_sum_rate"] - csi_row["approximate_sum_rate"])),
                "rel_diff_sum_rate": float(abs(raw_row["approximate_sum_rate"] - csi_row["approximate_sum_rate"]) / max(abs(raw_row["approximate_sum_rate"]), 1e-12)),
                "raw_symbol_mse": float(raw_row["symbol_mse"]),
                "csi_symbol_mse": float(csi_row["symbol_mse"]),
                "abs_diff_symbol_mse": float(abs(raw_row["symbol_mse"] - csi_row["symbol_mse"])),
                "raw_effective_sinr_db": float(raw_row["effective_sinr_db"]),
                "csi_effective_sinr_db": float(csi_row["effective_sinr_db"]),
                "abs_diff_sinr_db": float(abs(raw_row["effective_sinr_db"] - csi_row["effective_sinr_db"])),
                "ranking_raw": None,
                "ranking_csi": None,
                "ranking_consistent": None,
                "same_channel_tensor_used": tensor_signature(raw_context.h_full) == tensor_signature(csi_context.h_full),
                "same_bits_used": tensor_signature(raw_context.bits) == tensor_signature(csi_context.bits),
                "same_noise_config_used": raw_context.noise_var == csi_context.noise_var and tensor_signature(raw_context.context_meta.get("shared_rx_noise_grid")) == tensor_signature(csi_context.context_meta.get("shared_rx_noise_grid")),
                "same_receiver_config_used": raw_context.resource_grid.num_ofdm_symbols == csi_context.resource_grid.num_ofdm_symbols and raw_context.resource_grid.fft_size == csi_context.resource_grid.fft_size,
            }
        )

    rank_raw = _method_rank(rows, "raw_sum_rate")
    rank_csi = _method_rank(rows, "csi_sum_rate")
    for row in rows:
        row["ranking_raw"] = rank_raw[row["method"]]
        row["ranking_csi"] = rank_csi[row["method"]]
        row["ranking_consistent"] = row["ranking_raw"] == row["ranking_csi"]

    summary["status"] = "ok"
    summary["rows"] = rows
    summary["same_channel_tensor_used"] = all(row["same_channel_tensor_used"] for row in rows) if rows else False
    summary["same_bits_used"] = all(row["same_bits_used"] for row in rows) if rows else False
    summary["same_noise_config_used"] = all(row["same_noise_config_used"] for row in rows) if rows else False
    summary["same_receiver_config_used"] = all(row["same_receiver_config_used"] for row in rows) if rows else False
    summary["max_abs_diff_sum_rate"] = max((row["abs_diff_sum_rate"] for row in rows), default=None)
    summary["max_abs_diff_symbol_mse"] = max((row["abs_diff_symbol_mse"] for row in rows), default=None)
    summary["max_abs_diff_sinr_db"] = max((row["abs_diff_sinr_db"] for row in rows), default=None)
    summary["numeric_consistency_within_tolerance"] = bool(
        rows
        and all(row["abs_diff_sum_rate"] <= 1e-6 and row["abs_diff_symbol_mse"] <= 1e-6 and row["abs_diff_sinr_db"] <= 1e-6 for row in rows)
    )
    summary["ranking_consistent"] = bool(rows and all(row["ranking_consistent"] for row in rows))
    if not summary["numeric_consistency_within_tolerance"]:
        summary["notes"].append("Differences remain on same-batch comparison; inspect per-method diffs in CSV.")

    _write_csv(csv_path, rows if rows else [{"method": "none"}])
    write_json(out_path, summary)
    write_markdown(md_path, _md(summary))
    print(f"Saved same-batch equivalence summary to {out_path}")


if __name__ == "__main__":
    main()
